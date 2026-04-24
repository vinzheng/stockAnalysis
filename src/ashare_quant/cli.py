from __future__ import annotations

import argparse
import os
import json
import sys
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
from pandas.tseries.offsets import BDay

from ashare_quant.backtest import run_single_symbol_backtest
from ashare_quant.config import compute_config_fingerprint, load_config
from ashare_quant.data import MarketDataClient, resolve_date_window
from ashare_quant.strategy import add_signal_columns, latest_signal_summary

PRIORITY_ORDER = {"快退": 0, "卖出": 1, "买入": 2, "补仓": 3, "观察": 4}
REGIME_LEVELS = {"风险关": 0, "中性": 1, "风险开": 2, "未知": -1, "未启用": 2}
REGIME_DISPLAY_LABELS = {"风险开": "强势环境", "中性": "中性环境", "风险关": "弱势环境"}
SCAN_OUTPUT_PATH = Path("data") / "latest_scan.csv"
STALE_SCAN_PATH = Path("data") / "latest_scan_stale.csv"
PARTIAL_SCAN_PATH = Path("data") / "latest_scan_partial.csv"
SCAN_LOCK_PATH = Path("data") / ".scan.lock"
NON_PARTIAL_SCAN_CACHE_PATHS = (SCAN_OUTPUT_PATH, STALE_SCAN_PATH)
NON_PARTIAL_SCAN_RETENTION_TRADING_DAYS = 15
POST_CLOSE_HOUR = 16
SCAN_LOCK_STALE_SECONDS = 6 * 60 * 60


class DataNotReadyError(RuntimeError):
    """Raised when strict post-close mode requires fresher daily bars."""


class ScanInProgressError(RuntimeError):
    """Raised when another scan process is already running."""


def format_market_regime_label(regime: object) -> str:
    cleaned = str(regime or "").strip()
    if not cleaned:
        return "未知"
    return REGIME_DISPLAY_LABELS.get(cleaned, cleaned)


def format_market_regime_text(text: object) -> str:
    formatted = str(text or "").strip()
    if not formatted:
        return ""
    for raw_label, display_label in REGIME_DISPLAY_LABELS.items():
        formatted = formatted.replace(raw_label, display_label)
    return formatted


def format_scan_output_for_cli(result: pd.DataFrame) -> pd.DataFrame:
    if result.empty:
        return result

    display_df = result.copy()
    if "market_regime" in display_df.columns:
        display_df["market_regime"] = display_df["market_regime"].map(format_market_regime_label)
    for column in ["market_reason", "execution_advice"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(format_market_regime_text)
    return display_df


def latest_completed_market_day(reference: datetime | None = None) -> date:
    current = pd.Timestamp(reference or datetime.now())
    if current.weekday() >= 5:
        return (current - BDay(1)).date()
    if current.hour >= POST_CLOSE_HOUR:
        return current.date()
    return (current - BDay(1)).date()


def resolve_scan_target_date(as_of: str | None, reference: datetime | None = None) -> date:
    if as_of:
        return datetime.strptime(as_of, "%Y-%m-%d").date()
    return latest_completed_market_day(reference)


def get_scan_metadata_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".meta.json")


def load_scan_metadata(csv_path: Path) -> dict[str, object] | None:
    metadata_path = get_scan_metadata_path(csv_path)
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def get_scan_signal_date(csv_path: Path) -> date | None:
    if not csv_path.exists():
        return None
    try:
        scan_df = pd.read_csv(csv_path, usecols=["date"])
    except Exception:
        return None
    if scan_df.empty or "date" not in scan_df.columns:
        return None
    latest_signal = pd.to_datetime(scan_df["date"], errors="coerce").max()
    if pd.isna(latest_signal):
        return None
    return latest_signal.date()


def count_trading_day_lag(signal_date: date | None, reference_date: date | None) -> int | None:
    if signal_date is None or reference_date is None:
        return None
    if signal_date >= reference_date:
        return 0
    trading_days = pd.date_range(
        start=pd.Timestamp(signal_date) + pd.Timedelta(days=1),
        end=pd.Timestamp(reference_date),
        freq="B",
    )
    return len(trading_days)


def purge_expired_non_partial_scan_cache(
    reference_date: date | None = None,
    retention_trading_days: int = NON_PARTIAL_SCAN_RETENTION_TRADING_DAYS,
) -> list[Path]:
    target_date = reference_date or latest_completed_market_day()
    deleted_paths: list[Path] = []
    for csv_path in NON_PARTIAL_SCAN_CACHE_PATHS:
        signal_date = get_scan_signal_date(csv_path)
        trading_day_lag = count_trading_day_lag(signal_date, target_date)
        if trading_day_lag is None or trading_day_lag <= retention_trading_days:
            continue
        metadata_path = get_scan_metadata_path(csv_path)
        for path in (csv_path, metadata_path):
            if not path.exists():
                continue
            try:
                path.unlink()
                deleted_paths.append(path)
            except OSError:
                continue
    return deleted_paths


def _read_scan_lock() -> dict[str, object] | None:
    if not SCAN_LOCK_PATH.exists():
        return None
    try:
        return json.loads(SCAN_LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _remove_scan_lock_if_stale() -> None:
    lock_data = _read_scan_lock()
    if lock_data is None:
        return
    created_at_raw = lock_data.get("created_at")
    try:
        created_at = datetime.fromisoformat(str(created_at_raw))
    except (TypeError, ValueError):
        created_at = None
    if created_at is None or (datetime.now() - created_at).total_seconds() > SCAN_LOCK_STALE_SECONDS:
        try:
            SCAN_LOCK_PATH.unlink()
        except OSError:
            pass


@contextmanager
def acquire_scan_lock(as_of: str | None, top_n: int) -> None:
    SCAN_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    _remove_scan_lock_if_stale()
    payload = {
        "pid": os.getpid(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "as_of": as_of,
        "top_n": top_n,
    }
    try:
        fd = os.open(str(SCAN_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        lock_data = _read_scan_lock() or {}
        raise ScanInProgressError(
            "已有扫描任务正在运行，已阻止并发写入结果文件。"
            f" 当前锁信息: pid={lock_data.get('pid', 'unknown')} created_at={lock_data.get('created_at', 'unknown')}"
        ) from error

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
            json.dump(payload, lock_file, ensure_ascii=False, indent=2)
        yield
    finally:
        try:
            SCAN_LOCK_PATH.unlink()
        except OSError:
            pass


def write_scan_output(
    output: pd.DataFrame,
    csv_path: Path,
    config_path: str,
    scan_source: str,
    as_of: str | None,
    top_n: int,
) -> None:
    output.to_csv(csv_path, index=False, encoding="utf-8-sig")
    metadata = {
        "config_fingerprint": compute_config_fingerprint(config_path),
        "config_path": config_path,
        "scan_source": scan_source,
        "as_of": as_of,
        "top_n": top_n,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    get_scan_metadata_path(csv_path).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def evaluate_market_breadth(rows: list[dict[str, object]], config: dict) -> tuple[int, float, str]:
    timing_cfg = config.get("market_timing", {})
    if not timing_cfg.get("enabled", False) or not rows:
        return REGIME_LEVELS["未启用"], 0.0, "未启用"

    trend_count = sum(1 for row in rows if bool(row.get("trend_filter_passed", False)))
    breadth = trend_count / len(rows)
    risk_on_threshold = float(timing_cfg.get("min_trend_breadth_risk_on", 0.55))
    neutral_threshold = float(timing_cfg.get("min_trend_breadth_neutral", 0.45))
    if breadth >= risk_on_threshold:
        return REGIME_LEVELS["风险开"], breadth, "风险开"
    if breadth >= neutral_threshold:
        return REGIME_LEVELS["中性"], breadth, "中性"
    return REGIME_LEVELS["风险关"], breadth, "风险关"


def percentile_rank(series: pd.Series) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce")
    if numeric_series.notna().sum() == 0:
        return pd.Series(0.0, index=series.index, dtype=float)
    return numeric_series.rank(pct=True, method="average").fillna(0.0)


def build_candidate_pool(snapshot: pd.DataFrame, market: dict[str, object]) -> pd.DataFrame:
    filtered_snapshot = snapshot.loc[
        (snapshot["last_price"] >= float(market["min_price"]))
        & (snapshot["turnover_amount"] >= float(market["min_turnover_amount"]))
    ].copy()

    min_turnover_rate = float(market.get("min_turnover_rate", 0) or 0)
    if min_turnover_rate > 0 and "turnover_rate" in filtered_snapshot.columns:
        filtered_snapshot["turnover_rate"] = pd.to_numeric(filtered_snapshot["turnover_rate"], errors="coerce").fillna(0.0)
        filtered_snapshot = filtered_snapshot.loc[filtered_snapshot["turnover_rate"] >= min_turnover_rate].copy()

    selection_mode = str(market.get("candidate_selection_mode", "turnover_amount")).strip().lower()
    if filtered_snapshot.empty:
        return filtered_snapshot

    if selection_mode == "turnover_amount":
        filtered_snapshot["candidate_pool_score"] = pd.to_numeric(
            filtered_snapshot["turnover_amount"],
            errors="coerce",
        ).fillna(0.0)
    else:
        weights = market.get("candidate_selection_weights", {}) if isinstance(market.get("candidate_selection_weights", {}), dict) else {}
        turnover_amount_rank = percentile_rank(filtered_snapshot["turnover_amount"])
        turnover_rate_rank = percentile_rank(filtered_snapshot.get("turnover_rate", pd.Series(index=filtered_snapshot.index, dtype=float)).fillna(0.0))
        amplitude_rank = percentile_rank(filtered_snapshot.get("amplitude", pd.Series(index=filtered_snapshot.index, dtype=float)).fillna(0.0))
        pct_change_rank = percentile_rank(
            pd.to_numeric(filtered_snapshot.get("pct_change", pd.Series(index=filtered_snapshot.index, dtype=float)), errors="coerce").abs().fillna(0.0)
        )
        filtered_snapshot["candidate_pool_score"] = (
            turnover_amount_rank * float(weights.get("turnover_amount", 0.4))
            + turnover_rate_rank * float(weights.get("turnover_rate", 0.35))
            + amplitude_rank * float(weights.get("amplitude", 0.15))
            + pct_change_rank * float(weights.get("pct_change_abs", 0.10))
        )

    return filtered_snapshot.sort_values(
        ["candidate_pool_score", "turnover_amount"],
        ascending=[False, False],
    )


def evaluate_index_regime(client: MarketDataClient, config: dict, end_date: datetime) -> tuple[int | None, str, str]:
    timing_cfg = config.get("market_timing", {})
    if not timing_cfg.get("enabled", False) or not timing_cfg.get("use_index_filter", False):
        return None, "未启用", "指数过滤未启用"

    raw_benchmark_symbol = str(timing_cfg.get("benchmark_symbol", "000300")).strip()
    benchmark_symbol = raw_benchmark_symbol.zfill(6) if raw_benchmark_symbol.isdigit() else raw_benchmark_symbol
    long_window = int(timing_cfg.get("benchmark_long_ma", 60))
    short_window = int(timing_cfg.get("benchmark_short_ma", 20))
    end_timestamp = pd.Timestamp(end_date)
    start_date = end_timestamp - pd.Timedelta(days=long_window * 3)
    try:
        history = client.get_index_history(benchmark_symbol, start_date.to_pydatetime(), end_timestamp.to_pydatetime())
    except RuntimeError:
        return None, "未知", f"指数 {benchmark_symbol} 获取失败，已回退到宽度过滤"

    if history.empty or len(history) < long_window:
        return None, "未知", f"指数 {benchmark_symbol} 数据不足，已回退到宽度过滤"

    history = history.copy()
    history["ma_short"] = history["close"].rolling(short_window).mean()
    history["ma_long"] = history["close"].rolling(long_window).mean()
    latest = history.iloc[-1]
    if bool(latest["close"] > latest["ma_short"] > latest["ma_long"]):
        return REGIME_LEVELS["风险开"], "风险开", f"指数 {benchmark_symbol} 收于 MA{short_window}/MA{long_window} 上方"
    if bool(latest["close"] > latest["ma_long"] and latest["ma_short"] >= latest["ma_long"]):
        return REGIME_LEVELS["中性"], "中性", f"指数 {benchmark_symbol} 站上 MA{long_window}，但未形成强势多头"
    return REGIME_LEVELS["风险关"], "风险关", f"指数 {benchmark_symbol} 未站稳 MA{long_window}"


def evaluate_market_regime(client: MarketDataClient, rows: list[dict[str, object]], config: dict, end_date: datetime) -> dict[str, object]:
    breadth_level, breadth, breadth_regime = evaluate_market_breadth(rows, config)
    index_level, index_regime, index_reason = evaluate_index_regime(client, config, end_date)

    if index_level is None:
        regime_level = breadth_level
        regime = breadth_regime
        source = "宽度过滤"
        reason = f"{index_reason}；趋势广度 {breadth:.0%}"
    else:
        regime_level = min(breadth_level, index_level)
        regime = next((name for name, level in REGIME_LEVELS.items() if level == regime_level and name in {"风险关", "中性", "风险开"}), "未知")
        source = "指数+宽度"
        reason = f"指数状态 {index_regime}；{index_reason}；趋势广度 {breadth:.0%}"

    allow_new_buy = regime_level >= REGIME_LEVELS["风险开"]
    allow_add_on = regime_level >= REGIME_LEVELS["中性"]

    return {
        "market_filter_passed": allow_new_buy,
        "market_add_on_passed": allow_add_on,
        "market_breadth": breadth,
        "market_regime_level": regime_level,
        "market_regime": regime,
        "market_filter_source": source,
        "market_reason": reason,
    }


def apply_execution_priority(summary: dict[str, object], market_context: dict[str, object]) -> dict[str, object]:
    action = str(summary.get("action", "观察"))
    market_filter_passed = bool(market_context["market_filter_passed"])
    market_add_on_passed = bool(market_context["market_add_on_passed"])
    market_regime = str(market_context["market_regime"])
    market_breadth = float(market_context["market_breadth"])
    if bool(summary.get("sell_signal")):
        priority = "快退" if summary.get("sell_reason") == "突破失败快退" else "卖出"
    elif bool(summary.get("buy_signal")):
        priority = "买入" if market_filter_passed else "观察"
    elif bool(summary.get("add_on_signal")):
        priority = "补仓" if market_add_on_passed else "观察"
    else:
        priority = action if action == "卖出" else "观察"

    if action == "买入" and not market_filter_passed:
        if market_regime == "中性":
            summary["execution_advice"] = (
                f"市场环境当前为{market_regime}，趋势广度 {market_breadth:.0%}。"
                "中性环境下不建议开新仓，优先等环境重新转强。"
            )
        else:
            summary["execution_advice"] = (
                f"市场环境当前为{market_regime}，趋势广度 {market_breadth:.0%}，"
                "暂不建议开新仓，优先等待环境回暖。"
            )
    elif action == "补仓" and not market_add_on_passed:
        summary["execution_advice"] = (
            f"市场环境当前为{market_regime}，趋势广度 {market_breadth:.0%}，"
            "暂不建议补仓，优先等待环境改善。"
        )

    summary.update(
        {
            "market_filter_passed": market_filter_passed,
            "market_add_on_passed": market_add_on_passed,
            "market_regime": market_regime,
            "market_breadth": round(market_breadth, 4),
            "market_regime_level": market_context["market_regime_level"],
            "market_filter_source": market_context["market_filter_source"],
            "market_reason": market_context["market_reason"],
        }
    )
    summary["execution_priority"] = priority
    summary["execution_priority_rank"] = PRIORITY_ORDER.get(priority, 99)
    return summary


def load_scan_candidates(client: MarketDataClient, market: dict[str, object]) -> pd.DataFrame:
    snapshot = client.get_universe_snapshot()
    if market["exclude_st"]:
        snapshot = snapshot[~snapshot["name"].str.contains("ST", na=False)]
    return build_candidate_pool(snapshot, market).head(int(market["max_symbols"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A-share quant MVP")
    parser.add_argument("--config", default="config/universe.yaml", help="Path to config YAML")

    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Scan A-share market")
    scan_parser.add_argument("--as-of", default=None, help="End date, format YYYY-MM-DD")
    scan_parser.add_argument("--top", type=int, default=None, help="Override top_n in config")
    scan_parser.add_argument("--allow-stale", action="store_true", help="Allow stale daily bars in scan output")

    preheat_parser = subparsers.add_parser("preheat", help="Warm history cache for current candidate pool")
    preheat_parser.add_argument("--as-of", default=None, help="End date, format YYYY-MM-DD")
    preheat_parser.add_argument("--limit", type=int, default=None, help="Override number of candidate histories to warm")
    preheat_parser.add_argument(
        "--max-cache-lag",
        type=int,
        default=0,
        help="Reuse history cache within N trading days instead of refetching",
    )

    backtest_parser = subparsers.add_parser("backtest", help="Backtest one symbol")
    backtest_parser.add_argument("symbol", help="A-share symbol, for example 600519")
    backtest_parser.add_argument("--start", default="2022-01-01", help="Start date, format YYYY-MM-DD")
    backtest_parser.add_argument("--end", default=None, help="End date, format YYYY-MM-DD")

    return parser


def scan_market(config_path: str, as_of: str | None, top: int | None, allow_stale: bool = False) -> pd.DataFrame:
    config = load_config(config_path)
    market = config["market"]
    scan_cfg = config["scan"]
    top_n = top or scan_cfg["top_n"]
    strict_post_close_mode = bool(scan_cfg.get("strict_post_close_mode", False)) and not allow_stale

    with acquire_scan_lock(as_of, int(top_n)):
        client = MarketDataClient(Path("data"))
        candidates = load_scan_candidates(client, market)
        start_date, end_date = resolve_date_window(scan_cfg["history_days"], as_of)
        history_cache_max_trading_day_lag = int(scan_cfg.get("history_cache_max_trading_day_lag", 0) or 0)
        timeout_seconds = float(scan_cfg.get("scan_timeout_seconds", 0) or 0)
        started_at = perf_counter()
        processed_candidates = 0
        timed_out = False
        total_candidates = len(candidates)

        rows: list[dict[str, object]] = []
        for item in candidates.itertuples(index=False):
            if timeout_seconds and perf_counter() - started_at >= timeout_seconds:
                timed_out = True
                break
            try:
                history = client.get_history(
                    item.symbol,
                    start_date,
                    end_date,
                    max_cache_trading_day_lag=history_cache_max_trading_day_lag,
                )
            except RuntimeError as error:
                print(f"skip {item.symbol} {item.name}: {error}")
                processed_candidates += 1
                continue
            if history.empty or len(history) < config["strategy"]["ma_slow"]:
                processed_candidates += 1
                continue
            signal_history = add_signal_columns(history, config)
            summary = latest_signal_summary(signal_history, config)
            if not summary:
                processed_candidates += 1
                continue
            summary["trend_filter_passed"] = bool(
                signal_history.iloc[-1]["close"] > signal_history.iloc[-1]["ma_mid"]
                and signal_history.iloc[-1]["ma_fast"] > signal_history.iloc[-1]["ma_mid"]
            )
            rows.append(
                {
                    "symbol": item.symbol,
                    "name": item.name,
                    "turnover_amount": round(float(item.turnover_amount), 2),
                    **summary,
                }
            )
            processed_candidates += 1

        result = pd.DataFrame(rows)
        if result.empty:
            if timed_out:
                result.attrs["status_message"] = (
                    f"扫描超过 {timeout_seconds:.0f} 秒，当前没有足够的已完成结果。"
                    " 已保留上一次本地结果，建议稍后再刷新。"
                )
                result.attrs["is_partial"] = True
            return result

        market_context = evaluate_market_regime(client, rows, config, end_date)
        result = pd.DataFrame(
            [apply_execution_priority(dict(row), market_context) for row in rows]
        )
        if "trend_filter_passed" in result.columns:
            result = result.drop(columns=["trend_filter_passed"])

        result = result.sort_values(["execution_priority_rank", "score", "turnover_amount"], ascending=[True, False, False])
        output = result.head(top_n).reset_index(drop=True)
        output.attrs["status_message"] = None
        output.attrs["is_partial"] = False

        if timed_out:
            output.attrs["status_message"] = (
                f"扫描超过 {timeout_seconds:.0f} 秒，先展示已完成 {processed_candidates}/{total_candidates} 只候选股的部分结果。"
            )
            output.attrs["is_partial"] = True
            write_scan_output(output, PARTIAL_SCAN_PATH, config_path, "partial", as_of, top_n)
            return output

        if strict_post_close_mode:
            target_date = resolve_scan_target_date(as_of)
            latest_signal_date = pd.to_datetime(output["date"]).max().date()
            if latest_signal_date < target_date:
                write_scan_output(output, STALE_SCAN_PATH, config_path, "stale", as_of, top_n)
                raise DataNotReadyError(
                    "严格收盘后模式已启用："
                    f"请求日期 {target_date:%Y-%m-%d}，但最新完整日线仅到 {latest_signal_date:%Y-%m-%d}。"
                    " 已阻止输出次日选股建议。"
                )

        write_scan_output(output, SCAN_OUTPUT_PATH, config_path, "complete", as_of, top_n)
        return output


def preheat_history_cache(
    config_path: str,
    as_of: str | None,
    limit: int | None,
    max_cache_lag: int,
) -> str:
    config = load_config(config_path)
    market = config["market"]
    scan_cfg = config["scan"]

    client = MarketDataClient(Path("data"))
    candidates = load_scan_candidates(client, market)
    if limit is not None:
        candidates = candidates.head(limit)

    start_date, end_date = resolve_date_window(scan_cfg["history_days"], as_of)
    warmed = 0
    failed = 0
    warmed_symbols: list[str] = []
    failed_symbols: list[str] = []

    for item in candidates.itertuples(index=False):
        try:
            history = client.get_history(
                item.symbol,
                start_date,
                end_date,
                max_cache_trading_day_lag=max_cache_lag,
            )
        except RuntimeError:
            failed += 1
            failed_symbols.append(str(item.symbol))
            continue

        if history.empty:
            failed += 1
            failed_symbols.append(str(item.symbol))
            continue

        warmed += 1
        warmed_symbols.append(str(item.symbol))

    summary = (
        f"preheat_done target_date={end_date:%Y-%m-%d} requested={len(candidates)} "
        f"warmed={warmed} failed={failed} max_cache_lag={max_cache_lag}"
    )
    if warmed_symbols:
        summary = f"{summary}\nwarmed_symbols={','.join(warmed_symbols[:20])}"
    if failed_symbols:
        summary = f"{summary}\nfailed_symbols={','.join(failed_symbols[:20])}"
    return summary


def run_backtest(config_path: str, symbol: str, start: str, end: str | None) -> str:
    config = load_config(config_path)
    client = MarketDataClient(Path("data"))
    start_date, end_date = resolve_date_window(0, end)
    history = client.get_history(symbol, pd.to_datetime(start).to_pydatetime(), end_date)
    if history.empty:
        return f"{symbol} 没有获取到历史数据"

    signal_history = add_signal_columns(history, config)
    result = run_single_symbol_backtest(signal_history, symbol, config["strategy"]["atr_stop_multiple"])
    return (
        f"symbol={result.symbol} trades={result.trades} win_rate={result.win_rate:.2%} "
        f"total_return={result.total_return:.2%} max_drawdown={result.max_drawdown:.2%} "
        f"execution_basis={result.execution_basis}"
    )


def safe_print(text: object) -> None:
    message = str(text)
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.write(message.encode(encoding, errors="replace").decode(encoding, errors="replace") + "\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "scan":
        try:
            result = scan_market(args.config, args.as_of, args.top, allow_stale=args.allow_stale)
        except DataNotReadyError as error:
            safe_print(error)
            return
        except ScanInProgressError as error:
            safe_print(error)
            return
        if result.empty:
            safe_print("没有找到满足条件的标的")
            return
        safe_print(format_scan_output_for_cli(result).to_string(index=False))
        return

    if args.command == "preheat":
        safe_print(preheat_history_cache(args.config, args.as_of, args.limit, args.max_cache_lag))
        return

    if args.command == "backtest":
        safe_print(run_backtest(args.config, args.symbol, args.start, args.end))


if __name__ == "__main__":
    main()
