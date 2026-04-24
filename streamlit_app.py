from __future__ import annotations

import re
import sys
from datetime import date, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import akshare as ak
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ashare_quant.backtest import analyze_entry_slices, analyze_exit_slices, analyze_signal_statistics, run_single_symbol_backtest
from ashare_quant.cli import (
    DataNotReadyError,
    ScanInProgressError,
    load_scan_metadata,
    purge_expired_non_partial_scan_cache,
    resolve_scan_target_date,
    scan_market,
)
from ashare_quant.config import compute_config_fingerprint, load_config
from ashare_quant.data import MarketDataClient, purge_expired_history_cache
from ashare_quant.strategy import add_signal_columns, latest_signal_summary

st.set_page_config(page_title="A股量化看板", page_icon="A", layout="wide")

CONFIG_PATH = "config/universe.yaml"
DATA_DIR = Path("data")
WATCHLIST_PATH = DATA_DIR / "watchlist.txt"
FUND_WATCHLIST_PATH = DATA_DIR / "fund_watchlist.txt"
FUND_RANK_CACHE_PATH = DATA_DIR / "latest_fund_rank.csv"
MARKET_SNAPSHOT_SUMMARY_HISTORY_PATH = DATA_DIR / "market_snapshot_summary_history.csv"

PRIORITY_COLORS = {
    "快退": "#fee2e2",
    "卖出": "#fecaca",
    "买入": "#dcfce7",
    "补仓": "#fef3c7",
    "观察": "#e5e7eb",
}

PRIORITY_LABELS = ["快退", "卖出", "买入", "补仓", "观察"]

REGIME_BADGES = {
    "风险开": "#dcfce7",
    "强势环境": "#dcfce7",
    "中性": "#fef3c7",
    "中性环境": "#fef3c7",
    "风险关": "#fee2e2",
    "弱势环境": "#fee2e2",
    "未知": "#e5e7eb",
    "未启用": "#e5e7eb",
}

REGIME_DISPLAY_LABELS = {
    "风险开": "强势环境",
    "中性": "中性环境",
    "风险关": "弱势环境",
}

SCAN_FILE_SPECS = [
    (DATA_DIR / "latest_scan.csv", "complete"),
    (DATA_DIR / "latest_scan_stale.csv", "stale"),
    (DATA_DIR / "latest_scan_partial.csv", "partial"),
]

SCAN_SOURCE_PRIORITY = {"complete": 2, "stale": 1, "partial": 0}

SCAN_SOURCE_FILES = {
    "complete": "data/latest_scan.csv",
    "stale": "data/latest_scan_stale.csv",
    "partial": "data/latest_scan_partial.csv",
    "none": "未加载缓存",
    "invalidated": "缓存已失效",
}
NON_PARTIAL_CACHE_MAX_TRADING_DAYS = 3


@st.cache_resource(show_spinner=False)
def run_startup_maintenance(config_path: str, history_retention_days: int) -> dict[str, int]:
    deleted_scan_paths = purge_expired_non_partial_scan_cache()
    deleted_history_paths = purge_expired_history_cache(DATA_DIR, retention_trading_days=history_retention_days)
    return {
        "deleted_scan_files": len(deleted_scan_paths),
        "deleted_history_files": len(deleted_history_paths),
    }

PAGE_TABS_CSS = """
<style>
div[data-testid="stRadio"] > div {
    gap: 0.5rem;
}

div[data-testid="stRadio"] label {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 0.8rem;
    padding: 0.4rem 1rem;
    min-width: 4.5rem;
    justify-content: center;
    transition: all 0.15s ease;
}

div[data-testid="stRadio"] label:hover {
    border-color: #f43f5e;
    color: #e11d48;
}

div[data-testid="stRadio"] label:has(input:checked) {
    background: #fff1f2;
    border-color: #fb7185;
    color: #e11d48;
    font-weight: 600;
    box-shadow: inset 0 -2px 0 #f43f5e;
}

div[data-testid="stRadio"] label p {
    margin: 0;
}
</style>
"""

GOLD_SECTION_CSS = """
<style>
.gold-hero {
    background: linear-gradient(135deg, #fff8eb 0%, #fff1d6 100%);
    border: 1px solid #f3d8a6;
    border-radius: 1rem;
    padding: 1rem 1.1rem;
    margin: 0.25rem 0 0.9rem 0;
}

.gold-hero-title {
    font-size: 0.82rem;
    color: #92400e;
    margin-bottom: 0.35rem;
}

.gold-hero-main {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: flex-end;
    flex-wrap: wrap;
}

.gold-hero-label {
    font-size: 1.7rem;
    font-weight: 700;
    color: #7c2d12;
    line-height: 1.1;
}

.gold-hero-meta {
    font-size: 0.92rem;
    color: #7c2d12;
}

.gold-hero-summary {
    margin-top: 0.8rem;
    font-size: 0.95rem;
    color: #78350f;
    line-height: 1.6;
}
</style>
"""

MARKET_SIGNAL_CSS = """
<style>
.market-signal-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.75rem;
    margin: 0.35rem 0 0.9rem 0;
}

.market-signal-card {
    border-radius: 0.9rem;
    padding: 0.85rem 0.95rem;
    border: 1px solid transparent;
}

.market-signal-card h4 {
    margin: 0 0 0.35rem 0;
    font-size: 0.92rem;
}

.market-signal-card .status {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}

.market-signal-card .hint {
    font-size: 0.86rem;
    line-height: 1.5;
}

.market-signal-card .delta {
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.28rem;
}

.market-signal-card .delta-note {
    font-size: 0.76rem;
    opacity: 0.9;
    margin-bottom: 0.35rem;
}

.market-signal-green {
    background: #ecfdf5;
    border-color: #86efac;
    color: #166534;
}

.market-signal-yellow {
    background: #fffbeb;
    border-color: #fcd34d;
    color: #92400e;
}

.market-signal-red {
    background: #fef2f2;
    border-color: #fca5a5;
    color: #991b1b;
}

.market-signal-muted {
    background: #f8fafc;
    border-color: #cbd5e1;
    color: #475569;
}
</style>
"""


def style_scan_dataframe(dataframe: pd.DataFrame):
    if "执行优先级" not in dataframe.columns:
        return dataframe

    def highlight_priority(row: pd.Series) -> list[str]:
        styles = [""] * len(row)
        priority_color = PRIORITY_COLORS.get(str(row.get("执行优先级", "")), "")
        regime_color = REGIME_BADGES.get(str(row.get("市场环境", "")), "")
        if priority_color:
            styles = [f"background-color: {priority_color}"] * len(row)
        if regime_color and "市场环境" in dataframe.columns:
            regime_index = list(dataframe.columns).index("市场环境")
            styles[regime_index] = f"background-color: {regime_color}; font-weight: 600"
        intraday_action = str(row.get("盘中动作", ""))
        if intraday_action and intraday_action != "-" and "盘中动作" in dataframe.columns:
            intraday_index = list(dataframe.columns).index("盘中动作")
            intraday_colors = {
                "盘中试买": "background-color: #dcfce7; font-weight: 600",
                "盘中撤退": "background-color: #fecaca; font-weight: 600",
                "盘中观察": "background-color: #fef3c7; font-weight: 600",
            }
            if intraday_action in intraday_colors:
                styles[intraday_index] = intraday_colors[intraday_action]
        if "执行置信度" in dataframe.columns:
            confidence_value = str(row.get("执行置信度", ""))
            confidence_index = list(dataframe.columns).index("执行置信度")
            confidence_colors = {
                "高": "background-color: #dcfce7; font-weight: 700",
                "中": "background-color: #fef3c7; font-weight: 700",
                "低": "background-color: #fecaca; font-weight: 700",
            }
            if confidence_value in confidence_colors:
                styles[confidence_index] = confidence_colors[confidence_value]
        if "过热风险" in dataframe.columns:
            overheat_value = str(row.get("过热风险", ""))
            overheat_index = list(dataframe.columns).index("过热风险")
            overheat_colors = {
                "低": "background-color: #dcfce7; font-weight: 700",
                "中": "background-color: #fef3c7; font-weight: 700",
                "高": "background-color: #fecaca; font-weight: 700",
            }
            if overheat_value in overheat_colors:
                styles[overheat_index] = overheat_colors[overheat_value]
        return styles

    return dataframe.style.apply(highlight_priority, axis=1)


def _coerce_symbol_text(symbol: object) -> str:
    if pd.isna(symbol):
        return ""
    if isinstance(symbol, float) and symbol.is_integer():
        symbol = int(symbol)
    elif isinstance(symbol, int):
        symbol = int(symbol)
    text = str(symbol).strip().lower()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text:
        return ""
    if text.startswith(("sh", "sz", "bj")):
        prefix = text[:2]
        suffix = text[2:]
        return f"{prefix}{suffix.zfill(6)}" if suffix.isdigit() else text
    return text.zfill(6) if text.isdigit() else text


def format_symbol(symbol: object) -> str:
    return _coerce_symbol_text(symbol)


def normalize_market_symbol(symbol: object) -> str:
    symbol = _coerce_symbol_text(symbol)
    if symbol.startswith(("sh", "sz", "bj")):
        return symbol
    if symbol.startswith(("60", "68", "51", "56", "58")):
        return f"sh{symbol}"
    if symbol.startswith(("00", "30", "12", "15")):
        return f"sz{symbol}"
    if symbol.startswith(("4", "8", "92")):
        return f"bj{symbol}"
    return symbol


def normalize_benchmark_symbol(symbol: object) -> str:
    cleaned = str(symbol).strip()
    return cleaned.zfill(6) if cleaned.isdigit() else cleaned


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


def format_data_source_label(source: object, note: object = "") -> str:
    source_text = str(source or "").strip().lower()
    note_text = str(note or "").strip()
    labels = {
        "tushare": "Tushare",
        "efinance": "eFinance",
        "akshare_daily": "AkShare Daily",
        "akshare_hist": "AkShare Hist",
        "akshare_index": "AkShare Index",
        "akshare_spot": "AkShare Spot",
        "akshare_em": "AkShare EM",
        "baostock": "BaoStock",
        "cache": "本地缓存",
    }
    label = labels.get(source_text, source_text or "未知")
    if source_text == "cache" and note_text:
        return f"{label} ({note_text})"
    return label


def build_signal_cards_markup(signals: list[dict[str, object]]) -> str:
    return "".join(
        f"<div class='market-signal-card market-signal-{signal['tone']}'><h4>{signal['title']}</h4><div class='status'>{signal['status']}</div><div class='delta'>{signal.get('delta_status', '')}</div><div class='delta-note'>{signal.get('delta_note', '')}</div><div class='hint'>{signal['hint']}</div></div>"
        for signal in signals
    )


def classify_scan_overheat_risk(row: pd.Series) -> tuple[str, str, str]:
    score = 0
    rsi = pd.to_numeric(pd.Series([row.get("rsi")]), errors="coerce").iloc[0]
    live_delta = pd.to_numeric(pd.Series([row.get("live_delta")]), errors="coerce").iloc[0]
    volume_ratio = pd.to_numeric(pd.Series([row.get("volume_ratio")]), errors="coerce").iloc[0]
    entry_signal_type = str(row.get("entry_signal_type", "") or "")

    if pd.notna(rsi):
        if float(rsi) >= 80:
            score += 2
        elif float(rsi) >= 72:
            score += 1

    if pd.notna(live_delta):
        if float(live_delta) >= 0.055:
            score += 2
        elif float(live_delta) >= 0.025:
            score += 1

    if pd.notna(volume_ratio) and float(volume_ratio) >= 2.5:
        score += 1

    if entry_signal_type == "突破买入" and pd.notna(live_delta) and float(live_delta) >= 0.03:
        score += 1

    if score >= 4:
        return "高", "red", "追高与过热信号已较明显，优先等回踩。"
    if score >= 2:
        return "中", "yellow", "已有加速迹象，适合等更好的入场位置。"
    return "低", "green", "尚未出现明显过热特征。"


def classify_scan_execution_confidence(
    row: pd.Series,
    scan_source: str,
    latest_signal_date: date,
    requested_date: date,
) -> tuple[str, str, str]:
    score = 3
    market_filter_source = str(row.get("market_filter_source", "") or "")
    signal_age = pd.to_numeric(pd.Series([row.get("signal_age")]), errors="coerce").iloc[0]
    action = str(row.get("action", "") or "")

    if scan_source == "partial":
        score -= 2
    elif scan_source == "stale":
        score -= 2

    if latest_signal_date < requested_date:
        score -= 1

    if market_filter_source == "宽度过滤":
        score -= 1

    if pd.notna(signal_age) and float(signal_age) > 0 and action in {"买入", "补仓", "卖出"}:
        score -= 1

    if score <= 0:
        return "低", "red", "当前结果更适合观察，不适合机械执行。"
    if score == 1:
        return "中", "yellow", "可辅助判断，但下单前应再做人工确认。"
    return "高", "green", "结果完整度和时效性都较好，可作为主要参考。"


def parse_watchlist_symbols(raw_text: str) -> list[str]:
    tokens = re.split(r"[\s,;，；]+", raw_text.strip()) if raw_text.strip() else []
    symbols: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        cleaned = token.strip().lower()
        if not cleaned:
            continue
        normalized = normalize_market_symbol(cleaned)
        if normalized in seen:
            continue
        seen.add(normalized)
        symbols.append(normalized)
    return symbols


def load_watchlist_symbols() -> list[str]:
    if not WATCHLIST_PATH.exists():
        return []
    return parse_watchlist_symbols(WATCHLIST_PATH.read_text(encoding="utf-8"))


def save_watchlist_symbols(symbols: list[str]) -> None:
    WATCHLIST_PATH.write_text("\n".join(symbols), encoding="utf-8")


def normalize_fund_code(code: str) -> str:
    return re.sub(r"\D", "", str(code).strip())


def parse_fund_watchlist_codes(raw_text: str) -> list[str]:
    tokens = re.split(r"[\s,;，；]+", raw_text.strip()) if raw_text.strip() else []
    codes: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        normalized = normalize_fund_code(token)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        codes.append(normalized)
    return codes


def load_fund_watchlist_codes() -> list[str]:
    if not FUND_WATCHLIST_PATH.exists():
        return []
    return parse_fund_watchlist_codes(FUND_WATCHLIST_PATH.read_text(encoding="utf-8"))


def save_fund_watchlist_codes(codes: list[str]) -> None:
    FUND_WATCHLIST_PATH.write_text("\n".join(codes), encoding="utf-8")


def normalize_scan_dataframe(scan_df: pd.DataFrame) -> pd.DataFrame:
    if scan_df.empty:
        return scan_df

    defaults: dict[str, object] = {
        "add_on_signal": False,
        "entry_signal_type": "",
        "sell_reason": "",
        "execution_priority": "观察",
        "execution_priority_rank": 99,
        "position_state": "空仓",
        "holding_days": None,
        "atr_stop_price": None,
        "atr_stop_signal": False,
        "market_regime": "未知",
        "market_breadth": 0.0,
        "market_filter_source": "未知",
        "market_reason": "",
        "market_filter_passed": False,
        "market_add_on_passed": False,
    }
    normalized = scan_df.copy()
    if "symbol" in normalized.columns:
        normalized["symbol"] = normalized["symbol"].map(_coerce_symbol_text)
    for column, default_value in defaults.items():
        if column not in normalized.columns:
            normalized[column] = default_value
    return normalized


def get_latest_signal_date(scan_df: pd.DataFrame) -> date | None:
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


def load_best_available_scan_cache(
    config_fingerprint: str,
    allowed_sources: set[str] | None = None,
    max_trading_day_lag: int | None = None,
    reference_date: date | None = None,
) -> tuple[list[tuple[date | None, int, float, pd.DataFrame, str, bool]], bool]:
    existing_files = sorted(
        [(path, source) for path, source in SCAN_FILE_SPECS if path.exists()],
        key=lambda item: item[0].stat().st_mtime,
        reverse=True,
    )
    compatible_scans: list[tuple[date | None, int, float, pd.DataFrame, str, bool]] = []
    found_incompatible_cache = False
    for latest_path, source in existing_files:
        if allowed_sources is not None and source not in allowed_sources:
            continue
        metadata = load_scan_metadata(latest_path)
        if metadata is not None and metadata.get("config_fingerprint") != config_fingerprint:
            found_incompatible_cache = True
            continue
        scan_df = normalize_scan_dataframe(pd.read_csv(latest_path))
        latest_signal_date = get_latest_signal_date(scan_df)
        if max_trading_day_lag is not None:
            trading_day_lag = count_trading_day_lag(latest_signal_date, reference_date)
            if trading_day_lag is None or trading_day_lag > max_trading_day_lag:
                continue
        compatible_scans.append(
            (
                latest_signal_date,
                SCAN_SOURCE_PRIORITY.get(source, -1),
                latest_path.stat().st_mtime,
                scan_df,
                source,
                metadata is None,
            )
        )
    return compatible_scans, found_incompatible_cache


def load_latest_scan_from_disk(config_fingerprint: str, requested_date: date | None) -> tuple[pd.DataFrame, str | None, bool, str]:
    compatible_scans, found_incompatible_cache = load_best_available_scan_cache(config_fingerprint)
    if not compatible_scans and not found_incompatible_cache:
        return pd.DataFrame(), None, True, "none"

    source_messages = {
        "complete": "已先加载本地最近一次完整扫描结果；点击“刷新扫描”可重跑最新数据。",
        "stale": "已先加载本地参考结果；最新正式扫描仍可稍后刷新。",
        "partial": "已先加载本地部分扫描结果；点击“刷新扫描”可继续补全。",
    }

    if compatible_scans:
        latest_signal_date, _, _, scan_df, source, is_legacy_cache = max(
            compatible_scans,
            key=lambda item: (item[0] or date.min, item[1], item[2]),
        )
        status_message = source_messages.get(source)
        if is_legacy_cache:
            legacy_message = "当前展示的是旧版本地缓存结果，可能与当前参数不完全一致。"
            status_message = legacy_message if not status_message else f"{status_message} {legacy_message}"
        is_ready = requested_date is None or (latest_signal_date is not None and latest_signal_date >= requested_date)
        if requested_date is not None and latest_signal_date is not None and latest_signal_date < requested_date:
            freshness_message = (
                f"本地缓存最新信号日期仅到 {latest_signal_date:%Y-%m-%d}，"
                f"落后于当前应使用的 {requested_date:%Y-%m-%d}，已尝试自动刷新。"
            )
            status_message = freshness_message if not status_message else f"{status_message} {freshness_message}"
        return scan_df, status_message, is_ready, source

    if found_incompatible_cache:
        return pd.DataFrame(), "检测到策略参数已变更，本地扫描缓存已自动失效。", False, "invalidated"

    return pd.DataFrame(), None, True, "none"


def load_latest_non_partial_scan_from_disk(
    config_fingerprint: str,
    requested_date: date | None,
) -> tuple[pd.DataFrame, str | None, bool, str]:
    reference_date = requested_date or resolve_scan_target_date(None)
    compatible_scans, found_incompatible_cache = load_best_available_scan_cache(
        config_fingerprint,
        allowed_sources={"complete", "stale"},
        max_trading_day_lag=NON_PARTIAL_CACHE_MAX_TRADING_DAYS,
        reference_date=reference_date,
    )
    if compatible_scans:
        latest_signal_date, _, _, scan_df, source, is_legacy_cache = max(
            compatible_scans,
            key=lambda item: (item[0] or date.min, item[1], item[2]),
        )
        source_messages = {
            "complete": "已回退到本地最近一次完整扫描结果。",
            "stale": "已回退到本地最近一次参考扫描结果。",
        }
        status_message = source_messages.get(source)
        if is_legacy_cache:
            legacy_message = "当前展示的是旧版本地缓存结果，可能与当前参数不完全一致。"
            status_message = legacy_message if not status_message else f"{status_message} {legacy_message}"
        is_ready = requested_date is None or (latest_signal_date is not None and latest_signal_date >= requested_date)
        return scan_df, status_message, is_ready, source

    all_non_partial_scans, _ = load_best_available_scan_cache(
        config_fingerprint,
        allowed_sources={"complete", "stale"},
    )
    if all_non_partial_scans:
        newest_non_partial_date = max(item[0] for item in all_non_partial_scans if item[0] is not None)
        return (
            pd.DataFrame(),
            f"本地完整缓存最新只到 {newest_non_partial_date:%Y-%m-%d}，已超过 {NON_PARTIAL_CACHE_MAX_TRADING_DAYS} 个交易日，不再作为看板参照。",
            False,
            "none",
        )

    if found_incompatible_cache:
        return pd.DataFrame(), "检测到策略参数已变更，本地扫描缓存已自动失效。", False, "invalidated"

    return pd.DataFrame(), None, True, "none"


def compute_trailing_return(series: pd.Series, periods: int) -> float | None:
    numeric_series = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    if len(numeric_series) <= periods:
        return None
    start_value = float(numeric_series.iloc[-periods - 1])
    end_value = float(numeric_series.iloc[-1])
    if start_value == 0:
        return None
    return (end_value / start_value) - 1


@st.cache_data(ttl=900, show_spinner=False)
def cached_market_overview(
    config_path: str,
    config_fingerprint: str,
    latest_signal_date_text: str | None,
) -> dict[str, object]:
    del config_fingerprint
    config = load_config(config_path)
    timing_cfg = config.get("market_timing", {})
    benchmark_symbol = normalize_benchmark_symbol(timing_cfg.get("benchmark_symbol", "000300"))
    short_window = int(timing_cfg.get("benchmark_short_ma", 20))
    long_window = int(timing_cfg.get("benchmark_long_ma", 60))
    latest_signal_date = pd.to_datetime(latest_signal_date_text).date() if latest_signal_date_text else None
    target_date = latest_signal_date or resolve_scan_target_date(None)
    target_timestamp = pd.Timestamp(target_date)
    start_timestamp = target_timestamp - pd.Timedelta(days=max(long_window * 4, 180))

    overview: dict[str, object] = {
        "benchmark_symbol": benchmark_symbol,
        "latest_signal_date": target_date,
        "index_available": False,
        "index_trend": "未知",
        "index_trend_score": 0.0,
        "index_reason": "指数数据不可用。",
        "index_close": None,
        "index_ma_short": None,
        "index_ma_long": None,
        "index_return_5d": None,
        "index_return_10d": None,
        "index_return_20d": None,
        "index_data_date": None,
        "index_source": "未知",
        "index_source_note": "",
    }

    client = MarketDataClient(DATA_DIR)
    try:
        history = client.get_index_history(
            benchmark_symbol,
            start_timestamp.to_pydatetime(),
            target_timestamp.to_pydatetime(),
        )
        fetch_detail = client.get_last_fetch_detail("index")
        overview["index_source"] = fetch_detail.get("source", "未知")
        overview["index_source_note"] = fetch_detail.get("note", "")
    except RuntimeError as error:
        overview["index_reason"] = f"指数 {benchmark_symbol} 获取失败：{error}"
        return overview

    if history.empty or len(history) < long_window:
        overview["index_reason"] = f"指数 {benchmark_symbol} 数据不足，暂时无法给出指数趋势确认。"
        return overview

    history = history.sort_values("date").reset_index(drop=True).copy()
    history["ma_short"] = history["close"].rolling(short_window).mean()
    history["ma_long"] = history["close"].rolling(long_window).mean()
    latest_row = history.iloc[-1]
    ma_short = float(latest_row["ma_short"]) if pd.notna(latest_row["ma_short"]) else None
    ma_long = float(latest_row["ma_long"]) if pd.notna(latest_row["ma_long"]) else None
    close = float(latest_row["close"])
    short_slope_positive = False
    if len(history) > 5 and pd.notna(history["ma_short"].iloc[-6]) and ma_short is not None:
        short_slope_positive = ma_short > float(history["ma_short"].iloc[-6])

    if ma_short is not None and ma_long is not None and close > ma_short > ma_long and short_slope_positive:
        trend_label = "强势多头"
        trend_score = 2.0
        trend_reason = f"指数 {benchmark_symbol} 站上 MA{short_window}/MA{long_window}，且短均线继续上行。"
    elif ma_short is not None and ma_long is not None and close > ma_long and ma_short >= ma_long:
        trend_label = "偏多"
        trend_score = 1.0
        trend_reason = f"指数 {benchmark_symbol} 站上 MA{long_window}，但趋势确认仍偏早。"
    elif ma_short is not None and ma_long is not None and close < ma_short < ma_long and not short_slope_positive:
        trend_label = "偏空"
        trend_score = -2.0
        trend_reason = f"指数 {benchmark_symbol} 位于 MA{short_window}/MA{long_window} 下方，短均线仍在下压。"
    else:
        trend_label = "震荡"
        trend_score = 0.0
        trend_reason = f"指数 {benchmark_symbol} 尚未形成清晰单边趋势。"

    overview.update(
        {
            "index_available": True,
            "index_trend": trend_label,
            "index_trend_score": trend_score,
            "index_reason": trend_reason,
            "index_close": close,
            "index_ma_short": ma_short,
            "index_ma_long": ma_long,
            "index_return_5d": compute_trailing_return(history["close"], 5),
            "index_return_10d": compute_trailing_return(history["close"], 10),
            "index_return_20d": compute_trailing_return(history["close"], 20),
            "index_data_date": pd.to_datetime(latest_row["date"]).date(),
        }
    )
    return overview


@st.cache_data(ttl=900, show_spinner=False)
def cached_market_index_history(
    config_path: str,
    config_fingerprint: str,
    latest_signal_date_text: str | None,
) -> pd.DataFrame:
    del config_fingerprint
    config = load_config(config_path)
    timing_cfg = config.get("market_timing", {})
    benchmark_symbol = normalize_benchmark_symbol(timing_cfg.get("benchmark_symbol", "000300"))
    short_window = int(timing_cfg.get("benchmark_short_ma", 20))
    long_window = int(timing_cfg.get("benchmark_long_ma", 60))
    latest_signal_date = pd.to_datetime(latest_signal_date_text).date() if latest_signal_date_text else None
    target_date = latest_signal_date or resolve_scan_target_date(None)
    target_timestamp = pd.Timestamp(target_date)
    start_timestamp = target_timestamp - pd.Timedelta(days=max(long_window * 5, 240))

    client = MarketDataClient(DATA_DIR)
    try:
        history = client.get_index_history(
            benchmark_symbol,
            start_timestamp.to_pydatetime(),
            target_timestamp.to_pydatetime(),
        )
        fetch_detail = client.get_last_fetch_detail("index")
    except RuntimeError:
        cache_pattern = f"benchmark_{benchmark_symbol}_*.csv"
        cache_candidates = sorted(DATA_DIR.glob(cache_pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        history = pd.DataFrame()
        fetch_detail = {"source": "", "note": ""}
        for cache_path in cache_candidates:
            try:
                cached_history = pd.read_csv(cache_path, parse_dates=["date"])
            except Exception:
                continue
            if cached_history.empty or "date" not in cached_history.columns:
                continue
            history = cached_history
            fetch_detail = {"source": "cache", "note": cache_path.name}
            break
        if history.empty:
            return pd.DataFrame()

    if history.empty:
        return history
    history = history.sort_values("date").reset_index(drop=True).copy()
    history["ma_short"] = history["close"].rolling(short_window).mean()
    history["ma_long"] = history["close"].rolling(long_window).mean()
    history.attrs["benchmark_symbol"] = benchmark_symbol
    history.attrs["index_source"] = fetch_detail.get("source", "")
    history.attrs["index_source_note"] = fetch_detail.get("note", "")
    return history


def build_market_index_figure(history: pd.DataFrame) -> go.Figure:
    benchmark_symbol = str(history.attrs.get("benchmark_symbol", "基准指数"))
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["close"],
            mode="lines",
            name=benchmark_symbol,
            line={"width": 2.2, "color": "#0f172a"},
        )
    )
    if "ma_short" in history.columns:
        figure.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["ma_short"],
                mode="lines",
                name="MA20",
                line={"width": 1.8, "color": "#d97706"},
            )
        )
    if "ma_long" in history.columns:
        figure.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["ma_long"],
                mode="lines",
                name="MA60",
                line={"width": 1.8, "color": "#2563eb"},
            )
        )
    figure.update_layout(
        height=320,
        margin={"l": 12, "r": 12, "t": 20, "b": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hovermode="x unified",
    )
    return figure


def build_market_outlook(scan_df: pd.DataFrame, market_overview: dict[str, object]) -> dict[str, object]:
    if scan_df.empty:
        return {
            "label": "数据不足",
            "score": 0.0,
            "regime": "未知",
            "breadth": None,
            "buy_count": 0,
            "add_on_count": 0,
            "sell_count": 0,
            "confidence": "低",
            "summary": "当前扫描结果为空，无法对未来 1 到 2 周市场方向给出有意义判断。",
        }

    regime = str(scan_df.get("market_regime", pd.Series(["未知"])).iloc[0])
    breadth_value = pd.to_numeric(scan_df.get("market_breadth", pd.Series([None])), errors="coerce").dropna()
    breadth = float(breadth_value.iloc[0]) if not breadth_value.empty else None
    buy_count = int(pd.to_numeric(scan_df.get("buy_signal", pd.Series(dtype=bool)), errors="coerce").fillna(False).astype(bool).sum())
    add_on_count = int(pd.to_numeric(scan_df.get("add_on_signal", pd.Series(dtype=bool)), errors="coerce").fillna(False).astype(bool).sum())
    sell_count = int(pd.to_numeric(scan_df.get("sell_signal", pd.Series(dtype=bool)), errors="coerce").fillna(False).astype(bool).sum())

    score = 0.0
    score += {"风险开": 1.5, "中性": 0.0, "风险关": -1.5}.get(regime, 0.0)
    if breadth is not None:
        if breadth >= 0.65:
            score += 1.0
        elif breadth >= 0.55:
            score += 0.5
        elif breadth < 0.45:
            score -= 1.0
    score += float(market_overview.get("index_trend_score", 0.0) or 0.0)

    return_10d = market_overview.get("index_return_10d")
    if isinstance(return_10d, (int, float)):
        if return_10d >= 0.02:
            score += 0.5
        elif return_10d <= -0.02:
            score -= 0.5

    signal_bias = (buy_count + add_on_count) - sell_count
    if signal_bias >= 3:
        score += 0.5
    elif signal_bias <= -2:
        score -= 0.5

    if score >= 3.0:
        label = "看多"
    elif score >= 1.5:
        label = "偏多"
    elif score <= -3.0:
        label = "看空"
    elif score <= -1.5:
        label = "偏空"
    else:
        label = "中性"

    confidence_parts = 0
    if market_overview.get("index_available"):
        confidence_parts += 1
    if breadth is not None:
        confidence_parts += 1
    if len(scan_df) >= 10:
        confidence_parts += 1
    confidence = {0: "低", 1: "中", 2: "中高", 3: "高"}.get(confidence_parts, "中")

    breadth_text = f"趋势广度 {breadth:.0%}" if breadth is not None else "趋势广度未知"
    index_text = str(market_overview.get("index_reason", "指数趋势未知"))
    regime_display = format_market_regime_label(regime)
    summary = (
        f"未来 1 到 2 周判断偏向{label}。当前市场环境为{regime_display}，{breadth_text}；"
        f"指数侧判断为{market_overview.get('index_trend', '未知')}，{index_text}"
    )
    return {
        "label": label,
        "score": round(score, 2),
        "regime": regime,
        "regime_display": regime_display,
        "breadth": breadth,
        "buy_count": buy_count,
        "add_on_count": add_on_count,
        "sell_count": sell_count,
        "confidence": confidence,
        "summary": summary,
    }


def build_market_snapshot_summary(snapshot_df: pd.DataFrame) -> dict[str, object]:
    if snapshot_df.empty:
        return {
            "available": False,
            "total_count": 0,
            "up_count": 0,
            "down_count": 0,
            "flat_count": 0,
            "up_ratio": None,
            "mean_pct": None,
            "median_pct": None,
            "gt3_count": 0,
            "lt3_count": 0,
            "gt5_count": 0,
            "lt5_count": 0,
            "turnover_total_billion": None,
            "turnover_median_billion": None,
            "turnover_rate_median": None,
            "turnover_rate_mean": None,
            "high_turnover_count": 0,
            "top_turnover_positive_ratio": None,
            "top_turnover_df": pd.DataFrame(),
            "strongest_df": pd.DataFrame(),
            "weakest_df": pd.DataFrame(),
        }

    working = snapshot_df.copy()
    for column in ["pct_change", "turnover_amount", "turnover_rate"]:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    pct_change = working.get("pct_change", pd.Series(dtype=float))
    turnover_amount = working.get("turnover_amount", pd.Series(dtype=float))
    turnover_rate = working.get("turnover_rate", pd.Series(dtype=float))

    total_count = int(pct_change.notna().sum())
    up_count = int((pct_change > 0).sum())
    down_count = int((pct_change < 0).sum())
    flat_count = int((pct_change == 0).sum())

    top_turnover_columns = [column for column in ["symbol", "name", "last_price", "pct_change", "turnover_amount", "turnover_rate"] if column in working.columns]
    top_turnover_df = pd.DataFrame()
    strongest_df = pd.DataFrame()
    weakest_df = pd.DataFrame()
    top_turnover_positive_ratio = None
    if top_turnover_columns:
        top_turnover_df = working[top_turnover_columns].sort_values("turnover_amount", ascending=False, na_position="last").head(12).reset_index(drop=True)
        top_turnover_pct = pd.to_numeric(top_turnover_df.get("pct_change", pd.Series(dtype=float)), errors="coerce")
        if not top_turnover_pct.empty and top_turnover_pct.notna().any():
            top_turnover_positive_ratio = float((top_turnover_pct > 0).mean())
        strongest_df = working[top_turnover_columns].sort_values(["pct_change", "turnover_amount"], ascending=[False, False], na_position="last").head(10).reset_index(drop=True)
        weakest_df = working[top_turnover_columns].sort_values(["pct_change", "turnover_amount"], ascending=[True, False], na_position="last").head(10).reset_index(drop=True)

    return {
        "available": True,
        "total_count": total_count,
        "up_count": up_count,
        "down_count": down_count,
        "flat_count": flat_count,
        "up_ratio": (up_count / total_count) if total_count else None,
        "mean_pct": float(pct_change.mean()) if pct_change.notna().any() else None,
        "median_pct": float(pct_change.median()) if pct_change.notna().any() else None,
        "gt3_count": int((pct_change >= 3).sum()),
        "lt3_count": int((pct_change <= -3).sum()),
        "gt5_count": int((pct_change >= 5).sum()),
        "lt5_count": int((pct_change <= -5).sum()),
        "turnover_total_billion": (float(turnover_amount.sum()) / 100000000) if turnover_amount.notna().any() else None,
        "turnover_median_billion": (float(turnover_amount.median()) / 100000000) if turnover_amount.notna().any() else None,
        "turnover_rate_median": float(turnover_rate.median()) if turnover_rate.notna().any() else None,
        "turnover_rate_mean": float(turnover_rate.mean()) if turnover_rate.notna().any() else None,
        "high_turnover_count": int((turnover_rate >= 10).sum()) if turnover_rate.notna().any() else 0,
        "top_turnover_positive_ratio": top_turnover_positive_ratio,
        "top_turnover_df": top_turnover_df,
        "strongest_df": strongest_df,
        "weakest_df": weakest_df,
    }


def load_market_snapshot_summary_history() -> pd.DataFrame:
    if not MARKET_SNAPSHOT_SUMMARY_HISTORY_PATH.exists():
        return pd.DataFrame()
    try:
        history = pd.read_csv(
            MARKET_SNAPSHOT_SUMMARY_HISTORY_PATH,
            parse_dates=["snapshot_date", "snapshot_timestamp"],
        )
    except Exception:
        return pd.DataFrame()
    return history.sort_values("snapshot_date").reset_index(drop=True)


def update_market_snapshot_summary_history(
    snapshot_summary: dict[str, object],
    market_summary: dict[str, object],
    snapshot_timestamp: pd.Timestamp | None,
) -> pd.Series | None:
    if snapshot_timestamp is None or not snapshot_summary.get("available"):
        return None

    current_date = pd.Timestamp(snapshot_timestamp).normalize()
    record = {
        "snapshot_date": current_date,
        "snapshot_timestamp": pd.Timestamp(snapshot_timestamp),
        "up_ratio": snapshot_summary.get("up_ratio"),
        "turnover_total_billion": snapshot_summary.get("turnover_total_billion"),
        "top_turnover_positive_ratio": snapshot_summary.get("top_turnover_positive_ratio"),
        "continuation_score": market_summary.get("continuation_score"),
        "pullback_risk_score": market_summary.get("pullback_risk_score"),
        "composite_score": market_summary.get("composite_score"),
    }

    history = load_market_snapshot_summary_history()
    previous_row = None
    if not history.empty and "snapshot_date" in history.columns:
        previous_rows = history.loc[pd.to_datetime(history["snapshot_date"]).dt.normalize() < current_date]
        if not previous_rows.empty:
            previous_row = previous_rows.iloc[-1]

    history = history.loc[pd.to_datetime(history.get("snapshot_date", pd.Series(dtype="datetime64[ns]")), errors="coerce").dt.normalize() != current_date].copy() if not history.empty else pd.DataFrame()
    history = pd.concat([history, pd.DataFrame([record])], ignore_index=True)
    history = history.sort_values("snapshot_date").drop_duplicates(subset=["snapshot_date"], keep="last").reset_index(drop=True)
    history.to_csv(MARKET_SNAPSHOT_SUMMARY_HISTORY_PATH, index=False, encoding="utf-8-sig")
    return previous_row


def _compare_market_delta(
    current_value: object,
    previous_value: object,
    improve_threshold: float,
    weaken_threshold: float,
    *,
    relative: bool = False,
) -> tuple[str, str, str]:
    if pd.isna(current_value) or pd.isna(previous_value):
        return "首日记录", "muted", "暂无上一条日度摘要可比较。"

    current = float(current_value)
    previous = float(previous_value)
    if relative:
        if previous == 0:
            return "较前日持平", "yellow", "上一条记录基数过小，暂不输出相对变化。"
        delta_value = (current / previous) - 1
        note = f"相对上一条记录变化 {delta_value:+.2%}"
    else:
        delta_value = current - previous
        note = f"较上一条记录变化 {delta_value:+.2%}"

    if delta_value >= improve_threshold:
        return "较前日改善", "green", note
    if delta_value <= -weaken_threshold:
        return "较前日转弱", "red", note
    return "较前日持平", "yellow", note


def apply_market_signal_deltas(
    signals: list[dict[str, object]],
    snapshot_summary: dict[str, object],
    market_summary: dict[str, object],
    previous_row: pd.Series | None,
) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for signal in signals:
        item = dict(signal)
        title = str(item.get("title", ""))
        if previous_row is None:
            item["delta_status"] = "首日记录"
            item["delta_tone"] = "muted"
            item["delta_note"] = "暂无上一条日度摘要可比较。"
        elif title == "量能信号灯":
            delta_status, delta_tone, delta_note = _compare_market_delta(
                snapshot_summary.get("turnover_total_billion"),
                previous_row.get("turnover_total_billion"),
                0.08,
                0.08,
                relative=True,
            )
            item["delta_status"] = delta_status
            item["delta_tone"] = delta_tone
            item["delta_note"] = delta_note
        elif title == "广度信号灯":
            delta_status, delta_tone, delta_note = _compare_market_delta(
                snapshot_summary.get("up_ratio"),
                previous_row.get("up_ratio"),
                0.02,
                0.02,
            )
            item["delta_status"] = delta_status
            item["delta_tone"] = delta_tone
            item["delta_note"] = delta_note
        elif title == "主线信号灯":
            delta_status, delta_tone, delta_note = _compare_market_delta(
                snapshot_summary.get("top_turnover_positive_ratio"),
                previous_row.get("top_turnover_positive_ratio"),
                0.08,
                0.08,
            )
            item["delta_status"] = delta_status
            item["delta_tone"] = delta_tone
            item["delta_note"] = delta_note
        else:
            delta_status, delta_tone, delta_note = _compare_market_delta(
                market_summary.get("composite_score"),
                previous_row.get("composite_score"),
                0.35,
                0.35,
            )
            item["delta_status"] = delta_status
            item["delta_tone"] = delta_tone
            item["delta_note"] = delta_note
        enriched.append(item)
    return enriched


def build_market_summary_analysis(
    outlook: dict[str, object],
    market_overview: dict[str, object],
    snapshot_summary: dict[str, object],
) -> dict[str, object]:
    up_ratio = snapshot_summary.get("up_ratio")
    mean_pct = snapshot_summary.get("mean_pct")
    turnover_total_billion = snapshot_summary.get("turnover_total_billion")
    top_turnover_positive_ratio = snapshot_summary.get("top_turnover_positive_ratio")
    gt5_count = int(snapshot_summary.get("gt5_count") or 0)
    lt5_count = int(snapshot_summary.get("lt5_count") or 0)
    index_trend = str(market_overview.get("index_trend", "未知"))

    continuation_score = float(outlook.get("score", 0.0) or 0.0)
    if isinstance(up_ratio, (int, float)):
        if up_ratio >= 0.62:
            continuation_score += 1.0
        elif up_ratio >= 0.56:
            continuation_score += 0.5
        elif up_ratio < 0.48:
            continuation_score -= 0.8
    if isinstance(mean_pct, (int, float)):
        if mean_pct >= 0.8:
            continuation_score += 0.8
        elif mean_pct >= 0.3:
            continuation_score += 0.3
        elif mean_pct < -0.3:
            continuation_score -= 0.8
    if isinstance(turnover_total_billion, (int, float)):
        if turnover_total_billion >= 20000:
            continuation_score += 0.5
        elif turnover_total_billion >= 15000:
            continuation_score += 0.2
    if isinstance(top_turnover_positive_ratio, (int, float)):
        if top_turnover_positive_ratio >= 0.65:
            continuation_score += 0.5
        elif top_turnover_positive_ratio < 0.45:
            continuation_score -= 0.5

    if continuation_score >= 2.6:
        continuation_label = "继续上攻"
    elif continuation_score >= 1.2:
        continuation_label = "偏强震荡上行"
    elif continuation_score <= -1.2:
        continuation_label = "回调压力偏大"
    else:
        continuation_label = "震荡待确认"

    pullback_risk_score = 0
    if index_trend in {"震荡", "未知"}:
        pullback_risk_score += 1
    elif index_trend == "偏空":
        pullback_risk_score += 2
    if str(outlook.get("label", "中性")) in {"中性", "偏空", "看空"}:
        pullback_risk_score += 1
    if isinstance(up_ratio, (int, float)) and up_ratio < 0.55:
        pullback_risk_score += 1
    if isinstance(mean_pct, (int, float)) and mean_pct < 0.3:
        pullback_risk_score += 1
    if gt5_count <= lt5_count:
        pullback_risk_score += 1
    if isinstance(top_turnover_positive_ratio, (int, float)) and top_turnover_positive_ratio < 0.5:
        pullback_risk_score += 1

    if pullback_risk_score >= 4:
        pullback_risk = "高"
    elif pullback_risk_score >= 2:
        pullback_risk = "中"
    else:
        pullback_risk = "低"

    if isinstance(up_ratio, (int, float)) and isinstance(mean_pct, (int, float)):
        if up_ratio >= 0.6 and mean_pct >= 0.5:
            market_tone = "普涨偏强"
        elif up_ratio >= 0.55 and mean_pct >= 0.0:
            market_tone = "震荡偏强"
        elif up_ratio < 0.48 and mean_pct < 0.0:
            market_tone = "普跌承压"
        else:
            market_tone = "分化震荡"
    else:
        market_tone = "数据待确认"

    headline = (
        f"从最近数据与当天盘面综合看，当前大盘更接近“{continuation_label}”而非单边失速。"
        if continuation_label in {"继续上攻", "偏强震荡上行"}
        else f"从最近数据与当天盘面综合看，当前大盘更接近“{continuation_label}”，需要防范冲高回落。"
    )
    breadth_text = (
        f"全市场上涨占比约 {up_ratio:.0%}、平均涨幅约 {mean_pct:.2f}%" if isinstance(up_ratio, (int, float)) and isinstance(mean_pct, (int, float)) else "全市场强弱分布暂不完整"
    )
    turnover_text = (
        f"两市快照成交额约 {turnover_total_billion:.0f} 亿" if isinstance(turnover_total_billion, (int, float)) else "成交额统计暂缺"
    )
    top_turnover_text = (
        f"高成交核心标的中约 {top_turnover_positive_ratio:.0%} 保持红盘" if isinstance(top_turnover_positive_ratio, (int, float)) else "高成交主线强弱暂不完整"
    )
    summary = f"{headline}{breadth_text}，{turnover_text}，{top_turnover_text}。指数趋势侧当前仍为“{index_trend}”，因此短线继续上行与回调消化会并存。"

    watch_items = [
        "量能能否继续维持高位，而不是指数上行但成交额回落。",
        "上涨家数是否继续显著大于下跌家数，避免只剩权重护盘。",
        "高成交主线是否出现集中冲高回落，这是短线回调最常见前兆。",
    ]
    if pullback_risk == "高":
        risk_hint = "如果后续出现缩量、上涨家数收窄、热门股长上影增多，回调风险会快速抬升。"
    elif pullback_risk == "中":
        risk_hint = "当前仍有上行动能，但更适合把它视为偏强震荡，而不是无条件顺畅逼空。"
    else:
        risk_hint = "只要量能与广度不明显转弱，指数仍有继续抬台阶的基础。"

    if isinstance(turnover_total_billion, (int, float)):
        if turnover_total_billion >= 20000:
            volume_signal = {
                "title": "量能信号灯",
                "status": "绿灯",
                "tone": "green",
                "hint": f"当前成交额约 {turnover_total_billion:.0f} 亿，属于高位量能区，继续上攻仍有资金承接基础。",
            }
        elif turnover_total_billion >= 15000:
            volume_signal = {
                "title": "量能信号灯",
                "status": "黄灯",
                "tone": "yellow",
                "hint": f"当前成交额约 {turnover_total_billion:.0f} 亿，量能不弱，但还不足以支持无条件逼空。",
            }
        else:
            volume_signal = {
                "title": "量能信号灯",
                "status": "红灯",
                "tone": "red",
                "hint": f"当前成交额约 {turnover_total_billion:.0f} 亿，量能偏弱，指数继续上冲更容易受阻。",
            }
    else:
        volume_signal = {
            "title": "量能信号灯",
            "status": "黄灯",
            "tone": "yellow",
            "hint": "当前缺少完整总成交额参考，量能强弱只能做保守判断。",
        }

    if isinstance(up_ratio, (int, float)):
        if up_ratio >= 0.60:
            breadth_signal = {
                "title": "广度信号灯",
                "status": "绿灯",
                "tone": "green",
                "hint": f"上涨占比约 {up_ratio:.0%}，赚钱效应扩散较充分，指数上行不是只靠少数权重。",
            }
        elif up_ratio >= 0.53:
            breadth_signal = {
                "title": "广度信号灯",
                "status": "黄灯",
                "tone": "yellow",
                "hint": f"上涨占比约 {up_ratio:.0%}，市场仍偏强，但广度优势还不算绝对。",
            }
        else:
            breadth_signal = {
                "title": "广度信号灯",
                "status": "红灯",
                "tone": "red",
                "hint": f"上涨占比仅约 {up_ratio:.0%}，若指数继续走高而广度不跟，回调风险会上升。",
            }
    else:
        breadth_signal = {
            "title": "广度信号灯",
            "status": "黄灯",
            "tone": "yellow",
            "hint": "当前缺少完整涨跌分布，广度判断只能保守处理。",
        }

    if isinstance(top_turnover_positive_ratio, (int, float)):
        if top_turnover_positive_ratio >= 0.60:
            leadership_signal = {
                "title": "主线信号灯",
                "status": "绿灯",
                "tone": "green",
                "hint": f"高成交核心股约 {top_turnover_positive_ratio:.0%} 维持红盘，主线资金仍在承接。",
            }
        elif top_turnover_positive_ratio >= 0.45:
            leadership_signal = {
                "title": "主线信号灯",
                "status": "黄灯",
                "tone": "yellow",
                "hint": f"高成交核心股约 {top_turnover_positive_ratio:.0%} 维持红盘，主线开始分化，但未到集体转弱。",
            }
        else:
            leadership_signal = {
                "title": "主线信号灯",
                "status": "红灯",
                "tone": "red",
                "hint": f"高成交核心股仅约 {top_turnover_positive_ratio:.0%} 维持红盘，主线冲高回落风险偏大。",
            }
    else:
        leadership_signal = {
            "title": "主线信号灯",
            "status": "黄灯",
            "tone": "yellow",
            "hint": "当前缺少高成交主线强弱统计，暂按中性处理。",
        }

    if continuation_label in {"继续上攻", "偏强震荡上行"} and pullback_risk == "低":
        composite_signal = {
            "title": "综合风险灯",
            "status": "绿灯",
            "tone": "green",
            "hint": "继续上攻的基础较完整，只要后续量能与广度不塌，指数仍可偏乐观对待。",
        }
    elif pullback_risk == "高":
        composite_signal = {
            "title": "综合风险灯",
            "status": "红灯",
            "tone": "red",
            "hint": "短线回调风险已经偏高，若次日再出现缩量冲高或主线退潮，应优先防守。",
        }
    else:
        composite_signal = {
            "title": "综合风险灯",
            "status": "黄灯",
            "tone": "yellow",
            "hint": "当前更像偏强震荡，适合边走边看，不适合把行情理解成无条件单边上攻。",
        }

    return {
        "continuation_label": continuation_label,
        "continuation_score": continuation_score,
        "pullback_risk": pullback_risk,
        "pullback_risk_score": pullback_risk_score,
        "composite_score": continuation_score - (pullback_risk_score * 0.7),
        "market_tone": market_tone,
        "summary": summary,
        "watch_items": watch_items,
        "risk_hint": risk_hint,
        "signals": [volume_signal, breadth_signal, leadership_signal, composite_signal],
    }


def resolve_dashboard_market_regime(
    config_path: str,
    config_fingerprint: str,
    scan_df: pd.DataFrame,
) -> str:
    if not scan_df.empty and "market_regime" in scan_df.columns:
        regime_values = scan_df["market_regime"].dropna().astype(str).str.strip()
        regime_values = regime_values[regime_values != ""]
        if not regime_values.empty:
            return regime_values.iloc[0]

    latest_signal_date = get_latest_signal_date(scan_df)
    latest_signal_date_text = latest_signal_date.isoformat() if latest_signal_date else None
    market_overview = cached_market_overview(config_path, config_fingerprint, latest_signal_date_text)
    outlook = build_market_outlook(scan_df, market_overview)
    return str(outlook.get("regime", "未知") or "未知")


@st.cache_data(ttl=900, show_spinner=False)
def cached_scan(
    config_path: str,
    config_fingerprint: str,
    as_of_text: str | None,
    top: int,
    refresh_token: int,
) -> tuple[pd.DataFrame, str | None, bool, str]:
    del refresh_token
    try:
        result = scan_market(config_path, as_of_text, top)
        scan_df = normalize_scan_dataframe(result)
        status_message = result.attrs.get("status_message")
        scan_source = "partial" if result.attrs.get("is_partial") else "complete"
        return scan_df, status_message, True, scan_source
    except DataNotReadyError as error:
        fallback_df, _, fallback_ready, fallback_source = load_latest_scan_from_disk(
            config_fingerprint,
            resolve_scan_target_date(as_of_text),
        )
        return fallback_df, str(error), fallback_ready, fallback_source
    except ScanInProgressError as error:
        fallback_df, fallback_message, fallback_ready, fallback_source = load_latest_scan_from_disk(
            config_fingerprint,
            resolve_scan_target_date(as_of_text),
        )
        status_message = str(error)
        if fallback_message:
            status_message = f"{status_message} 当前展示最近一次本地缓存结果。{fallback_message}"
        else:
            status_message = f"{status_message} 当前展示最近一次本地缓存结果。"
        return fallback_df, status_message, fallback_ready, fallback_source


@st.cache_data(ttl=900, show_spinner=False)
def cached_history(
    symbol: str,
    start_date: date,
    end_date: date,
    config_path: str,
    config_fingerprint: str,
) -> pd.DataFrame:
    del config_fingerprint
    client = MarketDataClient(DATA_DIR)
    history = client.get_history(
        symbol,
        datetime.combine(start_date, datetime.min.time()),
        datetime.combine(end_date, datetime.min.time()),
    )
    fetch_detail = client.get_last_fetch_detail("history")
    if history.empty:
        history.attrs["history_source"] = fetch_detail.get("source", "")
        history.attrs["history_source_note"] = fetch_detail.get("note", "")
        return history
    config = load_config(config_path)
    enriched_history = add_signal_columns(history, config)
    enriched_history.attrs["history_source"] = fetch_detail.get("source", "")
    enriched_history.attrs["history_source_note"] = fetch_detail.get("note", "")
    return enriched_history


@st.cache_data(ttl=60, show_spinner=False)
def cached_symbol_snapshot(symbol: str) -> dict[str, object] | None:
    client = MarketDataClient(DATA_DIR)
    snapshot = client.get_symbol_snapshot(symbol)
    if snapshot is None:
        return None
    fetch_detail = client.get_last_fetch_detail("snapshot")
    snapshot["snapshot_source"] = fetch_detail.get("source", "")
    snapshot["snapshot_source_note"] = fetch_detail.get("note", "")
    return snapshot


@st.cache_data(ttl=60, show_spinner=False)
def cached_snapshot_dataframe() -> pd.DataFrame:
    snapshot_path = DATA_DIR / "latest_snapshot.csv"
    if snapshot_path.exists():
        snapshot_df = pd.read_csv(snapshot_path)
        snapshot_df.attrs["snapshot_source"] = "cache"
        snapshot_df.attrs["snapshot_source_note"] = snapshot_path.name
        return snapshot_df

    client = MarketDataClient(DATA_DIR)
    try:
        snapshot_df = client.get_universe_snapshot()
        fetch_detail = client.get_last_fetch_detail("snapshot")
        snapshot_df.attrs["snapshot_source"] = fetch_detail.get("source", "")
        snapshot_df.attrs["snapshot_source_note"] = fetch_detail.get("note", "")
        return snapshot_df
    except RuntimeError:
        return pd.DataFrame()


def normalize_fund_rank_dataframe(fund_df: pd.DataFrame) -> pd.DataFrame:
    if fund_df is None or fund_df.empty:
        return pd.DataFrame()

    result = fund_df.copy()
    if "基金代码" in result.columns:
        result["基金代码"] = result["基金代码"].map(normalize_fund_code)
    for column in ["单位净值", "累计净值", "近1月", "近3月", "近6月", "近1年"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if "日期" in result.columns:
        result["日期"] = pd.to_datetime(result["日期"], errors="coerce")
    return result


def parse_fee_value(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_percent_text(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def compute_max_drawdown(nav_series: pd.Series) -> float | None:
    nav = pd.to_numeric(nav_series, errors="coerce").dropna()
    if nav.empty:
        return None
    drawdown = (nav / nav.cummax()) - 1
    return abs(float(drawdown.min())) * 100


def compute_annualized_volatility(nav_series: pd.Series, annual_factor: float) -> float | None:
    nav = pd.to_numeric(nav_series, errors="coerce").dropna()
    if len(nav) < 3:
        return None
    returns = nav.pct_change().dropna()
    if returns.empty:
        return None
    return float(returns.std()) * (annual_factor ** 0.5) * 100


def normalize_etf_daily_dataframe(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df is None or snapshot_df.empty:
        return pd.DataFrame()

    result = snapshot_df.copy()
    nav_columns = [column for column in result.columns if column.endswith("-单位净值")]
    nav_dates = []
    for column in nav_columns:
        try:
            nav_dates.append((pd.to_datetime(column.split("-")[0]), column))
        except Exception:
            continue
    nav_dates = sorted(nav_dates, key=lambda item: item[0], reverse=True)
    latest_nav_column = nav_dates[0][1] if nav_dates else None
    previous_nav_column = nav_dates[1][1] if len(nav_dates) > 1 else None

    selected_columns = [column for column in ["基金代码", "基金简称", "类型", "市价", "折价率", "增长率", latest_nav_column, previous_nav_column] if column]
    result = result[selected_columns].copy()
    if "基金代码" in result.columns:
        result["基金代码"] = result["基金代码"].map(normalize_fund_code)
    rename_map = {}
    if latest_nav_column:
        rename_map[latest_nav_column] = "最新单位净值"
    if previous_nav_column:
        rename_map[previous_nav_column] = "前一单位净值"
    result = result.rename(columns=rename_map)
    for column in ["市价", "最新单位净值", "前一单位净值"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in ["折价率", "增长率"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column].map(parse_percent_text), errors="coerce")
    if {"市价", "前一单位净值", "增长率"}.issubset(result.columns):
        result["日收益偏离代理"] = (
            ((result["市价"] / result["前一单位净值"]) - 1) * 100 - result["增长率"]
        ).abs()
    return result


def classify_fund_category(name: object) -> str:
    text = str(name or "")
    debt_keywords = ["债", "固收", "同业存单", "国债", "信用", "可转债", "短债", "纯债", "中短债"]
    index_keywords = ["ETF", "指数", "联接"]
    mixed_keywords = ["混合", "配置", "均衡", "平衡", "灵活", "持有"]

    if any(keyword in text for keyword in debt_keywords):
        return "债基"
    if any(keyword in text.upper() for keyword in index_keywords):
        return "ETF/指数基金"
    if any(keyword in text for keyword in mixed_keywords):
        return "偏股基金"
    return "主动权益基金"


def percentile_score(series: pd.Series, ascending: bool = True) -> pd.Series:
    valid = pd.to_numeric(series, errors="coerce")
    if valid.notna().sum() == 0:
        return pd.Series(50.0, index=series.index, dtype=float)
    ranks = valid.rank(pct=True, ascending=ascending, method="average")
    return ranks.fillna(0.5) * 100


def build_fund_scored_dataframe(fund_df: pd.DataFrame) -> pd.DataFrame:
    if fund_df.empty:
        return fund_df

    result = fund_df.copy()
    result["基金类别"] = result["基金简称"].map(classify_fund_category)
    result["fee_value"] = result["手续费"].map(parse_fee_value)
    result["fee_score"] = percentile_score(result["fee_value"], ascending=False)

    result["score_1m"] = percentile_score(result["近1月"])
    result["score_3m"] = percentile_score(result["近3月"])
    result["score_6m"] = percentile_score(result["近6月"])
    result["score_1y"] = percentile_score(result["近1年"])

    stability_raw = -(
        (pd.to_numeric(result["近6月"], errors="coerce") - pd.to_numeric(result["近3月"], errors="coerce") * 2).abs()
        + (pd.to_numeric(result["近3月"], errors="coerce") - pd.to_numeric(result["近1月"], errors="coerce") * 3).abs() / 3
    )
    result["stability_score"] = percentile_score(stability_raw)

    result["基金评分"] = 0.0
    category_weights = {
        "偏股基金": {"score_3m": 0.25, "score_6m": 0.35, "score_1y": 0.35, "fee_score": 0.05},
        "主动权益基金": {"score_1m": 0.10, "score_3m": 0.20, "score_6m": 0.40, "score_1y": 0.25, "fee_score": 0.05},
        "ETF/指数基金": {"score_1m": 0.15, "score_3m": 0.20, "score_6m": 0.25, "score_1y": 0.35, "fee_score": 0.05},
        "债基": {"score_1m": 0.20, "score_3m": 0.25, "score_6m": 0.20, "score_1y": 0.15, "stability_score": 0.15, "fee_score": 0.05},
    }
    for category, weights in category_weights.items():
        mask = result["基金类别"] == category
        if not mask.any():
            continue
        category_score = pd.Series(0.0, index=result.index, dtype=float)
        for column, weight in weights.items():
            category_score = category_score + result[column].fillna(50.0) * weight
        result.loc[mask, "基金评分"] = category_score.loc[mask]

    result["基础评分"] = result["基金评分"]

    return result.sort_values(["基金类别", "基金评分", "近1年", "近6月"], ascending=[True, False, False, False], na_position="last")


def build_fund_action_signals(fund_df: pd.DataFrame) -> pd.DataFrame:
    if fund_df.empty:
        return fund_df

    result = fund_df.copy()
    for column in ["基金评分", "近1月", "近3月", "近6月", "近1年"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    category_rules = {
        "偏股基金": {"buy_score": 78.0, "strong_buy_score": 86.0, "sell_score": 48.0, "clear_score": 40.0},
        "主动权益基金": {"buy_score": 80.0, "strong_buy_score": 88.0, "sell_score": 45.0, "clear_score": 38.0},
        "ETF/指数基金": {"buy_score": 75.0, "strong_buy_score": 84.0, "sell_score": 45.0, "clear_score": 38.0},
        "债基": {"buy_score": 68.0, "strong_buy_score": 78.0, "sell_score": 40.0, "clear_score": 34.0},
    }

    actions: list[str] = []
    execution_actions: list[str] = []
    reasons: list[str] = []
    action_ranks: list[int] = []
    action_order = {"清仓": 0, "减仓": 1, "观察": 2, "定投": 3, "一次性买入": 4}

    for row in result.itertuples(index=False):
        category = str(getattr(row, "基金类别", "主动权益基金") or "主动权益基金")
        rule = category_rules.get(category, category_rules["主动权益基金"])
        score = float(getattr(row, "基金评分", float("nan"))) if pd.notna(getattr(row, "基金评分", pd.NA)) else float("nan")
        return_1m = float(getattr(row, "近1月", float("nan"))) if pd.notna(getattr(row, "近1月", pd.NA)) else float("nan")
        return_3m = float(getattr(row, "近3月", float("nan"))) if pd.notna(getattr(row, "近3月", pd.NA)) else float("nan")
        return_6m = float(getattr(row, "近6月", float("nan"))) if pd.notna(getattr(row, "近6月", pd.NA)) else float("nan")
        return_1y = float(getattr(row, "近1年", float("nan"))) if pd.notna(getattr(row, "近1年", pd.NA)) else float("nan")

        strong_trend = all(pd.notna(value) and value > 0 for value in [return_1m, return_3m, return_6m])
        very_strong_trend = all(pd.notna(value) for value in [return_1m, return_3m, return_6m]) and (
            (category == "债基" and return_1m >= 0.3 and return_3m >= 1.0 and return_6m >= 2.0)
            or (category != "债基" and return_1m >= 1.0 and return_3m >= 3.0 and return_6m >= 8.0)
        )
        weak_trend = (
            (pd.notna(return_1m) and pd.notna(return_3m) and return_1m < 0 and return_3m < 0)
            or (pd.notna(return_3m) and pd.notna(return_6m) and return_3m < 0 and return_6m < 0)
            or (pd.notna(return_1y) and return_1y < -5)
        )
        severe_weakness = (
            (pd.notna(return_1m) and pd.notna(return_3m) and pd.notna(return_6m) and return_1m < 0 and return_3m < 0 and return_6m < 0)
            or (pd.notna(return_1y) and return_1y < -10)
        )
        score_high = pd.notna(score) and score >= rule["buy_score"]
        score_low = pd.notna(score) and score <= rule["sell_score"]
        score_very_high = pd.notna(score) and score >= rule["strong_buy_score"]
        score_very_low = pd.notna(score) and score <= rule["clear_score"]

        if score_low or weak_trend:
            action = "卖出"
            execution_action = "清仓" if score_very_low or severe_weakness else "减仓"
            reason_parts = []
            if score_low:
                reason_parts.append(f"评分偏低({score:.1f})")
            if pd.notna(return_1m) and pd.notna(return_3m) and return_1m < 0 and return_3m < 0:
                reason_parts.append("近1月和近3月同步转弱")
            elif pd.notna(return_3m) and pd.notna(return_6m) and return_3m < 0 and return_6m < 0:
                reason_parts.append("近3月和近6月同步走弱")
            elif pd.notna(return_1y) and return_1y < -5:
                reason_parts.append("近1年回报明显偏弱")
            reason = "，".join(reason_parts) or "评分与趋势共振走弱"
        elif score_high and strong_trend:
            action = "买入"
            execution_action = "一次性买入" if score_very_high and very_strong_trend else "定投"
            reason_parts = [f"评分靠前({score:.1f})", "近1/3/6月收益保持正向"]
            if pd.notna(return_1y) and return_1y > 0:
                reason_parts.append("近1年仍保持正收益")
            if execution_action == "一次性买入":
                reason_parts.append("强度足以支持更主动建仓")
            else:
                reason_parts.append("适合分批进入而非一次性重仓")
            reason = "，".join(reason_parts)
        else:
            action = "观察"
            execution_action = "观察"
            reason_parts = []
            if pd.notna(score):
                reason_parts.append(f"评分中性({score:.1f})")
            if strong_trend and not score_high:
                reason_parts.append("收益趋势尚可，但强度不够")
            elif not weak_trend:
                reason_parts.append("等待收益与评分进一步确认")
            reason = "，".join(reason_parts) or "等待进一步确认"

        actions.append(action)
        execution_actions.append(execution_action)
        reasons.append(reason)
        action_ranks.append(action_order.get(execution_action, 99))

    result["操作建议"] = pd.Series(actions, index=result.index, dtype="object")
    result["建议动作"] = pd.Series(execution_actions, index=result.index, dtype="object")
    result["判断依据"] = pd.Series(reasons, index=result.index, dtype="object")
    result["action_rank"] = pd.Series(action_ranks, index=result.index, dtype=int)
    return result.sort_values(["基金类别", "基金评分", "近1年", "近6月"], ascending=[True, False, False, False], na_position="last")


def render_fund_category_table(fund_df: pd.DataFrame, category: str, row_limit: int) -> None:
    category_df = fund_df.loc[fund_df["基金类别"] == category].copy()
    category_df = category_df.sort_values(["基金评分", "近1年", "近6月"], ascending=[False, False, False], na_position="last")
    if category_df.empty:
        st.info(f"当前没有可展示的{category}数据。")
        return

    category_rules = {
        "偏股基金": "评分更看重 3个月/6个月/1年持续收益，适合筛选股票仓位较高的混合型基金。",
        "主动权益基金": "评分更看重 6个月/1年中期胜率，同时保留少量 1个月强度用于识别景气延续。",
        "ETF/指数基金": "评分更强调中长期涨幅、低手续费、低折溢价和较小日收益偏离代理，适合指数化配置与轮动筛选。",
        "债基": "评分更强调短中期收益稳定性、低费率、近1年低回撤与低波动，避免只按高收益追债基。",
    }
    st.caption(category_rules.get(category, ""))

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("数量", int(len(category_df)))
    metric2.metric("平均评分", format_two_decimals(category_df["基金评分"].mean()))
    metric3.metric("近6月均值", format_percent_value(category_df["近6月"].mean()))
    metric4.metric("近1年均值", format_percent_value(category_df["近1年"].mean()))

    action_counts = category_df["建议动作"].value_counts() if "建议动作" in category_df.columns else pd.Series(dtype=int)
    action1, action2, action3, action4 = st.columns(4)
    action1.metric("一次性买入", int(action_counts.get("一次性买入", 0)))
    action2.metric("定投", int(action_counts.get("定投", 0)))
    action3.metric("观察", int(action_counts.get("观察", 0)))
    action4.metric("减仓/清仓", int(action_counts.get("减仓", 0) + action_counts.get("清仓", 0)))

    display_columns = ["建议动作", "操作建议", "基金评分", "基金代码", "基金简称", "日期", "单位净值", "累计净值", "近1月", "近3月", "近6月", "近1年", "手续费", "判断依据"]
    if category == "债基":
        display_columns.extend(["最大回撤_1y", "波动率_1y"])
    if category == "ETF/指数基金":
        display_columns.extend(["折价率", "日收益偏离代理"])
    display_df = category_df[[column for column in display_columns if column in category_df.columns]].head(row_limit).copy()
    if "日期" in display_df.columns:
        display_df["日期"] = display_df["日期"].map(lambda value: value.strftime("%Y-%m-%d") if pd.notna(value) else "-")
    if "基金评分" in display_df.columns:
        display_df["基金评分"] = display_df["基金评分"].map(format_two_decimals)
    for column in ["单位净值", "累计净值"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(format_two_decimals)
    for column in ["近1月", "近3月", "近6月", "近1年"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(format_percent_value)
    for column in ["最大回撤_1y", "波动率_1y", "折价率", "日收益偏离代理"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(format_percent_value)

    display_df = display_df.rename(
        columns={
            "最大回撤_1y": "近1年最大回撤",
            "波动率_1y": "近1年波动率",
            "折价率": "当前折溢价率",
            "日收益偏离代理": "日收益偏离代理",
        }
    )

    st.dataframe(display_df, width="stretch", hide_index=True)


def render_fund_watchlist_section(fund_df: pd.DataFrame) -> None:
    st.subheader("自选基金")

    default_codes = st.session_state.get("fund_watchlist_codes")
    if default_codes is None:
        default_codes = load_fund_watchlist_codes()
        st.session_state.fund_watchlist_codes = default_codes

    input_col, action_col = st.columns([2.2, 1])
    raw_watchlist = input_col.text_area(
        "输入自选基金代码",
        value="\n".join(default_codes),
        height=110,
        placeholder="每行一个基金代码，或用逗号分隔，例如：008528, 511010, 021694",
    )
    save_watchlist = action_col.button("保存基金自选", width="stretch")

    parsed_codes = parse_fund_watchlist_codes(raw_watchlist)
    if save_watchlist:
        save_fund_watchlist_codes(parsed_codes)
        st.session_state.fund_watchlist_codes = parsed_codes
        st.success(f"已保存 {len(parsed_codes)} 只自选基金。")

    watchlist_codes = parsed_codes if raw_watchlist.strip() else st.session_state.get("fund_watchlist_codes", [])
    st.session_state.fund_watchlist_codes = watchlist_codes
    if not watchlist_codes:
        st.info("在这里填入你建仓后想持续观察的基金代码，基金页会单独列出。")
        return

    watchlist_df = fund_df.loc[fund_df["基金代码"].astype(str).isin(watchlist_codes)].copy()
    missing_codes = [code for code in watchlist_codes if code not in set(watchlist_df["基金代码"].astype(str))]
    if watchlist_df.empty:
        st.warning("当前自选基金没有匹配到可展示的数据。")
        if missing_codes:
            st.caption(f"未匹配到的基金代码：{', '.join(missing_codes)}")
        return

    watchlist_df["watchlist_order"] = watchlist_df["基金代码"].astype(str).map({code: index for index, code in enumerate(watchlist_codes)})
    watchlist_df = watchlist_df.sort_values(["watchlist_order", "action_rank", "基金评分"], ascending=[True, True, False], na_position="last")
    watchlist_df["当前价格"] = pd.to_numeric(watchlist_df.get("市价"), errors="coerce")

    display_columns = ["基金类别", "建议动作", "操作建议", "基金评分", "基金代码", "基金简称", "日期", "当前价格", "单位净值", "累计净值", "近1月", "近3月", "近6月", "近1年", "手续费", "最大回撤_1y", "波动率_1y", "折价率", "日收益偏离代理", "判断依据"]
    display_df = watchlist_df[[column for column in display_columns if column in watchlist_df.columns]].copy()
    if "日期" in display_df.columns:
        display_df["日期"] = display_df["日期"].map(lambda value: value.strftime("%Y-%m-%d") if pd.notna(value) else "-")
    for column in ["基金评分", "当前价格", "单位净值", "累计净值"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(format_two_decimals)
    for column in ["近1月", "近3月", "近6月", "近1年", "最大回撤_1y", "波动率_1y", "折价率", "日收益偏离代理"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(format_percent_value)

    display_df = display_df.rename(
        columns={
            "基金类别": "基金类型",
            "建议动作": "建议动作",
            "操作建议": "操作",
            "基金评分": "评分",
            "最大回撤_1y": "近1年最大回撤",
            "波动率_1y": "近1年波动率",
            "折价率": "当前折溢价率",
            "日收益偏离代理": "日收益偏离代理",
        }
    )
    display_df = drop_uniform_display_columns(display_df, ["近1年最大回撤", "近1年波动率", "当前折溢价率", "日收益偏离代理"])

    st.caption(
        f"当前已跟踪 {len(watchlist_df)} 只自选基金。这里会结合评分和近 1/3/6 月收益结构，细化为一次性买入、定投、观察、减仓或清仓。"
        " 当前价格列只在能拿到实时市价时显示，拿不到则留空。"
    )
    if missing_codes:
        st.caption(f"未匹配到的基金代码：{', '.join(missing_codes)}")
    st.dataframe(display_df, width="stretch", hide_index=True)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_fund_rank() -> tuple[pd.DataFrame, str, str | None]:
    try:
        fund_df = ak.fund_open_fund_rank_em(symbol="全部")
        result = normalize_fund_rank_dataframe(fund_df)
        if result.empty:
            raise RuntimeError("基金接口返回空结果")
        result.to_csv(FUND_RANK_CACHE_PATH, index=False, encoding="utf-8-sig")
        return result, "实时接口", None
    except Exception as error:
        if FUND_RANK_CACHE_PATH.exists():
            cached_df = pd.read_csv(FUND_RANK_CACHE_PATH)
            result = normalize_fund_rank_dataframe(cached_df)
            if not result.empty:
                return result, "本地缓存", str(error)
        return pd.DataFrame(), "无可用数据", str(error)


@st.cache_data(ttl=21600, show_spinner=False)
def cached_debt_fund_risk_metrics(symbols_key: str) -> pd.DataFrame:
    symbols = [item.strip() for item in symbols_key.splitlines() if item.strip()]
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        try:
            history = ak.fund_open_fund_info_em(symbol=symbol, indicator="单位净值走势", period="近1年")
        except Exception:
            continue
        if history is None or history.empty or "单位净值" not in history.columns:
            continue
        history = history.copy()
        history["净值日期"] = pd.to_datetime(history["净值日期"], errors="coerce")
        history = history.sort_values("净值日期")
        nav_series = pd.to_numeric(history["单位净值"], errors="coerce")
        median_days = history["净值日期"].diff().dt.days.dropna().median()
        annual_factor = 52.0 if pd.notna(median_days) and median_days >= 4 else 252.0
        rows.append(
            {
                "基金代码": normalize_fund_code(symbol),
                "最大回撤_1y": compute_max_drawdown(nav_series),
                "波动率_1y": compute_annualized_volatility(nav_series, annual_factor),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_etf_daily_metrics() -> pd.DataFrame:
    try:
        daily_df = ak.fund_etf_fund_daily_em()
    except Exception:
        return pd.DataFrame()
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()

    result = normalize_etf_daily_dataframe(daily_df)
    if result.empty:
        return result
    if "折价率" in result.columns:
        result["折价率"] = pd.to_numeric(result["折价率"], errors="coerce")
        result["绝对折溢价率"] = result["折价率"].abs()
    return result


@st.cache_data(ttl=1800, show_spinner=False)
def cached_gold_history(symbol: str = "Au99.99") -> tuple[pd.DataFrame, str | None]:
    try:
        history = ak.spot_hist_sge(symbol=symbol)
    except Exception as error:
        return pd.DataFrame(), str(error)

    if history is None or history.empty:
        return pd.DataFrame(), "黄金行情接口返回空结果。"

    result = history.copy()
    result["date"] = pd.to_datetime(result.get("date"), errors="coerce")
    for column in ["open", "close", "low", "high"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return result, None


@st.cache_data(ttl=1800, show_spinner=False)
def cached_gold_benchmark() -> tuple[pd.DataFrame, str | None]:
    try:
        benchmark_df = ak.spot_golden_benchmark_sge()
    except Exception as error:
        return pd.DataFrame(), str(error)

    if benchmark_df is None or benchmark_df.empty:
        return pd.DataFrame(), "黄金基准价接口返回空结果。"

    result = benchmark_df.copy()
    result["交易时间"] = pd.to_datetime(result.get("交易时间"), errors="coerce")
    for column in ["晚盘价", "早盘价"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["交易时间"]).sort_values("交易时间").reset_index(drop=True)
    return result, None


def classify_gold_etf_category(name: object) -> str | None:
    text = str(name or "").strip()
    if not text:
        return None
    if "黄金股ETF" in text:
        return "黄金股ETF"
    if any(keyword in text for keyword in ["黄金ETF", "上海金ETF", "金ETF"]):
        if "天天金ETF" in text:
            return None
        return "实物黄金ETF"
    return None


@st.cache_data(ttl=900, show_spinner=False)
def cached_gold_etf_spot() -> tuple[pd.DataFrame, str | None]:
    try:
        snapshot_df = ak.fund_etf_spot_em()
    except Exception as error:
        return pd.DataFrame(), str(error)

    if snapshot_df is None or snapshot_df.empty:
        return pd.DataFrame(), "黄金 ETF 接口返回空结果。"

    result = snapshot_df.copy()
    rename_map = {"代码": "基金代码", "名称": "基金简称"}
    result = result.rename(columns={key: value for key, value in rename_map.items() if key in result.columns})
    if "基金代码" in result.columns:
        result["基金代码"] = result["基金代码"].map(normalize_fund_code)
    if "基金简称" not in result.columns:
        return pd.DataFrame(), "黄金 ETF 接口缺少名称列。"

    result["黄金ETF类型"] = result["基金简称"].map(classify_gold_etf_category)
    result = result[result["黄金ETF类型"].notna()].copy()
    if result.empty:
        return pd.DataFrame(), "当前 ETF 行情里没有匹配到黄金相关标的。"

    for column in ["最新价", "IOPV实时估值", "基金折价率", "涨跌额", "涨跌幅", "成交额", "成交量", "开盘价", "最高价", "最低价", "昨收"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if "更新时间" in result.columns:
        result["更新时间"] = pd.to_datetime(result["更新时间"], errors="coerce")
    if {"最新价", "IOPV实时估值"}.issubset(result.columns):
        result["相对估值偏离"] = ((result["最新价"] / result["IOPV实时估值"]) - 1) * 100
    if {"最新价", "昨收"}.issubset(result.columns):
        result["日内振幅代理"] = ((result["最新价"] / result["昨收"]) - 1) * 100
    result = result.sort_values(["黄金ETF类型", "成交额", "涨跌幅"], ascending=[True, False, False], na_position="last").reset_index(drop=True)
    return result, None


def build_gold_outlook(history: pd.DataFrame) -> dict[str, object]:
    if history.empty or len(history) < 60:
        return {
            "label": "数据不足",
            "score": 0.0,
            "confidence": "低",
            "summary": "黄金历史数据不足，暂时无法给出未来两周的规则化判断。",
            "upside_probability": None,
            "expected_move_pct": None,
            "range_low": None,
            "range_high": None,
            "return_5d": None,
            "return_10d": None,
            "return_20d": None,
            "close": None,
            "ma10": None,
            "ma20": None,
            "ma60": None,
            "support_level": None,
            "resistance_level": None,
            "support_gap_pct": None,
            "resistance_gap_pct": None,
            "level_comment": "",
        }

    working = history.copy()
    working["ma10"] = working["close"].rolling(10).mean()
    working["ma20"] = working["close"].rolling(20).mean()
    working["ma60"] = working["close"].rolling(60).mean()
    working["pct_change"] = working["close"].pct_change()
    latest = working.iloc[-1]

    close = float(latest["close"])
    ma10 = float(latest["ma10"]) if pd.notna(latest["ma10"]) else None
    ma20 = float(latest["ma20"]) if pd.notna(latest["ma20"]) else None
    ma60 = float(latest["ma60"]) if pd.notna(latest["ma60"]) else None
    return_5d = compute_trailing_return(working["close"], 5)
    return_10d = compute_trailing_return(working["close"], 10)
    return_20d = compute_trailing_return(working["close"], 20)
    recent_low_10 = pd.to_numeric(working["low"], errors="coerce").dropna().tail(10).min()
    recent_low_20 = pd.to_numeric(working["low"], errors="coerce").dropna().tail(20).min()
    recent_low_60 = pd.to_numeric(working["low"], errors="coerce").dropna().tail(60).min()
    recent_high_10 = pd.to_numeric(working["high"], errors="coerce").dropna().tail(10).max()
    recent_high_20 = pd.to_numeric(working["high"], errors="coerce").dropna().tail(20).max()
    recent_high_60 = pd.to_numeric(working["high"], errors="coerce").dropna().tail(60).max()

    support_candidates = [
        value
        for value in [ma10, ma20, ma60, recent_low_10, recent_low_20, recent_low_60]
        if isinstance(value, (int, float)) and pd.notna(value) and float(value) <= close
    ]
    resistance_candidates = [
        value
        for value in [recent_high_10, recent_high_20, recent_high_60]
        if isinstance(value, (int, float)) and pd.notna(value) and float(value) >= close
    ]
    support_level = max(float(value) for value in support_candidates) if support_candidates else None
    resistance_level = min(float(value) for value in resistance_candidates) if resistance_candidates else None
    support_gap_pct = ((close / support_level) - 1) if support_level not in (None, 0) else None
    resistance_gap_pct = ((resistance_level / close) - 1) if resistance_level not in (None, 0) else None

    recent_vol = pd.to_numeric(working["pct_change"], errors="coerce").dropna().tail(20)
    expected_move_pct = float(recent_vol.std()) * (10 ** 0.5) if len(recent_vol) >= 5 else None
    range_low = close * (1 - expected_move_pct) if expected_move_pct is not None else None
    range_high = close * (1 + expected_move_pct) if expected_move_pct is not None else None

    score = 0.0
    reason_parts: list[str] = []
    if ma10 is not None:
        if close > ma10:
            score += 0.8
            reason_parts.append("现价站上 MA10")
        else:
            score -= 0.8
            reason_parts.append("现价跌回 MA10 下方")
    if ma20 is not None:
        if close > ma20:
            score += 1.0
            reason_parts.append("现价高于 MA20")
        else:
            score -= 1.0
            reason_parts.append("现价低于 MA20")
    if ma60 is not None:
        if close > ma60:
            score += 1.2
            reason_parts.append("中期趋势仍在 MA60 上方")
        else:
            score -= 1.2
            reason_parts.append("中期趋势已落到 MA60 下方")
    if ma20 is not None and ma60 is not None:
        if ma20 > ma60:
            score += 0.8
            reason_parts.append("MA20 仍压着 MA60 向上")
        else:
            score -= 0.8
            reason_parts.append("MA20 未能维持在 MA60 上方")

    for horizon, value, up_threshold, down_threshold in [
        ("5日", return_5d, 0.01, -0.01),
        ("10日", return_10d, 0.02, -0.02),
        ("20日", return_20d, 0.035, -0.035),
    ]:
        if value is None:
            continue
        if value >= up_threshold:
            score += 0.7
            reason_parts.append(f"近{horizon}动量偏强({value:+.2%})")
        elif value <= down_threshold:
            score -= 0.7
            reason_parts.append(f"近{horizon}动量偏弱({value:+.2%})")

    if pd.notna(recent_high_20) and close >= float(recent_high_20) * 0.995:
        score += 0.5
        reason_parts.append("接近近20日高位，趋势有延续性")
    if pd.notna(recent_low_20) and close <= float(recent_low_20) * 1.005:
        score -= 0.5
        reason_parts.append("接近近20日低位，短线承压")

    if score >= 3.5:
        label = "看多"
    elif score >= 1.5:
        label = "偏多"
    elif score <= -3.5:
        label = "看空"
    elif score <= -1.5:
        label = "偏空"
    else:
        label = "震荡"

    confidence_factors = 0
    if ma20 is not None and ma60 is not None:
        confidence_factors += 1
    if return_10d is not None and return_20d is not None:
        confidence_factors += 1
    if expected_move_pct is not None:
        confidence_factors += 1
    if abs(score) >= 2.0:
        confidence_factors += 1
    confidence = {0: "低", 1: "中", 2: "中高", 3: "高", 4: "高"}.get(confidence_factors, "中")

    level_comment = ""
    if support_gap_pct is not None and support_gap_pct <= 0.015:
        level_comment = f"当前价距离短线支撑仅约 {support_gap_pct:.2%}，若跌破 {support_level:.2f}，两周判断容易转弱。"
    elif resistance_gap_pct is not None and resistance_gap_pct <= 0.015:
        level_comment = f"当前价距离上方压力仅约 {resistance_gap_pct:.2%}，若放量突破 {resistance_level:.2f}，两周趋势有望转强。"
    elif support_level is not None and resistance_level is not None:
        level_comment = f"当前主要运行区间可先看 {support_level:.2f} - {resistance_level:.2f}。"

    upside_probability = min(max(0.5 + score * 0.07, 0.2), 0.8)
    move_text = (
        f"按近20日波动率估算，未来两周常规波动区间大约在 {range_low:.2f} - {range_high:.2f}。"
        if range_low is not None and range_high is not None
        else "当前波动样本不足，暂不输出两周区间估算。"
    )
    reason_text = "；".join(reason_parts[:5]) if reason_parts else "暂未形成清晰趋势特征。"
    summary = f"未来两周黄金规则模型判断偏向{label}，上行概率估计约 {upside_probability:.0%}。{reason_text}。{move_text} {level_comment}".strip()

    return {
        "label": label,
        "score": round(score, 2),
        "confidence": confidence,
        "summary": summary,
        "upside_probability": upside_probability,
        "expected_move_pct": expected_move_pct,
        "range_low": range_low,
        "range_high": range_high,
        "return_5d": return_5d,
        "return_10d": return_10d,
        "return_20d": return_20d,
        "close": close,
        "ma10": ma10,
        "ma20": ma20,
        "ma60": ma60,
        "support_level": support_level,
        "resistance_level": resistance_level,
        "support_gap_pct": support_gap_pct,
        "resistance_gap_pct": resistance_gap_pct,
        "level_comment": level_comment,
    }


def build_gold_figure(
    history: pd.DataFrame,
    support_level: float | None = None,
    resistance_level: float | None = None,
    range_low: float | None = None,
    range_high: float | None = None,
    projection_days: int = 10,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Candlestick(
            x=history["date"],
            open=history["open"],
            high=history["high"],
            low=history["low"],
            close=history["close"],
            name="Au99.99",
        )
    )
    for column, label, color in [("ma10", "MA10", "#d97706"), ("ma20", "MA20", "#0f766e"), ("ma60", "MA60", "#1d4ed8")]:
        if column in history.columns:
            figure.add_trace(
                go.Scatter(
                    x=history["date"],
                    y=history[column],
                    mode="lines",
                    name=label,
                    line={"width": 2, "color": color},
                )
            )
    if (
        range_low is not None
        and range_high is not None
        and not history.empty
        and projection_days > 0
    ):
        latest_date = pd.to_datetime(history["date"]).max()
        projection_end = latest_date + pd.offsets.BDay(projection_days)
        band_label_y = float(range_high) - ((float(range_high) - float(range_low)) * 0.08)
        figure.add_shape(
            type="rect",
            x0=latest_date,
            x1=projection_end,
            y0=float(range_low),
            y1=float(range_high),
            xref="x",
            yref="y",
            line={"width": 0},
            fillcolor="rgba(245, 158, 11, 0.18)",
            layer="below",
        )
        figure.add_trace(
            go.Scatter(
                x=[latest_date, projection_end],
                y=[float(range_low), float(range_low)],
                mode="lines",
                name="两周波动下沿",
                line={"width": 1.5, "color": "#f59e0b", "dash": "dot"},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[latest_date, projection_end],
                y=[float(range_high), float(range_high)],
                mode="lines",
                name="两周波动上沿",
                line={"width": 1.5, "color": "#f59e0b", "dash": "dot"},
            )
        )
        figure.add_annotation(
            x=projection_end,
            y=band_label_y,
            text=f"两周波动带 {float(range_low):.2f} - {float(range_high):.2f}",
            showarrow=False,
            xanchor="right",
            font={"size": 11, "color": "#92400e"},
            bgcolor="rgba(255, 247, 237, 0.9)",
            bordercolor="rgba(245, 158, 11, 0.35)",
            borderwidth=1,
        )
    if support_level is not None:
        figure.add_hline(
            y=float(support_level),
            line_width=1.5,
            line_dash="dash",
            line_color="#16a34a",
            annotation_text=f"支撑 {float(support_level):.2f}",
            annotation_position="bottom right",
        )
    if resistance_level is not None:
        figure.add_hline(
            y=float(resistance_level),
            line_width=1.5,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=f"压力 {float(resistance_level):.2f}",
            annotation_position="top right",
        )
    figure.update_layout(
        height=620,
        margin={"l": 12, "r": 12, "t": 24, "b": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    if (
        range_low is not None
        and range_high is not None
        and not history.empty
        and projection_days > 0
    ):
        latest_date = pd.to_datetime(history["date"]).max()
        projection_end = latest_date + pd.offsets.BDay(projection_days)
        first_date = pd.to_datetime(history["date"]).min()
        figure.update_xaxes(range=[first_date, projection_end])
    return figure


def render_gold_section() -> None:
    st.markdown(GOLD_SECTION_CSS, unsafe_allow_html=True)
    st.subheader("黄金走势与两周展望")
    st.caption("数据源：上海金 Au99.99 日线。这里的“两周预测”是基于趋势、动量与波动率的规则模型预估，不是事件驱动预测。")

    with st.spinner("正在加载黄金行情..."):
        history, history_error = cached_gold_history("Au99.99")
        benchmark_df, benchmark_error = cached_gold_benchmark()

    if history.empty:
        st.warning("当前没有拿到可展示的黄金日线数据。")
        if history_error:
            st.caption(f"黄金数据失败原因：{history_error}")
        return

    history = history.copy()
    history["ma10"] = history["close"].rolling(10).mean()
    history["ma20"] = history["close"].rolling(20).mean()
    history["ma60"] = history["close"].rolling(60).mean()
    outlook = build_gold_outlook(history)
    latest_date = pd.to_datetime(history["date"]).max().date()
    status_label = (
        "临近支撑"
        if isinstance(outlook.get("support_gap_pct"), (int, float)) and outlook["support_gap_pct"] <= 0.015
        else "临近压力"
        if isinstance(outlook.get("resistance_gap_pct"), (int, float)) and outlook["resistance_gap_pct"] <= 0.015
        else "区间中部"
    )
    st.markdown(
        f"""
        <div class="gold-hero">
            <div class="gold-hero-title">黄金两周判断</div>
            <div class="gold-hero-main">
                <div>
                    <div class="gold-hero-label">{str(outlook.get('label', '-'))}</div>
                    <div class="gold-hero-meta">最新日线日期：{latest_date:%Y-%m-%d}</div>
                </div>
                <div class="gold-hero-meta">最新收盘：{format_two_decimals(outlook.get('close'))} | 判断把握：{str(outlook.get('confidence', '-'))} | 区间状态：{status_label}</div>
            </div>
            <div class="gold-hero-summary">{str(outlook.get('summary', ''))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview_tab, chart_tab, data_tab = st.tabs(["概览", "图表", "数据"])

    with overview_tab:
        metric1, metric2, metric3, metric4, metric5 = st.columns(5)
        metric1.metric("最新收盘", format_two_decimals(outlook.get("close")))
        metric2.metric("未来两周倾向", str(outlook.get("label", "-")))
        metric3.metric("上行概率估计", f"{outlook['upside_probability']:.0%}" if isinstance(outlook.get("upside_probability"), (int, float)) else "-")
        metric4.metric("两周波动参考", f"{outlook['expected_move_pct']:.2%}" if isinstance(outlook.get("expected_move_pct"), (int, float)) else "-")
        metric5.metric("判断把握", str(outlook.get("confidence", "-")))

        trend1, trend2, trend3, trend4 = st.columns(4)
        trend1.metric("近5日", f"{outlook['return_5d']:+.2%}" if isinstance(outlook.get("return_5d"), (int, float)) else "-")
        trend2.metric("近10日", f"{outlook['return_10d']:+.2%}" if isinstance(outlook.get("return_10d"), (int, float)) else "-")
        trend3.metric("近20日", f"{outlook['return_20d']:+.2%}" if isinstance(outlook.get("return_20d"), (int, float)) else "-")
        trend4.metric(
            "两周参考区间",
            f"{outlook['range_low']:.2f} - {outlook['range_high']:.2f}"
            if isinstance(outlook.get("range_low"), (int, float)) and isinstance(outlook.get("range_high"), (int, float))
            else "-",
        )

        level1, level2, level3 = st.columns(3)
        level1.metric(
            "短线支撑位",
            format_two_decimals(outlook.get("support_level")),
            delta=(f"当前高出 {outlook['support_gap_pct']:.2%}" if isinstance(outlook.get("support_gap_pct"), (int, float)) else None),
        )
        level2.metric(
            "上方压力位",
            format_two_decimals(outlook.get("resistance_level")),
            delta=(f"距离上压 {outlook['resistance_gap_pct']:.2%}" if isinstance(outlook.get("resistance_gap_pct"), (int, float)) else None),
        )
        level3.metric("支撑/压力状态", status_label)

        if str(outlook.get("level_comment", "")).strip():
            st.caption(str(outlook.get("level_comment", "")))

        if benchmark_df.empty:
            if benchmark_error:
                st.caption(f"黄金基准价补充数据暂不可用：{benchmark_error}")
        else:
            st.caption("黄金基准价")
            latest_benchmark = benchmark_df.iloc[-1]
            bench_col1, bench_col2, bench_col3 = st.columns(3)
            bench_col1.metric("黄金基准日期", pd.to_datetime(latest_benchmark['交易时间']).strftime("%Y-%m-%d"))
            bench_col2.metric("基准早盘价", format_two_decimals(latest_benchmark.get("早盘价")))
            bench_col3.metric("基准晚盘价", format_two_decimals(latest_benchmark.get("晚盘价")))

    with chart_tab:
        view_options = {"近6月": 120, "近1年": 250, "近2年": 500, "全部": None}
        control_col1, control_col2 = st.columns([1, 1.2])
        selected_view = control_col1.selectbox("图表区间", options=list(view_options.keys()), index=1)
        show_table_in_chart = control_col2.checkbox("图表下方显示最近20行日线", value=False)

        lookback_days = view_options[selected_view]
        display_history = history.tail(lookback_days).copy() if lookback_days else history.copy()
        st.plotly_chart(
            build_gold_figure(
                display_history,
                support_level=outlook.get("support_level"),
                resistance_level=outlook.get("resistance_level"),
                range_low=outlook.get("range_low"),
                range_high=outlook.get("range_high"),
            ),
            width="stretch",
        )
        st.caption("图中绿色虚线为短线支撑位，红色虚线为上方压力位，右侧橙色阴影为按近20日波动率估算的未来两周参考波动带。")

        if show_table_in_chart:
            table_df = history[[column for column in ["date", "open", "high", "low", "close", "ma10", "ma20", "ma60"] if column in history.columns]].tail(20).copy()
            table_df["date"] = table_df["date"].dt.strftime("%Y-%m-%d")
            for column in ["open", "high", "low", "close", "ma10", "ma20", "ma60"]:
                if column in table_df.columns:
                    table_df[column] = table_df[column].map(format_two_decimals)
            table_df = table_df.rename(
                columns={
                    "date": "日期",
                    "open": "开盘",
                    "high": "最高",
                    "low": "最低",
                    "close": "收盘",
                    "ma10": "MA10",
                    "ma20": "MA20",
                    "ma60": "MA60",
                }
            )
            st.dataframe(table_df, width="stretch", hide_index=True)

    with data_tab:
        data_table = history[[column for column in ["date", "open", "high", "low", "close", "ma10", "ma20", "ma60"] if column in history.columns]].tail(60).copy()
        data_table["date"] = data_table["date"].dt.strftime("%Y-%m-%d")
        for column in ["open", "high", "low", "close", "ma10", "ma20", "ma60"]:
            if column in data_table.columns:
                data_table[column] = data_table[column].map(format_two_decimals)
        data_table = data_table.rename(
            columns={
                "date": "日期",
                "open": "开盘",
                "high": "最高",
                "low": "最低",
                "close": "收盘",
                "ma10": "MA10",
                "ma20": "MA20",
                "ma60": "MA60",
            }
        )
        st.caption("最近60行黄金日线数据")
        st.dataframe(data_table, width="stretch", hide_index=True)


def enhance_fund_scores(fund_df: pd.DataFrame, shortlist_size: int) -> pd.DataFrame:
    if fund_df.empty:
        return fund_df

    result = fund_df.copy()

    debt_candidates = result.loc[result["基金类别"] == "债基"].nlargest(shortlist_size, "基金评分")
    if not debt_candidates.empty:
        debt_metrics = cached_debt_fund_risk_metrics("\n".join(debt_candidates["基金代码"].astype(str)))
        if not debt_metrics.empty:
            result = result.merge(debt_metrics, on="基金代码", how="left")
            result["drawdown_score"] = percentile_score(result["最大回撤_1y"], ascending=False)
            result["volatility_score"] = percentile_score(result["波动率_1y"], ascending=False)
            debt_mask = result["基金类别"] == "债基"
            result.loc[debt_mask, "基金评分"] = (
                result.loc[debt_mask, "基础评分"].fillna(50.0) * 0.70
                + result.loc[debt_mask, "drawdown_score"].fillna(50.0) * 0.15
                + result.loc[debt_mask, "volatility_score"].fillna(50.0) * 0.15
            )

    etf_metrics = cached_etf_daily_metrics()
    if not etf_metrics.empty:
        result = result.merge(
            etf_metrics[[column for column in ["基金代码", "类型", "市价", "折价率", "绝对折溢价率", "日收益偏离代理"] if column in etf_metrics.columns]],
            on="基金代码",
            how="left",
        )
        result["premium_score"] = percentile_score(result["绝对折溢价率"], ascending=False)
        result["tracking_proxy_score"] = percentile_score(result["日收益偏离代理"], ascending=False)
        etf_mask = result["基金类别"] == "ETF/指数基金"
        result.loc[etf_mask, "基金评分"] = (
            result.loc[etf_mask, "基础评分"].fillna(50.0) * 0.75
            + result.loc[etf_mask, "premium_score"].fillna(50.0) * 0.15
            + result.loc[etf_mask, "tracking_proxy_score"].fillna(50.0) * 0.10
        )

    return result.sort_values(["基金类别", "基金评分", "近1年", "近6月"], ascending=[True, False, False, False], na_position="last")


def build_intraday_context(history: pd.DataFrame, snapshot: dict[str, object] | None) -> dict[str, object] | None:
    if snapshot is None or history.empty:
        return None

    latest_row = history.iloc[-1]
    latest_close = float(latest_row["close"])
    live_price = float(snapshot["last_price"])
    pct_from_close = (live_price / latest_close) - 1 if latest_close else 0.0
    ma_fast = float(latest_row["ma_fast"]) if pd.notna(latest_row["ma_fast"]) else None
    breakout_high = float(latest_row["breakout_high"]) if pd.notna(latest_row["breakout_high"]) else None
    above_ma_fast = ma_fast is not None and live_price > ma_fast
    above_breakout = breakout_high is not None and live_price >= breakout_high
    strong_gap = abs(pct_from_close) >= 0.03

    notes: list[str] = []
    if strong_gap:
        notes.append(f"当前价较昨收偏离 {pct_from_close:+.2%}。")
    if above_ma_fast and latest_close <= ma_fast:
        notes.append("盘中价格已重新站上 MA20。")
    if above_breakout and latest_close < breakout_high:
        notes.append("盘中价格已上穿最近突破位。")

    if notes:
        notes.append("但日线策略仍需等待今天收盘，量能与收盘站稳后才能确认是否从卖出/观察切回买入。")

    return {
        "symbol": snapshot["symbol"],
        "name": snapshot["name"],
        "live_price": live_price,
        "turnover_amount": float(snapshot["turnover_amount"]),
        "pct_from_close": pct_from_close,
        "ma_fast": ma_fast,
        "breakout_high": breakout_high,
        "above_ma_fast": above_ma_fast,
        "above_breakout": above_breakout,
        "message": " ".join(notes),
        "is_material": bool(notes),
    }


def get_intraday_prompt_config(config: dict[str, object] | None) -> dict[str, float]:
    prompt_cfg = dict((config or {}).get("intraday_prompt", {}) or {})
    return {
        "breakout_confirm_pct": float(prompt_cfg.get("breakout_confirm_pct", 0.003)),
        "breakout_chase_limit_pct": float(prompt_cfg.get("breakout_chase_limit_pct", 0.04)),
        "pullback_near_ma_pct": float(prompt_cfg.get("pullback_near_ma_pct", 0.015)),
        "pullback_breakout_buffer_pct": float(prompt_cfg.get("pullback_breakout_buffer_pct", 0.015)),
        "ma_break_buffer_pct": float(prompt_cfg.get("ma_break_buffer_pct", 0.005)),
        "breakout_fail_buffer_pct": float(prompt_cfg.get("breakout_fail_buffer_pct", 0.005)),
        "flat_intraday_pct": float(prompt_cfg.get("flat_intraday_pct", 0.01)),
        "scan_trial_min_live_delta_pct": float(prompt_cfg.get("scan_trial_min_live_delta_pct", 0.005)),
        "scan_observe_abs_delta_pct": float(prompt_cfg.get("scan_observe_abs_delta_pct", 0.03)),
        "scan_sell_confirm_delta_pct": float(prompt_cfg.get("scan_sell_confirm_delta_pct", -0.01)),
        "trial_score_min": float(prompt_cfg.get("trial_score_min", 75)),
        "pullback_score_min": float(prompt_cfg.get("pullback_score_min", 70)),
    }


def resolve_intraday_prompt_config(
    config: dict[str, object] | None,
    preset: str = "跟随配置",
) -> dict[str, float]:
    base_config = get_intraday_prompt_config(config)
    preset_name = str(preset or "跟随配置").strip()
    if preset_name == "保守":
        return {
            **base_config,
            "breakout_confirm_pct": 0.003,
            "breakout_chase_limit_pct": 0.04,
            "pullback_near_ma_pct": 0.015,
            "pullback_breakout_buffer_pct": 0.015,
            "ma_break_buffer_pct": 0.005,
            "breakout_fail_buffer_pct": 0.005,
            "flat_intraday_pct": 0.01,
            "scan_trial_min_live_delta_pct": 0.005,
            "scan_observe_abs_delta_pct": 0.03,
            "scan_sell_confirm_delta_pct": -0.01,
            "trial_score_min": 75.0,
            "pullback_score_min": 70.0,
        }
    if preset_name == "积极":
        return {
            **base_config,
            "breakout_confirm_pct": 0.0015,
            "breakout_chase_limit_pct": 0.055,
            "pullback_near_ma_pct": 0.02,
            "pullback_breakout_buffer_pct": 0.02,
            "ma_break_buffer_pct": 0.007,
            "breakout_fail_buffer_pct": 0.007,
            "flat_intraday_pct": 0.008,
            "scan_trial_min_live_delta_pct": 0.003,
            "scan_observe_abs_delta_pct": 0.025,
            "scan_sell_confirm_delta_pct": -0.008,
            "trial_score_min": 70.0,
            "pullback_score_min": 65.0,
        }
    if preset_name == "超积极":
        return {
            **base_config,
            "breakout_confirm_pct": 0.001,
            "breakout_chase_limit_pct": 0.065,
            "pullback_near_ma_pct": 0.025,
            "pullback_breakout_buffer_pct": 0.025,
            "ma_break_buffer_pct": 0.009,
            "breakout_fail_buffer_pct": 0.009,
            "flat_intraday_pct": 0.006,
            "scan_trial_min_live_delta_pct": 0.002,
            "scan_observe_abs_delta_pct": 0.02,
            "scan_sell_confirm_delta_pct": -0.006,
            "trial_score_min": 65.0,
            "pullback_score_min": 60.0,
        }
    return base_config


def format_intraday_preset_caption(preset: str, prompt_config: dict[str, float]) -> str:
    return (
        f"盘中提示档位：{preset}"
        f" | 突破确认 +{prompt_config['breakout_confirm_pct']:.2%}"
        f" | 试买评分 >= {int(prompt_config['trial_score_min'])}"
    )


def get_intraday_preset_warning(preset: str) -> str | None:
    if str(preset or "").strip() != "超积极":
        return None
    return (
        "当前使用的是“超积极”档。该模式会明显放宽盘中试买条件，"
        "更容易提前上车，也更容易在震荡行情里出现误触发，不适合重仓追价。"
    )


def build_intraday_execution_prompt(
    history: pd.DataFrame,
    summary: dict[str, object] | None,
    snapshot: dict[str, object] | None,
    market_regime: str = "",
    prompt_config: dict[str, float] | None = None,
) -> dict[str, object] | None:
    intraday_context = build_intraday_context(history, snapshot)
    if intraday_context is None or summary is None:
        return None

    cfg = prompt_config or get_intraday_prompt_config(None)

    latest_row = history.iloc[-1]
    live_price = float(intraday_context["live_price"])
    latest_close = float(latest_row["close"])
    ma_fast = intraday_context["ma_fast"]
    breakout_high = intraday_context["breakout_high"]
    atr_stop_price = summary.get("atr_stop_price")
    position_state = str(summary.get("position_state", "空仓"))
    action = str(summary.get("action", "观察"))
    score = float(summary.get("score") or 0.0)
    normalized_regime = str(market_regime or "").strip()
    market_open = normalized_regime in {"", "风险开"}
    market_neutral_or_better = normalized_regime in {"", "风险开", "中性"}

    if position_state in {"待买入", "持仓", "持仓待加仓", "待卖出"}:
        retreat_reasons: list[str] = []
        if atr_stop_price is not None and live_price <= float(atr_stop_price):
            retreat_reasons.append(f"当前价已触及 ATR 风控价 {float(atr_stop_price):.2f}")
        if ma_fast is not None and live_price < ma_fast * (1 - cfg["ma_break_buffer_pct"]):
            retreat_reasons.append(f"当前价跌回 MA20 下方 ({ma_fast:.2f})")
        if breakout_high is not None and latest_close >= breakout_high and live_price < breakout_high * (1 - cfg["breakout_fail_buffer_pct"]):
            retreat_reasons.append(f"当前价重新跌回突破位下方 ({breakout_high:.2f})")
        if retreat_reasons:
            return {
                "action": "盘中撤退",
                "message": f"{'；'.join(retreat_reasons)}，若你采用盘中提前执行规则，应优先减仓或撤回试仓。",
                "strength": "high",
            }

    breakout_trial_ready = (
        breakout_high is not None
        and ma_fast is not None
        and position_state in {"空仓", "待买入"}
        and action in {"观察", "买入"}
        and score >= cfg["trial_score_min"]
        and live_price >= breakout_high * (1 + cfg["breakout_confirm_pct"])
        and live_price <= breakout_high * (1 + cfg["breakout_chase_limit_pct"])
        and live_price >= ma_fast
    )
    if breakout_trial_ready:
        if market_open:
            return {
                "action": "盘中试买",
                "message": (
                    f"当前价站上突破位 {breakout_high:.2f} 并高于 MA20 ({ma_fast:.2f})，"
                    "可按盘中提前执行规则先试 1/3 仓。"
                ),
                "strength": "high",
            }
        regime_text = format_market_regime_label(normalized_regime)
        return {
            "action": "盘中观察",
            "message": f"当前价已突破关键位，但市场环境为 {regime_text}，暂不建议盘中追入。",
            "strength": "medium",
        }

    pullback_trial_ready = (
        breakout_high is not None
        and ma_fast is not None
        and position_state in {"空仓", "待买入"}
        and action in {"观察", "买入"}
        and score >= cfg["pullback_score_min"]
        and live_price >= ma_fast
        and live_price <= ma_fast * (1 + cfg["pullback_near_ma_pct"])
        and live_price >= breakout_high * (1 - cfg["pullback_breakout_buffer_pct"])
        and live_price < breakout_high
    )
    if pullback_trial_ready:
        if market_neutral_or_better:
            return {
                "action": "盘中试买",
                "message": (
                    f"当前价围绕 MA20 ({ma_fast:.2f}) 附近强势整理，"
                    "若你采用盘中提前执行规则，可先小仓试买。"
                ),
                "strength": "medium",
            }
        regime_text = format_market_regime_label(normalized_regime)
        return {
            "action": "盘中观察",
            "message": f"当前更像回踩确认，但市场环境为 {regime_text}，先观察不抢。",
            "strength": "low",
        }

    if intraday_context["is_material"]:
        return {
            "action": "盘中观察",
            "message": intraday_context["message"],
            "strength": "low",
        }

    flat_distance = abs((live_price / latest_close) - 1) if latest_close else 0.0
    if flat_distance < cfg["flat_intraday_pct"]:
        return {
            "action": "-",
            "message": "当前盘中价格相对平稳，暂无额外提前执行提示。",
            "strength": "low",
        }

    return {
        "action": "盘中观察",
        "message": "当前盘中波动存在，但还没到提前执行阈值，继续等收盘确认。",
        "strength": "low",
    }


def build_scan_intraday_prompt(
    row: pd.Series,
    snapshot: dict[str, object] | None,
    market_regime: str = "",
    prompt_config: dict[str, float] | None = None,
) -> dict[str, object] | None:
    if snapshot is None:
        return None

    cfg = prompt_config or get_intraday_prompt_config(None)

    close = float(row["close"]) if pd.notna(row.get("close")) else None
    if close is None or close <= 0:
        return None

    live_price = float(snapshot["last_price"])
    live_delta = (live_price / close) - 1.0
    atr_stop_price = float(row["atr_stop_price"]) if pd.notna(row.get("atr_stop_price")) else None
    score = float(row.get("score") or 0.0)
    buy_signal = bool(row.get("buy_signal", False))
    add_on_signal = bool(row.get("add_on_signal", False))
    sell_signal = bool(row.get("sell_signal", False))
    position_state = str(row.get("position_state", "空仓"))
    execution_priority = str(row.get("execution_priority", "观察"))
    normalized_regime = str(market_regime or "").strip()
    market_open = normalized_regime in {"", "风险开"}
    market_neutral_or_better = normalized_regime in {"", "风险开", "中性"}

    if atr_stop_price is not None and position_state in {"持仓", "持仓待加仓", "待卖出"} and live_price <= atr_stop_price:
        return {
            "action": "盘中撤退",
            "message": f"当前价已触及 ATR 风控价 {atr_stop_price:.2f}，持仓应优先防守。",
        }

    if sell_signal and live_delta <= cfg["scan_sell_confirm_delta_pct"]:
        return {
            "action": "盘中撤退",
            "message": "日线卖点已出现，且盘中继续走弱，不适合逆势硬扛。",
        }

    if buy_signal and execution_priority == "买入" and live_delta >= cfg["scan_trial_min_live_delta_pct"] and score >= cfg["trial_score_min"]:
        if market_open:
            return {
                "action": "盘中试买",
                "message": "扫描摘要显示买点成立，且盘中价格继续走强，可按试仓规则先做小仓位。",
            }
        return {
            "action": "盘中观察",
            "message": f"个股盘中偏强，但市场环境为 {format_market_regime_label(normalized_regime)}，先观察不追。",
        }

    if add_on_signal and market_neutral_or_better and live_delta >= -cfg["scan_trial_min_live_delta_pct"]:
        return {
            "action": "盘中观察",
            "message": "扫描摘要存在补仓条件，若盘中继续稳住，可转到单股页再确认。",
        }

    if abs(live_delta) >= cfg["scan_observe_abs_delta_pct"]:
        return {
            "action": "盘中观察",
            "message": f"当前价较收盘偏离 {live_delta:+.2%}，波动较大，优先等单股页确认。",
        }

    return None


def lookup_snapshot_row(snapshot: pd.DataFrame, symbol: str) -> dict[str, object] | None:
    if snapshot.empty:
        return None

    normalized_symbol = normalize_market_symbol(symbol)
    candidates = {normalized_symbol}
    if normalized_symbol.startswith(("sh", "sz", "bj")):
        candidates.add(normalized_symbol[2:])

    matched = snapshot.loc[snapshot["symbol"].astype(str).str.lower().isin(candidates)]
    if matched.empty:
        return None

    row = matched.iloc[0]
    return {
        "symbol": str(row["symbol"]),
        "name": str(row.get("name", "") or ""),
        "last_price": float(row["last_price"]),
        "turnover_amount": float(row["turnover_amount"]),
    }


@st.cache_data(ttl=180, show_spinner=False)
def cached_watchlist_dataframe(
    watchlist_key: str,
    config_path: str,
    config_fingerprint: str,
    refresh_token: int,
    market_regime: str,
    intraday_preset: str,
) -> pd.DataFrame:
    del config_fingerprint, refresh_token
    symbols = parse_watchlist_symbols(watchlist_key)
    if not symbols:
        return pd.DataFrame()

    config = load_config(config_path)
    intraday_prompt_config = resolve_intraday_prompt_config(config, intraday_preset)
    client = MarketDataClient(DATA_DIR)
    end_datetime = datetime.combine(date.today(), datetime.min.time())
    start_datetime = end_datetime - pd.Timedelta(days=max(int(config["scan"]["history_days"]) * 2, 400))

    snapshot_path = DATA_DIR / "latest_snapshot.csv"
    if snapshot_path.exists():
        snapshot_df = pd.read_csv(snapshot_path)
    else:
        try:
            snapshot_df = client.get_universe_snapshot()
        except RuntimeError:
            snapshot_df = pd.DataFrame()


    rows: list[dict[str, object]] = []
    for symbol in symbols:
        try:
            history = client.get_history(symbol, start_datetime, end_datetime)

        except RuntimeError as error:
            rows.append(
                {
                    "symbol": symbol,
                    "name": "-",
                    "date": None,
                    "close": None,
                    "live_price": None,
                    "live_delta": None,
                    "action": "数据失败",
                    "signal_age": None,
                    "holding_days": None,
                    "position_state": "数据失败",
                    "atr_stop_price": None,
                    "entry_signal_type": "",
                    "sell_reason": "",
                    "score": None,
                    "volume_ratio": None,
                    "rsi": None,
                    "hint": str(error),
                }
            )
            continue

        if history.empty:
            rows.append(
                {
                    "symbol": symbol,
                    "name": "-",
                    "date": None,
                    "close": None,
                    "live_price": None,
                    "live_delta": None,
                    "action": "无数据",
                    "signal_age": None,
                    "holding_days": None,
                    "position_state": "无数据",
                    "atr_stop_price": None,
                    "entry_signal_type": "",
                    "sell_reason": "",
                    "score": None,
                    "volume_ratio": None,
                    "rsi": None,
                    "hint": "没有获取到历史数据。",
                }
            )
            continue

        signal_history = add_signal_columns(history, config)
        summary = latest_signal_summary(signal_history, config)
        snapshot = lookup_snapshot_row(snapshot_df, symbol)
        intraday_context = build_intraday_context(signal_history, snapshot)
        intraday_prompt = build_intraday_execution_prompt(
            signal_history,
            summary,
            snapshot,
            market_regime,
            intraday_prompt_config,
        )
        latest_row = signal_history.iloc[-1]
        action = summary["action"] if summary else "观察"
        hint = summary["execution_advice"] if summary else "当前没有新的执行信号，继续观察。"
        if intraday_context is not None and intraday_context["is_material"]:
            hint = f"{hint} {intraday_context['message']}"
        if intraday_prompt is not None and intraday_prompt["action"] != "-":
            hint = f"{hint} 盘中提示：{intraday_prompt['message']}"

        rows.append(
            {
                "symbol": normalize_market_symbol(symbol),
                "name": snapshot["name"] if snapshot else "-",
                "date": latest_row["date"].date(),
                "close": round(float(latest_row["close"]), 2),
                "live_price": round(float(snapshot["last_price"]), 2) if snapshot else None,
                "live_delta": ((float(snapshot["last_price"]) / float(latest_row["close"])) - 1) if snapshot else None,
                "action": action,
                "signal_age": summary["signal_age"] if summary else None,
                "holding_days": summary["holding_days"] if summary else None,
                "position_state": summary["position_state"] if summary else "空仓",
                "atr_stop_price": summary["atr_stop_price"] if summary else None,
                "entry_signal_type": summary["entry_signal_type"] if summary else "",
                "sell_reason": summary["sell_reason"] if summary else "",
                "score": summary["score"] if summary else None,
                "volume_ratio": summary["volume_ratio"] if summary else None,
                "rsi": summary["rsi"] if summary else None,
                "intraday_action": intraday_prompt["action"] if intraday_prompt else "-",
                "intraday_hint": intraday_prompt["message"] if intraday_prompt else "暂无盘中提示。",
                "hint": hint,
            }
        )

    watchlist_df = pd.DataFrame(rows)
    if watchlist_df.empty:
        return watchlist_df

    action_order = {"卖出": 0, "买入": 1, "补仓": 2, "持仓": 3, "观察": 4, "无数据": 5, "数据失败": 6}
    watchlist_df["action_rank"] = watchlist_df["action"].map(lambda value: action_order.get(str(value), 99))
    return watchlist_df.sort_values(["action_rank", "score"], ascending=[True, False], na_position="last").reset_index(drop=True)


def style_watchlist_dataframe(dataframe: pd.DataFrame):
    if "当前动作" not in dataframe.columns:
        return dataframe

    action_colors = {
        "卖出": "#fecaca",
        "买入": "#dcfce7",
        "补仓": "#fef3c7",
        "持仓": "#dbeafe",
        "观察": "#e5e7eb",
        "无数据": "#e5e7eb",
        "数据失败": "#fee2e2",
    }

    def highlight_action(row: pd.Series) -> list[str]:
        styles = [""] * len(row)
        color = action_colors.get(str(row.get("当前动作", "")), "")
        if color:
            styles = [f"background-color: {color}"] * len(row)
        intraday_action = str(row.get("盘中动作", ""))
        if intraday_action and intraday_action != "-" and "盘中动作" in dataframe.columns:
            intraday_index = list(dataframe.columns).index("盘中动作")
            intraday_colors = {
                "盘中试买": "background-color: #dcfce7; font-weight: 600",
                "盘中撤退": "background-color: #fecaca; font-weight: 600",
                "盘中观察": "background-color: #fef3c7; font-weight: 600",
            }
            if intraday_action in intraday_colors:
                styles[intraday_index] = intraday_colors[intraday_action]
        return styles

    return dataframe.style.apply(highlight_action, axis=1)


def build_signal_markers(history: pd.DataFrame, signal_column: str, anchor: str) -> pd.DataFrame:
    signal_mask = history[signal_column].fillna(False).astype(bool)
    event_points = history.loc[signal_mask & ~signal_mask.shift(1, fill_value=False)].copy()
    if event_points.empty:
        return event_points

    price_span = (history["high"].max() - history["low"].min()) or history["close"].max() * 0.03 or 1.0
    offset = price_span * 0.025
    if anchor == "below":
        event_points["marker_y"] = event_points["low"] - offset
    else:
        event_points["marker_y"] = event_points["high"] + offset
    return event_points


def format_score_value(value: object) -> str:
    if pd.isna(value):
        return "-"
    return str(int(round(float(value))))


def format_percent_value(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.2f}%"


def format_two_decimals(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.2f}"


def _canonical_table_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (int, float)):
        return round(float(value), 4)
    return str(value)


def drop_uniform_display_columns(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

    hidden_columns: list[str] = []
    for column in columns:
        if column not in dataframe.columns:
            continue
        normalized_values = {_canonical_table_value(value) for value in dataframe[column]}
        if len(normalized_values) <= 1:
            hidden_columns.append(column)

    if not hidden_columns:
        return dataframe
    return dataframe.drop(columns=hidden_columns)


def build_price_figure(history: pd.DataFrame) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.7, 0.3],
    )
    figure.add_trace(
        go.Candlestick(
            x=history["date"],
            open=history["open"],
            high=history["high"],
            low=history["low"],
            close=history["close"],
            name="K线",
        ),
        row=1,
        col=1,
    )
    for column, name, color in [
        ("ma_fast", "MA20", "#0f766e"),
        ("ma_mid", "MA50", "#ea580c"),
        ("ma_slow", "MA100", "#1d4ed8"),
    ]:
        figure.add_trace(
            go.Scatter(
                x=history["date"],
                y=history[column],
                mode="lines",
                name=name,
                line={"width": 2, "color": color},
            ),
            row=1,
            col=1,
        )

    buy_points = build_signal_markers(history, "buy_signal", "below")
    add_on_points = build_signal_markers(history, "add_on_signal", "below")
    sell_points = build_signal_markers(history, "sell_signal", "above")
    if not buy_points.empty:
        figure.add_trace(
            go.Scatter(
                x=buy_points["date"],
                y=buy_points["marker_y"],
                mode="markers",
                name="买点",
                customdata=buy_points[["close", "volume_ratio", "rsi"]],
                hovertemplate="买点<br>%{x|%Y-%m-%d}<br>收盘: %{customdata[0]:.2f}<br>量比: %{customdata[1]:.2f}<br>RSI: %{customdata[2]:.2f}<extra></extra>",
                marker={
                    "symbol": "triangle-up",
                    "size": 13,
                    "color": "#16a34a",
                    "line": {"width": 1.8, "color": "#ffffff"},
                },
            ),
            row=1,
            col=1,
        )
    if not add_on_points.empty:
        figure.add_trace(
            go.Scatter(
                x=add_on_points["date"],
                y=add_on_points["marker_y"],
                mode="markers",
                name="补仓点",
                customdata=add_on_points[["close", "volume_ratio", "rsi"]],
                hovertemplate="补仓点<br>%{x|%Y-%m-%d}<br>收盘: %{customdata[0]:.2f}<br>量比: %{customdata[1]:.2f}<br>RSI: %{customdata[2]:.2f}<extra></extra>",
                marker={
                    "symbol": "diamond",
                    "size": 11,
                    "color": "#f59e0b",
                    "line": {"width": 1.6, "color": "#ffffff"},
                },
            ),
            row=1,
            col=1,
        )
    if not sell_points.empty:
        figure.add_trace(
            go.Scatter(
                x=sell_points["date"],
                y=sell_points["marker_y"],
                mode="markers",
                name="卖点",
                customdata=sell_points[["close", "sell_reason"]],
                hovertemplate="卖点<br>%{x|%Y-%m-%d}<br>收盘: %{customdata[0]:.2f}<br>原因: %{customdata[1]}<extra></extra>",
                marker={
                    "symbol": "triangle-down",
                    "size": 13,
                    "color": "#ef4444",
                    "line": {"width": 1.8, "color": "#ffffff"},
                },
            ),
            row=1,
            col=1,
        )

    figure.add_trace(
        go.Bar(
            x=history["date"],
            y=history["volume"],
            name="成交量",
            marker_color="#94a3b8",
        ),
        row=2,
        col=1,
    )
    figure.update_layout(
        height=720,
        margin={"l": 12, "r": 12, "t": 24, "b": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    figure.update_yaxes(row=1, col=1, automargin=True)
    return figure


def render_scan_section(config_path: str, intraday_preset: str = "跟随配置") -> pd.DataFrame:
    st.subheader("市场扫描")
    purge_expired_non_partial_scan_cache()
    config = load_config(config_path)
    intraday_prompt_config = resolve_intraday_prompt_config(config, intraday_preset)
    scan_col1, scan_col2, scan_col3 = st.columns([1, 1, 1.4])
    top_n = scan_col1.slider("候选股数量", min_value=5, max_value=50, value=15, step=5)
    use_today = scan_col2.checkbox("使用今天作为扫描日期", value=True)
    scan_date = None if use_today else scan_col3.date_input("扫描截止日", value=date.today())
    refresh = st.button("刷新扫描", type="primary", width="stretch")
    as_of_text = None if scan_date is None else scan_date.isoformat()
    requested_date = resolve_scan_target_date(as_of_text)
    config_fingerprint = compute_config_fingerprint(config_path)
    refresh_token = int(st.session_state.get("scan_refresh_token", 0))

    if "scan_df" not in st.session_state:
        scan_df, status_message, is_ready, scan_source = load_latest_scan_from_disk(config_fingerprint, requested_date)
        if scan_source == "invalidated":
            status_message = "检测到策略参数变化，本地扫描缓存已失效。点击“刷新扫描”后再按当前参数重跑。"
        elif not is_ready:
            stale_message = "本地缓存日期落后，当前先展示最近缓存。点击“刷新扫描”后再尝试拉取最新结果。"
            status_message = stale_message if not status_message else f"{status_message} {stale_message}"
        st.session_state.scan_df = scan_df
        st.session_state.scan_status_message = status_message
        st.session_state.scan_is_ready = is_ready
        st.session_state.scan_source = scan_source
        st.session_state.scan_manual_refresh = False
        st.session_state.scan_top_n = top_n
        st.session_state.scan_config_fingerprint = config_fingerprint
    elif st.session_state.get("scan_config_fingerprint") != config_fingerprint:
        fallback_df, fallback_status, fallback_ready, fallback_source = load_latest_scan_from_disk(config_fingerprint, requested_date)
        st.session_state.scan_df = fallback_df
        st.session_state.scan_status_message = fallback_status or "检测到策略参数变化。点击“刷新扫描”后按当前参数重跑。"
        st.session_state.scan_is_ready = fallback_ready
        st.session_state.scan_source = fallback_source
        st.session_state.scan_manual_refresh = False
        st.session_state.scan_top_n = top_n
        st.session_state.scan_config_fingerprint = config_fingerprint
    elif refresh:
        refresh_token += 1
        st.session_state.scan_refresh_token = refresh_token
        with st.spinner("正在获取全市场数据并计算信号..."):
            scan_df, status_message, is_ready, scan_source = cached_scan(
                config_path,
                config_fingerprint,
                as_of_text,
                top_n,
                refresh_token,
            )
            st.session_state.scan_df = scan_df
            st.session_state.scan_status_message = status_message
            st.session_state.scan_is_ready = is_ready
            st.session_state.scan_source = scan_source
        st.session_state.scan_manual_refresh = True
        st.session_state.scan_top_n = top_n
        st.session_state.scan_config_fingerprint = config_fingerprint
    elif st.session_state.get("scan_top_n") != top_n:
        st.session_state.scan_status_message = "候选数量参数已更新。点击“刷新扫描”后按新数量重跑。"
        st.session_state.scan_manual_refresh = False
        st.session_state.scan_top_n = top_n
        st.session_state.scan_config_fingerprint = config_fingerprint
    current_scan_df = st.session_state.get("scan_df", pd.DataFrame())
    if (
        st.session_state.get("scan_source") == "partial"
        and len(current_scan_df) < top_n
        and not st.session_state.get("scan_manual_refresh", False)
    ):
        fallback_df, fallback_status, fallback_ready, fallback_source = load_latest_non_partial_scan_from_disk(
            config_fingerprint,
            requested_date,
        )
        if not fallback_df.empty and len(fallback_df) > len(current_scan_df):
            st.session_state.scan_df = fallback_df
            partial_rows = len(current_scan_df)
            fallback_rows = len(fallback_df)
            fallback_note = (
                f"最新缓存仅有 {partial_rows} 只股票，已临时回退到本地更完整的 {fallback_rows} 只缓存结果。"
            )
            st.session_state.scan_status_message = (
                fallback_note if not fallback_status else f"{fallback_note} {fallback_status}"
            )
            st.session_state.scan_is_ready = fallback_ready
            st.session_state.scan_source = fallback_source
            st.session_state.scan_top_n = top_n
            st.session_state.scan_config_fingerprint = config_fingerprint

    if (
        st.session_state.get("scan_manual_refresh", False)
        and st.session_state.get("scan_source") == "partial"
        and len(st.session_state.get("scan_df", pd.DataFrame())) < top_n
    ):
        manual_refresh_note = (
            f"本次手动刷新只拿到 {len(st.session_state.get('scan_df', pd.DataFrame()))} 只最新股票，"
            "因此当前优先展示最新日期结果，而不再自动回退到旧缓存。"
        )
        existing_message = st.session_state.get("scan_status_message")
        if existing_message and manual_refresh_note not in existing_message:
            st.session_state.scan_status_message = f"{existing_message} {manual_refresh_note}"
        elif not existing_message:
            st.session_state.scan_status_message = manual_refresh_note

    scan_df = st.session_state.get("scan_df", pd.DataFrame())
    status_message = st.session_state.get("scan_status_message")
    is_ready = st.session_state.get("scan_is_ready", True)
    scan_source = st.session_state.get("scan_source", "none")
    market_filter_sources = (
        scan_df["market_filter_source"].dropna().astype(str).unique().tolist()
        if not scan_df.empty and "market_filter_source" in scan_df.columns
        else []
    )
    market_reason_sample = ""
    if not scan_df.empty and "market_reason" in scan_df.columns:
        market_reason_values = scan_df["market_reason"].dropna().astype(str)
        if not market_reason_values.empty:
            market_reason_sample = market_reason_values.iloc[0]

    if status_message:
        if scan_source == "stale":
            st.warning(status_message)
            st.info("当前展示的是本地参考结果，建议在数据源稳定后再刷新正式结果。")
        elif scan_source == "partial":
            st.warning(status_message)
        else:
            st.info(status_message)

    if scan_df.empty:
        st.warning("当前没有满足条件的标的，可以调低参数或稍后重试。")
        return scan_df

    latest_signal_date = pd.to_datetime(scan_df["date"]).max().date()
    if latest_signal_date < requested_date:
        st.warning(
            f"当前扫描请求日期是 {requested_date:%Y-%m-%d}，但最新可用日线信号只到 {latest_signal_date:%Y-%m-%d}。"
            " 这说明数据源还没有给出更近一个交易日的完整日线，不能把它当作今天收盘后的正式选股结果。"
        )
    else:
        st.caption(f"当前扫描使用的最新信号日期：{latest_signal_date:%Y-%m-%d}")

    confidence_label = "高"
    confidence_note = "当前结果来自完整扫描，可作为正式盘后复盘和次日计划的主要参考。"
    if scan_source == "partial":
        confidence_label = "低"
        confidence_note = (
            f"当前只拿到 {len(scan_df)} 只股票的部分结果，优先用于观察主线和情绪，"
            "不建议把缺样本的排名直接当成正式交易清单。"
        )
    elif scan_source == "stale" or latest_signal_date < requested_date:
        confidence_label = "低"
        confidence_note = "当前结果落后于请求日期，只适合参考，不适合作为新的正式开仓依据。"
    elif market_filter_sources == ["宽度过滤"]:
        confidence_label = "中"
        confidence_note = "指数过滤未成功，本次市场环境仅靠宽度代理，开仓结论需要比平时更保守。"

    if scan_source == "partial":
        st.error(
            f"结果可信度：{confidence_label}。{confidence_note}"
        )
    elif confidence_label == "中":
        st.warning(f"结果可信度：{confidence_label}。{confidence_note}")
    else:
        st.info(f"结果可信度：{confidence_label}。{confidence_note}")

    if "宽度过滤" in market_filter_sources:
        breadth_fallback_message = (
            "本次市场环境判断已回退到宽度过滤。指数过滤没有成功返回，"
            "因此“风险开 / 中性 / 风险关”更适合用来辅助节奏判断，"
            "不宜单独当成机械开仓许可。"
        )
        if market_reason_sample:
            breadth_fallback_message += f" 当前原因：{market_reason_sample}"
        st.warning(breadth_fallback_message)

    st.markdown(MARKET_SIGNAL_CSS, unsafe_allow_html=True)
    scan_status_cards = [
        {
            "title": "结果状态",
            "status": {"complete": "完整", "partial": "部分", "stale": "参考"}.get(scan_source, "待确认"),
            "tone": "green" if scan_source == "complete" else ("red" if scan_source == "partial" else "yellow"),
            "hint": status_message or "当前结果可直接作为盘后复盘参考。",
        },
        {
            "title": "执行可信度",
            "status": confidence_label,
            "tone": "green" if confidence_label == "高" else ("yellow" if confidence_label == "中" else "red"),
            "hint": confidence_note,
        },
        {
            "title": "市场过滤",
            "status": " / ".join(market_filter_sources) if market_filter_sources else "未知",
            "tone": "yellow" if "宽度过滤" in market_filter_sources else "green",
            "hint": market_reason_sample or "当前市场环境过滤正常。",
        },
        {
            "title": "信号日期",
            "status": latest_signal_date.strftime("%Y-%m-%d"),
            "tone": "green" if latest_signal_date >= requested_date else "red",
            "hint": (
                "当前信号日期与请求日期一致。"
                if latest_signal_date >= requested_date
                else "信号日期落后于请求日期，不能把它当作最新正式结果。"
            ),
        },
    ]
    scan_cards_markup = build_signal_cards_markup(scan_status_cards)
    if scan_cards_markup:
        st.markdown(f"<div class='market-signal-grid'>{scan_cards_markup}</div>", unsafe_allow_html=True)

    source_file = SCAN_SOURCE_FILES.get(scan_source, "未知来源")
    source_col1, source_col2, source_col3, source_col4 = st.columns([1.1, 0.8, 1.0, 1.3])
    source_col1.caption(f"当前加载结果：{source_file}")
    source_col2.caption(f"当前展示股票数：{len(scan_df)}")
    source_col3.caption(f"结果可信度：{confidence_label}")
    source_col4.caption(
        "盘中提示档位："
        f"{intraday_preset}"
        f" | 突破确认 +{intraday_prompt_config['breakout_confirm_pct']:.2%}"
        f" | 试买评分 >= {int(intraday_prompt_config['trial_score_min'])}"
    )
    scan_preset_warning = get_intraday_preset_warning(intraday_preset)
    if scan_preset_warning:
        st.warning(scan_preset_warning)

    buy_count = int(scan_df["buy_signal"].sum())
    add_on_count = int(scan_df["add_on_signal"].sum())
    avg_score = float(scan_df["score"].mean())
    best_score = float(scan_df["score"].max())
    summary_metrics = [
        ("买点数量", buy_count),
        ("补仓数量", add_on_count),
        ("平均评分", format_score_value(avg_score)),
        ("最高评分", format_score_value(best_score)),
    ]
    if "execution_priority" in scan_df.columns:
        priority_counts = scan_df["execution_priority"].value_counts()
        summary_metrics.extend((label, int(priority_counts.get(label, 0))) for label in PRIORITY_LABELS)

    metric_columns = st.columns(len(summary_metrics))
    for metric_column, (label, value) in zip(metric_columns, summary_metrics):
        metric_column.metric(label, value)

    display_df = scan_df.copy()
    market_regime = resolve_dashboard_market_regime(config_path, config_fingerprint, scan_df)
    snapshot_df = cached_snapshot_dataframe()
    live_prices: list[float | None] = []
    live_deltas: list[float | None] = []
    intraday_actions: list[str] = []
    intraday_hints: list[str] = []
    for row in display_df.itertuples(index=False):
        snapshot = lookup_snapshot_row(snapshot_df, str(row.symbol))
        live_price = float(snapshot["last_price"]) if snapshot else None
        row_close = float(row.close) if pd.notna(row.close) else None
        live_delta = ((live_price / row_close) - 1.0) if snapshot and row_close else None
        prompt = build_scan_intraday_prompt(pd.Series(row._asdict()), snapshot, market_regime, intraday_prompt_config)
        live_prices.append(live_price)
        live_deltas.append(live_delta)
        intraday_actions.append(prompt["action"] if prompt else "-")
        intraday_hints.append(prompt["message"] if prompt else "暂无盘中提示。")

    display_df["live_price"] = live_prices
    display_df["live_delta"] = live_deltas
    display_df["intraday_action"] = intraday_actions
    display_df["intraday_hint"] = intraday_hints
    if "symbol" in display_df.columns:
        display_df["symbol"] = display_df["symbol"].map(format_symbol)
    execution_confidence_values: list[str] = []
    overheat_risk_values: list[str] = []
    for row in display_df.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        confidence, _, _ = classify_scan_execution_confidence(row_series, scan_source, latest_signal_date, requested_date)
        overheat_risk, _, _ = classify_scan_overheat_risk(row_series)
        execution_confidence_values.append(confidence)
        overheat_risk_values.append(overheat_risk)
    display_df["execution_confidence"] = execution_confidence_values
    display_df["overheat_risk"] = overheat_risk_values
    intraday_order = {"盘中试买": 0, "盘中撤退": 1, "盘中观察": 2, "-": 3}
    confidence_order = {"高": 0, "中": 1, "低": 2}
    overheat_order = {"低": 0, "中": 1, "高": 2}
    display_df["intraday_action_rank"] = display_df["intraday_action"].map(lambda value: intraday_order.get(str(value), 9))
    display_df["execution_confidence_rank"] = display_df["execution_confidence"].map(lambda value: confidence_order.get(str(value), 9))
    display_df["overheat_risk_rank"] = display_df["overheat_risk"].map(lambda value: overheat_order.get(str(value), 9))
    sort_columns = [column for column in ["intraday_action_rank", "execution_confidence_rank", "overheat_risk_rank", "execution_priority_rank", "score", "turnover_amount"] if column in display_df.columns]
    ascending_flags = [column not in {"score", "turnover_amount"} for column in sort_columns]
    if sort_columns:
        display_df = display_df.sort_values(sort_columns, ascending=ascending_flags, na_position="last").reset_index(drop=True)
    priority_columns = [
        "execution_priority",
        "intraday_action",
        "position_state",
        "entry_signal_type",
        "execution_confidence",
        "overheat_risk",
        "atr_stop_price",
        "symbol",
        "name",
        "date",
        "close",
        "live_price",
        "live_delta",
        "buy_signal",
        "add_on_signal",
        "sell_signal",
        "sell_reason",
        "score",
        "volume_ratio",
        "rsi",
        "turnover_amount",
        "intraday_hint",
    ]
    display_df = display_df[[column for column in priority_columns if column in display_df.columns]]
    display_df = drop_uniform_display_columns(
        display_df,
        ["position_state", "entry_signal_type", "atr_stop_price", "sell_reason"],
    )
    display_df["turnover_amount"] = display_df["turnover_amount"].map(lambda value: f"{value / 1e8:.2f} 亿")
    display_df["close"] = display_df["close"].map(lambda value: f"{float(value):.2f}" if pd.notna(value) else "-")
    if "live_price" in display_df.columns:
        display_df["live_price"] = display_df["live_price"].map(lambda value: f"{float(value):.2f}" if pd.notna(value) else "-")
    if "live_delta" in display_df.columns:
        display_df["live_delta"] = display_df["live_delta"].map(lambda value: f"{value:+.2%}" if pd.notna(value) else "-")
    display_df["buy_signal"] = display_df["buy_signal"].map(lambda value: "是" if bool(value) else "否")
    display_df["add_on_signal"] = display_df["add_on_signal"].map(lambda value: "是" if bool(value) else "否")
    display_df["sell_signal"] = display_df["sell_signal"].map(lambda value: "是" if bool(value) else "否")
    if "atr_stop_price" in display_df.columns:
        display_df["atr_stop_price"] = display_df["atr_stop_price"].map(lambda value: f"{float(value):.2f}" if pd.notna(value) else "-")
    if "score" in display_df.columns:
        display_df["score"] = display_df["score"].map(format_score_value)
    for numeric_column in ["volume_ratio", "rsi"]:
        if numeric_column in display_df.columns:
            display_df[numeric_column] = display_df[numeric_column].map(format_two_decimals)
    drop_rank_columns = [column for column in ["execution_priority_rank", "execution_confidence_rank", "overheat_risk_rank"] if column in display_df.columns]
    if drop_rank_columns:
        display_df = display_df.drop(columns=drop_rank_columns)
    display_df = display_df.rename(
        columns={
            "symbol": "代码",
            "name": "名称",
            "live_price": "当前价格",
            "live_delta": "实时偏离",
            "intraday_action": "盘中动作",
            "position_state": "持仓状态",
            "entry_signal_type": "买点类型",
            "execution_confidence": "执行置信度",
            "overheat_risk": "过热风险",
            "atr_stop_price": "ATR风控价",
            "turnover_amount": "成交额",
            "date": "日期",
            "close": "收盘价",
            "buy_signal": "买点信号",
            "add_on_signal": "补仓信号",
            "sell_signal": "卖点信号",
            "sell_reason": "卖点原因",
            "execution_priority": "执行优先级",
            "score": "评分",
            "volume_ratio": "量比",
            "rsi": "RSI",
            "intraday_hint": "盘中提示",
        }
    )
    st.caption("执行置信度看结果是否适合机械执行；过热风险看当前是否已经进入追高区，二者应结合而不是只看评分。")
    st.dataframe(style_scan_dataframe(display_df), width="stretch", hide_index=True)
    return scan_df


def render_market_overview_section(config_path: str, config_fingerprint: str, scan_df: pd.DataFrame) -> None:
    st.markdown(MARKET_SIGNAL_CSS, unsafe_allow_html=True)
    st.subheader("市场总览")

    latest_signal_date = get_latest_signal_date(scan_df)
    latest_signal_date_text = latest_signal_date.isoformat() if latest_signal_date else None
    market_overview = cached_market_overview(config_path, config_fingerprint, latest_signal_date_text)
    market_index_history = cached_market_index_history(config_path, config_fingerprint, latest_signal_date_text)
    outlook = build_market_outlook(scan_df, market_overview)
    snapshot_df = cached_snapshot_dataframe()
    snapshot_summary = build_market_snapshot_summary(snapshot_df)
    market_summary = build_market_summary_analysis(outlook, market_overview, snapshot_summary)
    snapshot_path = DATA_DIR / "latest_snapshot.csv"
    snapshot_timestamp = pd.Timestamp(snapshot_path.stat().st_mtime, unit="s") if snapshot_path.exists() else None
    snapshot_date = snapshot_timestamp.date() if snapshot_timestamp is not None else None
    index_data_date = market_overview.get("index_data_date")
    index_source_label = format_data_source_label(
        market_overview.get("index_source"),
        market_overview.get("index_source_note"),
    )
    snapshot_source_label = format_data_source_label(
        snapshot_df.attrs.get("snapshot_source", ""),
        snapshot_df.attrs.get("snapshot_source_note", ""),
    )
    timeline_col1, timeline_col2, timeline_col3, timeline_col4 = st.columns(4)
    timeline_col1.metric("扫描信号日期", latest_signal_date.isoformat() if latest_signal_date else "-")
    timeline_col2.metric("指数数据日期", index_data_date.isoformat() if isinstance(index_data_date, date) else "-")
    timeline_col3.metric("快照文件时间", snapshot_timestamp.strftime("%Y-%m-%d %H:%M") if snapshot_timestamp is not None else "-")

    aligned_dates = {value for value in [latest_signal_date, index_data_date, snapshot_date] if isinstance(value, date)}
    if not aligned_dates:
        data_alignment_label = "待确认"
        data_alignment_help = "当前缺少足够的日期锚点，市场判断应以保守解读为主。"
    elif len(aligned_dates) == 1:
        data_alignment_label = "一致"
        data_alignment_help = "扫描、指数和快照口径基本对齐，框架判断与盘面温度可一起参考。"
    else:
        data_alignment_label = "混合"
        data_alignment_help = "当前页面混用了不同交易时点的数据：扫描/指数更偏收盘框架，快照更偏盘中温度，执行上不要机械叠加。"
    timeline_col4.metric("数据口径", data_alignment_label)
    st.caption(data_alignment_help)
    st.caption(f"数据来源：指数 {index_source_label}；盘面快照 {snapshot_source_label}。")

    display_outlook = outlook.copy()
    display_market_summary = market_summary.copy()
    if data_alignment_label != "一致":
        confidence_downgrade_map = {"高": "中高", "中高": "中", "中": "低", "低": "低"}
        label_downgrade_map = {"看多": "偏多", "偏多": "中性", "看空": "偏空", "偏空": "中性"}
        continuation_downgrade_map = {"继续上攻": "偏强震荡上行", "偏强震荡上行": "震荡待确认", "回调压力偏大": "震荡待确认"}
        tone_downgrade_map = {"普涨偏强": "震荡偏强", "普跌承压": "分化震荡"}
        pullback_downgrade_map = {"低": "中", "中": "中", "高": "高"}

        display_outlook["confidence"] = confidence_downgrade_map.get(str(outlook.get("confidence", "中")), "中")
        display_outlook["label"] = label_downgrade_map.get(str(outlook.get("label", "中性")), str(outlook.get("label", "中性")))
        if isinstance(display_outlook.get("score"), (int, float)):
            display_outlook["score"] = round(float(display_outlook["score"]) * 0.6, 2)
        display_outlook["summary"] = f"当前为{data_alignment_label}口径，以下框架判断按谨慎参考处理。{outlook['summary']}"

        display_market_summary["continuation_label"] = continuation_downgrade_map.get(
            str(market_summary.get("continuation_label", "震荡待确认")),
            str(market_summary.get("continuation_label", "震荡待确认")),
        )
        display_market_summary["market_tone"] = tone_downgrade_map.get(
            str(market_summary.get("market_tone", "分化震荡")),
            str(market_summary.get("market_tone", "分化震荡")),
        )
        display_market_summary["pullback_risk"] = pullback_downgrade_map.get(
            str(market_summary.get("pullback_risk", "中")),
            str(market_summary.get("pullback_risk", "中")),
        )
        display_market_summary["summary"] = f"当前为{data_alignment_label}口径，以下盘面温度判断按谨慎参考处理。{market_summary['summary']}"
        display_market_summary["risk_hint"] = f"当前为{data_alignment_label}口径，执行上优先看风险控制和仓位节奏。{market_summary['risk_hint']}"

    if latest_signal_date is not None and snapshot_date is not None and latest_signal_date != snapshot_date:
        st.warning(
            f"当前市场页混合使用了不同日期的数据：扫描信号停留在 {latest_signal_date:%Y-%m-%d}，"
            f"但全市场快照文件更新时间是 {snapshot_timestamp:%Y-%m-%d %H:%M:%S}。"
            " 前者更适合看 1-2 周框架，后者更适合看当下情绪，不建议把两者直接当成同一时点的结论。"
        )
    elif latest_signal_date is None and snapshot_timestamp is not None:
        st.warning("当前市场页缺少扫描信号日期，只能更多依赖盘面快照，1-2 周框架判断的稳定性会明显下降。")

    previous_snapshot_row = update_market_snapshot_summary_history(snapshot_summary, market_summary, snapshot_timestamp)
    market_signals = apply_market_signal_deltas(market_summary.get("signals", []), snapshot_summary, market_summary, previous_snapshot_row)

    overview_tab, market_tab = st.tabs(["框架判断", "盘中温度"])

    with overview_tab:
        st.caption("这一页更偏收盘后的框架判断，核心依据是扫描结果和基准指数趋势。")
        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("未来1-2周判断", display_outlook["label"])
        metric2.metric("市场环境", str(display_outlook.get("regime_display") or format_market_regime_label(display_outlook["regime"])))
        metric3.metric(
            "趋势广度",
            f"{display_outlook['breadth']:.0%}" if isinstance(display_outlook.get("breadth"), (int, float)) else "-",
        )
        metric4.metric("判断把握", display_outlook["confidence"])

        signal_col1, signal_col2, signal_col3, signal_col4 = st.columns(4)
        signal_col1.metric("买点数", display_outlook["buy_count"])
        signal_col2.metric("补仓数", display_outlook["add_on_count"])
        signal_col3.metric("卖点数", display_outlook["sell_count"])
        signal_col4.metric("判断分数", format_two_decimals(display_outlook["score"]))

        outlook_label = display_outlook["label"]
        if outlook_label in {"看多", "偏多"}:
            st.success(display_outlook["summary"])
        elif outlook_label in {"看空", "偏空"}:
            st.error(display_outlook["summary"])
        else:
            st.info(display_outlook["summary"])

        reason_text = None
        if not scan_df.empty and "market_reason" in scan_df.columns:
            market_reason = str(scan_df["market_reason"].iloc[0]).strip()
            if market_reason:
                reason_text = format_market_regime_text(market_reason)
        if reason_text:
            st.caption(f"环境判定依据：{reason_text}")

        benchmark_symbol = str(market_overview.get("benchmark_symbol", "000300"))
        if market_overview.get("index_available"):
            benchmark_df = pd.DataFrame(
                [
                    {
                        "基准指数": benchmark_symbol,
                        "指数日期": market_overview["index_data_date"],
                        "指数趋势": market_overview["index_trend"],
                        "收盘": market_overview["index_close"],
                        "MA20": market_overview["index_ma_short"],
                        "MA60": market_overview["index_ma_long"],
                        "5日涨跌": market_overview["index_return_5d"],
                        "10日涨跌": market_overview["index_return_10d"],
                        "20日涨跌": market_overview["index_return_20d"],
                    }
                ]
            )
            for column in ["收盘", "MA20", "MA60"]:
                benchmark_df[column] = benchmark_df[column].map(format_two_decimals)
            for column in ["5日涨跌", "10日涨跌", "20日涨跌"]:
                benchmark_df[column] = benchmark_df[column].map(lambda value: format_percent_value(value * 100) if pd.notna(value) else "-")
            st.dataframe(benchmark_df, width="stretch", hide_index=True)
            st.caption(f"{market_overview.get('index_reason', '')} 数据来源：{index_source_label}。")
        else:
            st.warning(str(market_overview.get("index_reason", f"指数 {benchmark_symbol} 暂无可用数据。")))

    with market_tab:
        st.caption("这一页更偏盘中温度和情绪快照，适合看今天的强弱分布，不宜单独替代趋势框架。")
        summary_metric1, summary_metric2, summary_metric3, summary_metric4, summary_metric5 = st.columns(5)
        summary_metric1.metric("当日风格", display_market_summary["market_tone"])
        summary_metric2.metric("延续判断", display_market_summary["continuation_label"])
        summary_metric3.metric("回调风险", display_market_summary["pullback_risk"])
        summary_metric4.metric(
            "两市成交额",
            f"{snapshot_summary['turnover_total_billion']:.0f} 亿" if isinstance(snapshot_summary.get("turnover_total_billion"), (int, float)) else "-",
        )
        summary_metric5.metric(
            "上涨占比",
            f"{snapshot_summary['up_ratio']:.0%}" if isinstance(snapshot_summary.get("up_ratio"), (int, float)) else "-",
        )

        breadth_col1, breadth_col2, breadth_col3, breadth_col4, breadth_col5 = st.columns(5)
        breadth_col1.metric("上涨家数", int(snapshot_summary.get("up_count") or 0))
        breadth_col2.metric("下跌家数", int(snapshot_summary.get("down_count") or 0))
        breadth_col3.metric("平盘家数", int(snapshot_summary.get("flat_count") or 0))
        breadth_col4.metric("涨超5%", int(snapshot_summary.get("gt5_count") or 0))
        breadth_col5.metric("跌超5%", int(snapshot_summary.get("lt5_count") or 0))

        signal_cards_markup = "".join(
            f"<div class='market-signal-card market-signal-{signal['tone']}'><h4>{signal['title']}</h4><div class='status'>{signal['status']}</div><div class='delta'>{signal.get('delta_status', '')}</div><div class='delta-note'>{signal.get('delta_note', '')}</div><div class='hint'>{signal['hint']}</div></div>"
            for signal in market_signals
        )
        if signal_cards_markup:
            st.markdown(f"<div class='market-signal-grid'>{signal_cards_markup}</div>", unsafe_allow_html=True)

        if display_market_summary["continuation_label"] in {"继续上攻", "偏强震荡上行"} and display_market_summary["pullback_risk"] != "高":
            st.success(display_market_summary["summary"])
        elif display_market_summary["pullback_risk"] == "高":
            st.error(display_market_summary["summary"])
        else:
            st.warning(display_market_summary["summary"])
        st.caption(display_market_summary["risk_hint"])

        left_col, right_col = st.columns([1.15, 1])
        with left_col:
            indicator_df = pd.DataFrame(
                [
                    {"指标": "上涨/下跌家数", "数值": f"{int(snapshot_summary.get('up_count') or 0)} / {int(snapshot_summary.get('down_count') or 0)}", "解读": "用于观察赚钱效应是否仍在扩散。"},
                    {"指标": "全市场平均涨幅", "数值": f"{snapshot_summary['mean_pct']:.2f}%" if isinstance(snapshot_summary.get('mean_pct'), (int, float)) else "-", "解读": "比单看指数更能反映真实市场温度。"},
                    {"指标": "两市成交额", "数值": f"{snapshot_summary['turnover_total_billion']:.0f} 亿" if isinstance(snapshot_summary.get('turnover_total_billion'), (int, float)) else "-", "解读": "量能是否足够，是判断指数能否持续上攻的核心条件。"},
                    {"指标": "中位换手率", "数值": f"{snapshot_summary['turnover_rate_median']:.2f}%" if isinstance(snapshot_summary.get('turnover_rate_median'), (int, float)) else "-", "解读": "反映中位股票活跃度，避免只看少数热门股。"},
                    {"指标": "高换手个股数", "数值": int(snapshot_summary.get('high_turnover_count') or 0), "解读": "换手率 >= 10% 的个股数量，可观察短线情绪拥挤度。"},
                    {"指标": "指数趋势", "数值": str(market_overview.get('index_trend', '未知')), "解读": str(market_overview.get('index_reason', '指数趋势未知'))},
                ]
            )
            indicator_df["数值"] = indicator_df["数值"].astype(str)
            st.caption("关键盘面指标")
            st.dataframe(indicator_df, width="stretch", hide_index=True)

        with right_col:
            st.caption("复盘重点")
            st.markdown("\n".join(f"- {item}" for item in market_summary["watch_items"]))
            if snapshot_timestamp is not None:
                st.caption(f"盘面快照时间：{snapshot_timestamp:%Y-%m-%d %H:%M:%S}")
            st.caption(f"盘面快照来源：{snapshot_source_label}")
            st.caption("说明：当前全市场快照没有统一成交量字段，因此这里按行业常用复盘口径，使用成交额、换手率、上涨/下跌家数和强弱分布做代理。")

        turnover_df = snapshot_summary.get("top_turnover_df", pd.DataFrame()).copy()
        if not turnover_df.empty:
            if "symbol" in turnover_df.columns:
                turnover_df["symbol"] = turnover_df["symbol"].map(format_symbol)
            if "last_price" in turnover_df.columns:
                turnover_df["last_price"] = turnover_df["last_price"].map(format_two_decimals)
            for column in ["pct_change", "turnover_rate"]:
                if column in turnover_df.columns:
                    turnover_df[column] = turnover_df[column].map(lambda value: f"{float(value):+.2f}%" if pd.notna(value) else "-")
            if "turnover_amount" in turnover_df.columns:
                turnover_df["turnover_amount"] = turnover_df["turnover_amount"].map(lambda value: f"{float(value) / 100000000:.2f} 亿" if pd.notna(value) else "-")
            turnover_df = turnover_df.rename(columns={"symbol": "代码", "name": "名称", "last_price": "价格", "pct_change": "涨跌幅", "turnover_amount": "成交额", "turnover_rate": "换手率"})
            st.caption("高成交主线观察")
            st.dataframe(turnover_df, width="stretch", hide_index=True)

        strongest_df = snapshot_summary.get("strongest_df", pd.DataFrame()).copy()
        weakest_df = snapshot_summary.get("weakest_df", pd.DataFrame()).copy()
        if not strongest_df.empty or not weakest_df.empty:
            leader_col, weak_col = st.columns(2)
            if not strongest_df.empty:
                if "symbol" in strongest_df.columns:
                    strongest_df["symbol"] = strongest_df["symbol"].map(format_symbol)
                if "last_price" in strongest_df.columns:
                    strongest_df["last_price"] = strongest_df["last_price"].map(format_two_decimals)
                for column in ["pct_change", "turnover_rate"]:
                    if column in strongest_df.columns:
                        strongest_df[column] = strongest_df[column].map(lambda value: f"{float(value):+.2f}%" if pd.notna(value) else "-")
                if "turnover_amount" in strongest_df.columns:
                    strongest_df["turnover_amount"] = strongest_df["turnover_amount"].map(lambda value: f"{float(value) / 100000000:.2f} 亿" if pd.notna(value) else "-")
                strongest_df = strongest_df.rename(columns={"symbol": "代码", "name": "名称", "last_price": "价格", "pct_change": "涨跌幅", "turnover_amount": "成交额", "turnover_rate": "换手率"})
                leader_col.caption("当日强势代表")
                leader_col.dataframe(strongest_df, width="stretch", hide_index=True)
            if not weakest_df.empty:
                if "symbol" in weakest_df.columns:
                    weakest_df["symbol"] = weakest_df["symbol"].map(format_symbol)
                if "last_price" in weakest_df.columns:
                    weakest_df["last_price"] = weakest_df["last_price"].map(format_two_decimals)
                for column in ["pct_change", "turnover_rate"]:
                    if column in weakest_df.columns:
                        weakest_df[column] = weakest_df[column].map(lambda value: f"{float(value):+.2f}%" if pd.notna(value) else "-")
                if "turnover_amount" in weakest_df.columns:
                    weakest_df["turnover_amount"] = weakest_df["turnover_amount"].map(lambda value: f"{float(value) / 100000000:.2f} 亿" if pd.notna(value) else "-")
                weakest_df = weakest_df.rename(columns={"symbol": "代码", "name": "名称", "last_price": "价格", "pct_change": "涨跌幅", "turnover_amount": "成交额", "turnover_rate": "换手率"})
                weak_col.caption("当日承压代表")
                weak_col.dataframe(weakest_df, width="stretch", hide_index=True)

        if not market_index_history.empty:
            chart_col, stat_col = st.columns([1.55, 1])
            display_index_history = market_index_history.tail(120).copy()
            chart_col.caption("基准指数近阶段趋势")
            chart_col.plotly_chart(build_market_index_figure(display_index_history), width="stretch")

            latest_index = market_index_history.iloc[-1]
            stat_col.caption("指数趋势补充")
            stat1, stat2 = stat_col.columns(2)
            stat1.metric("最新收盘", format_two_decimals(latest_index.get("close")))
            stat2.metric("MA20-MA60", format_two_decimals((latest_index.get("ma_short") - latest_index.get("ma_long")) if pd.notna(latest_index.get("ma_short")) and pd.notna(latest_index.get("ma_long")) else pd.NA))
            stat3, stat4 = stat_col.columns(2)
            stat3.metric("近5日", format_percent_value((market_overview.get("index_return_5d") or 0) * 100) if isinstance(market_overview.get("index_return_5d"), (int, float)) else "-")
            stat4.metric("近10日", format_percent_value((market_overview.get("index_return_10d") or 0) * 100) if isinstance(market_overview.get("index_return_10d"), (int, float)) else "-")
            stat_col.caption("图中黑线为基准指数，橙线/蓝线分别为 MA20 和 MA60，用于辅助判断当前是趋势上移还是高位震荡。")


def render_symbol_section(
    config_path: str,
    config: dict,
    config_fingerprint: str,
    scan_df: pd.DataFrame,
    intraday_preset: str = "跟随配置",
) -> None:
    st.subheader("个股分析")
    symbol_intraday_cfg = resolve_intraday_prompt_config(config, intraday_preset)
    st.caption(format_intraday_preset_caption(intraday_preset, symbol_intraday_cfg))
    options = []
    option_map: dict[str, str] = {}
    if not scan_df.empty:
        for item in scan_df.itertuples(index=False):
            label = f"{item.symbol} | {item.name}"
            options.append(label)
            option_map[label] = item.symbol

    select_col, input_col = st.columns([1.2, 1])
    selected_label = select_col.selectbox("从扫描结果选择", options=options, index=0 if options else None, placeholder="先执行扫描或手动输入代码")
    manual_symbol = input_col.text_input("手动输入股票代码", value="")

    left_col, right_col = st.columns([1, 1])
    start_date = left_col.date_input("回看起始日", value=date(2024, 1, 1))
    end_date = right_col.date_input("结束日", value=date.today())

    symbol = manual_symbol.strip() or option_map.get(selected_label, "") or "600519"

    if not symbol:
        st.info("请输入股票代码，或先运行一次市场扫描。")
        return

    with st.spinner(f"正在加载 {symbol} 的历史数据..."):
        history = cached_history(
            format_symbol(symbol),
            start_date,
            end_date,
            config_path,
            config_fingerprint,
        )
    if history.empty:
        st.error("该股票没有拿到历史数据。")
        return

    snapshot = cached_symbol_snapshot(format_symbol(symbol))
    intraday_context = build_intraday_context(history, snapshot)
    market_regime = resolve_dashboard_market_regime(config_path, config_fingerprint, scan_df)
    history_source_label = format_data_source_label(
        history.attrs.get("history_source", ""),
        history.attrs.get("history_source_note", ""),
    )
    snapshot_source_label = format_data_source_label(
        (snapshot or {}).get("snapshot_source", ""),
        (snapshot or {}).get("snapshot_source_note", ""),
    )

    latest_bar_date = pd.to_datetime(history["date"]).max().date()
    if latest_bar_date < end_date:
        st.warning(
            f"当前个股分析的结束日期是 {end_date:%Y-%m-%d}，但最新可用日线只到 {latest_bar_date:%Y-%m-%d}。"
            " 如果你要做下一交易日计划，应该优先等待最新交易日收盘数据落地。"
        )
    else:
        st.caption(f"当前个股分析使用的最新日线日期：{latest_bar_date:%Y-%m-%d}")
    st.caption(f"当前个股日线来源：{history_source_label}")

    summary = latest_signal_summary(history, config)
    if summary:
        intraday_prompt = build_intraday_execution_prompt(
            history,
            summary,
            snapshot,
            market_regime,
            symbol_intraday_cfg,
        )
        info1, info2, info3, info4, info5, info6 = st.columns(6)
        info1.metric("最新收盘价", f"{summary['close']:.2f}")
        info2.metric("最新评分", format_score_value(summary["score"]))
        info3.metric("买点", "是" if summary["buy_signal"] else "否")
        info4.metric("补仓", "是" if summary["add_on_signal"] else "否")
        info5.metric("卖点", "是" if summary["sell_signal"] else "否")
        status_age = summary["signal_age"] if summary["signal_age"] is not None else summary.get("holding_days")
        info6.metric("信号/持仓天数", f"{status_age} 天" if status_age is not None else "无")
        st.info(f"当前建议动作：{summary['action']}。{summary['execution_advice']}")
        status1, status2, status3 = st.columns(3)
        status1.metric("当前状态", summary.get("position_state", "空仓"))
        status2.metric("ATR风控价", f"{summary['atr_stop_price']:.2f}" if summary.get("atr_stop_price") is not None else "-")
        status3.metric("买点类型", summary.get("entry_signal_type") or "-")
        st.caption(f"ATR止损是否触发：{'是' if summary.get('atr_stop_signal') else '否'}")
        if intraday_prompt is not None:
            prompt_cols = st.columns(3)
            prompt_cols[0].metric("盘中动作", intraday_prompt["action"])
            prompt_cols[1].metric("市场环境", format_market_regime_label(market_regime))
            prompt_cols[2].metric("盘中快照", "可用" if intraday_context is not None else "不可用")
            if intraday_prompt["action"] == "盘中试买":
                st.success(f"盘中提示：{intraday_prompt['message']}")
            elif intraday_prompt["action"] == "盘中撤退":
                st.error(f"盘中提示：{intraday_prompt['message']}")
            elif intraday_prompt["action"] != "-":
                st.warning(f"盘中提示：{intraday_prompt['message']}")
        if intraday_context is not None:
            live_cols = st.columns(4)
            live_cols[0].metric(
                "当前价格",
                f"{intraday_context['live_price']:.2f}",
                delta=f"{intraday_context['pct_from_close']:+.2%}",
            )
            live_cols[1].metric("实时成交额", f"{intraday_context['turnover_amount'] / 100000000:.2f} 亿")
            live_cols[2].metric(
                "相对 MA20",
                "上方" if intraday_context["above_ma_fast"] else "下方",
                delta=(
                    f"{intraday_context['live_price'] - intraday_context['ma_fast']:+.2f}"
                    if intraday_context["ma_fast"] is not None
                    else None
                ),
            )
            live_cols[3].metric(
                "相对突破位",
                "上方" if intraday_context["above_breakout"] else "下方",
                delta=(
                    f"{intraday_context['live_price'] - intraday_context['breakout_high']:+.2f}"
                    if intraday_context["breakout_high"] is not None
                    else None
                ),
            )
            st.caption(f"上方实时数据来自盘中快照（{snapshot_source_label}）；买卖信号本身仍以收盘日线确认。")
            if intraday_context["is_material"]:
                st.warning(f"盘中偏离提示：{intraday_context['message']}")
        if summary["sell_reason"]:
            st.caption(f"最近卖点原因：{summary['sell_reason']}")
        if not scan_df.empty and symbol in set(scan_df["symbol"]):
            selected_row = scan_df.loc[scan_df["symbol"] == symbol].iloc[0]
            if "execution_priority" in selected_row:
                st.caption(f"当前扫描执行优先级：{selected_row['execution_priority']}。")

    figure = build_price_figure(history)
    st.plotly_chart(figure, width="stretch")

    backtest = run_single_symbol_backtest(history, symbol, atr_stop_multiple=config["strategy"]["atr_stop_multiple"])
    bt1, bt2, bt3, bt4, bt5, bt6 = st.columns(6)
    bt1.metric("交易次数", backtest.trades)
    bt2.metric("胜率", f"{backtest.win_rate:.2%}")
    bt3.metric("累计收益", f"{backtest.total_return:.2%}")
    bt4.metric("最大回撤", f"{backtest.max_drawdown:.2%}")
    bt5.metric("单笔期望", f"{backtest.expectancy:.2%}")
    bt6.metric("利润因子", "∞" if backtest.profit_factor == float("inf") else f"{backtest.profit_factor:.2f}")
    bt7, bt8 = st.columns(2)
    bt7.metric("平均盈利", f"{backtest.avg_win_return:.2%}")
    bt8.metric("平均亏损", f"{backtest.avg_loss_return:.2%}")
    st.caption("回测口径：信号在收盘确认，默认按下一交易日开盘价执行。")
    st.caption("单笔期望看每笔平均能赚多少；利润因子看总盈利相对总亏损的放大倍数，比单看胜率更接近真实盈亏质量。")

    signal_stats = analyze_signal_statistics(history)
    if not signal_stats.empty:
        display_stats = signal_stats.copy()
        for column in display_stats.columns:
            if column.endswith("平均收益") or column.endswith("命中率"):
                display_stats[column] = display_stats[column].map(
                    lambda value: f"{value:.2%}" if pd.notna(value) else "-"
                )
        st.caption("历史信号统计验证")
        st.dataframe(display_stats, width="stretch", hide_index=True)
        st.caption("说明：买点/补仓的命中率表示未来收益为正的比例；卖点/快退的命中率表示信号后股价继续下跌的比例。")

    entry_slice_stats = analyze_entry_slices(history)
    if not entry_slice_stats.empty:
        display_entry_slices = entry_slice_stats.copy()
        for column in display_entry_slices.columns:
            if column.endswith("平均收益") or column.endswith("命中率"):
                display_entry_slices[column] = display_entry_slices[column].map(
                    lambda value: f"{value:.2%}" if pd.notna(value) else "-"
                )
        display_entry_slices = display_entry_slices.rename(
            columns={
                "signal_kind": "信号类别",
                "entry_signal_type": "买点类型",
                "market_regime": "市场环境",
            }
        )
        st.caption("按买点类型与市场环境拆分的 5 日样本表现")
        st.dataframe(display_entry_slices, width="stretch", hide_index=True)
        st.caption("这张表更适合看哪类入场在什么环境里更稳定，而不是只看总体胜率。")

    exit_slice_stats = analyze_exit_slices(history)
    if not exit_slice_stats.empty:
        display_exit_slices = exit_slice_stats.copy()
        for column in display_exit_slices.columns:
            if column.endswith("平均跌幅") or column.endswith("有效率"):
                display_exit_slices[column] = display_exit_slices[column].map(
                    lambda value: f"{value:.2%}" if pd.notna(value) else "-"
                )
        display_exit_slices = display_exit_slices.rename(
            columns={
                "sell_reason": "卖点原因",
                "market_regime": "市场环境",
            }
        )
        st.caption("按卖点原因与市场环境拆分的 5 日样本表现")
        st.dataframe(display_exit_slices, width="stretch", hide_index=True)
        st.caption("这张表更适合看不同离场原因在什么环境里更有效，帮助区分趋势破坏、快退和止损的质量。")

    signal_table = history.loc[
        history["buy_signal"] | history["add_on_signal"] | history["sell_signal"],
        [
            "date",
            "close",
            "position_state",
            "entry_signal_type",
            "atr_stop_price",
            "buy_signal",
            "add_on_signal",
            "sell_signal",
            "sell_reason",
            "score",
            "rsi",
            "volume_ratio",
        ],
    ].copy()
    signal_table["date"] = signal_table["date"].dt.strftime("%Y-%m-%d")
    signal_table["close"] = signal_table["close"].map(lambda value: f"{float(value):.2f}" if pd.notna(value) else "-")
    signal_table["buy_signal"] = signal_table["buy_signal"].map(lambda value: "是" if bool(value) else "否")
    signal_table["add_on_signal"] = signal_table["add_on_signal"].map(lambda value: "是" if bool(value) else "否")
    signal_table["sell_signal"] = signal_table["sell_signal"].map(lambda value: "是" if bool(value) else "否")
    signal_table["atr_stop_price"] = signal_table["atr_stop_price"].map(lambda value: f"{float(value):.2f}" if pd.notna(value) else "-")
    signal_table["score"] = signal_table["score"].map(format_score_value)
    for numeric_column in ["volume_ratio", "rsi"]:
        if numeric_column in signal_table.columns:
            signal_table[numeric_column] = signal_table[numeric_column].map(format_two_decimals)
    signal_table = drop_uniform_display_columns(
        signal_table,
        ["position_state", "entry_signal_type", "atr_stop_price", "sell_reason"],
    )
    signal_table = signal_table.rename(
        columns={
            "date": "日期",
            "close": "收盘价",
            "position_state": "持仓状态",
            "entry_signal_type": "买点类型",
            "atr_stop_price": "ATR风控价",
            "buy_signal": "买点信号",
            "add_on_signal": "补仓信号",
            "sell_signal": "卖点信号",
            "sell_reason": "卖点原因",
            "score": "评分",
            "rsi": "RSI",
            "volume_ratio": "量比",
        }
    )
    st.caption("最近触发的交易信号")
    st.dataframe(signal_table.tail(20), width="stretch", hide_index=True)


def render_watchlist_section(config_path: str, config_fingerprint: str, intraday_preset: str = "跟随配置") -> None:
    st.subheader("自选关注")
    watchlist_intraday_cfg = resolve_intraday_prompt_config(load_config(config_path), intraday_preset)
    st.caption(format_intraday_preset_caption(intraday_preset, watchlist_intraday_cfg))
    default_symbols = st.session_state.get("watchlist_symbols")
    if default_symbols is None:
        default_symbols = load_watchlist_symbols()
        st.session_state.watchlist_symbols = default_symbols

    default_text = "\n".join(default_symbols)
    input_col, action_col = st.columns([2.2, 1])
    raw_watchlist = input_col.text_area(
        "输入自选股票代码",
        value=default_text,
        height=110,
        placeholder="每行一个代码，或用逗号分隔，例如：600513, 300308, 688525",
    )
    save_watchlist = action_col.button("保存自选", width="stretch")
    refresh_watchlist = action_col.button("刷新自选", width="stretch")

    parsed_symbols = parse_watchlist_symbols(raw_watchlist)
    if save_watchlist:
        save_watchlist_symbols(parsed_symbols)
        st.session_state.watchlist_symbols = parsed_symbols
        st.success(f"已保存 {len(parsed_symbols)} 只自选股。")

    watchlist_symbols = parsed_symbols if raw_watchlist.strip() else st.session_state.get("watchlist_symbols", [])
    st.session_state.watchlist_symbols = watchlist_symbols
    if not watchlist_symbols:
        st.info("在这里填入你想长期跟踪的股票代码，看板会单独给出操作提示。")
        return

    watchlist_refresh_token = int(st.session_state.get("watchlist_refresh_token", 0))
    if refresh_watchlist:
        watchlist_refresh_token += 1
        st.session_state.watchlist_refresh_token = watchlist_refresh_token

    latest_scan_df, _, _, _ = load_latest_scan_from_disk(config_fingerprint, None)
    watchlist_market_regime = resolve_dashboard_market_regime(config_path, config_fingerprint, latest_scan_df)

    with st.spinner("正在更新自选股操作提示..."):
        watchlist_df = cached_watchlist_dataframe(
            "\n".join(watchlist_symbols),
            config_path,
            config_fingerprint,
            watchlist_refresh_token,
            watchlist_market_regime,
            intraday_preset,
        )

    if watchlist_df.empty:
        st.warning("当前自选股没有可展示的数据。")
        return

    display_df = watchlist_df.drop(columns=[column for column in ["action_rank"] if column in watchlist_df.columns]).copy()
    if "live_delta" in display_df.columns:
        display_df["live_delta"] = display_df["live_delta"].map(lambda value: f"{value:+.2%}" if pd.notna(value) else "-")
    if "score" in display_df.columns:
        display_df["score"] = display_df["score"].map(format_score_value)
    for numeric_column in ["volume_ratio", "rsi"]:
        if numeric_column in display_df.columns:
            display_df[numeric_column] = display_df[numeric_column].map(format_two_decimals)
    if "signal_age" in display_df.columns:
        display_df["signal_age"] = display_df["signal_age"].map(lambda value: f"{int(value)} 天" if pd.notna(value) else "-")
    if "holding_days" in display_df.columns:
        display_df["holding_days"] = display_df["holding_days"].map(lambda value: f"{int(value)} 天" if pd.notna(value) else "-")
    if "date" in display_df.columns:
        display_df["date"] = display_df["date"].map(lambda value: value.strftime("%Y-%m-%d") if pd.notna(value) else "-")
    if "close" in display_df.columns:
        display_df["close"] = display_df["close"].map(lambda value: f"{float(value):.2f}" if pd.notna(value) else "-")
    if "live_price" in display_df.columns:
        display_df["live_price"] = display_df["live_price"].map(lambda value: f"{float(value):.2f}" if pd.notna(value) else "-")
    if "atr_stop_price" in display_df.columns:
        display_df["atr_stop_price"] = display_df["atr_stop_price"].map(lambda value: f"{float(value):.2f}" if pd.notna(value) else "-")

    display_df = display_df.rename(
        columns={
            "symbol": "代码",
            "name": "名称",
            "date": "最新日线",
            "close": "最新收盘",
            "live_price": "当前价格",
            "live_delta": "实时偏离",
            "action": "当前动作",
            "intraday_action": "盘中动作",
            "signal_age": "信号年龄",
            "holding_days": "持仓天数",
            "position_state": "当前状态",
            "entry_signal_type": "买点类型",
            "atr_stop_price": "ATR风控价",
            "sell_reason": "最近卖点原因",
            "score": "评分",
            "volume_ratio": "量比",
            "rsi": "RSI",
            "intraday_hint": "盘中提示",
            "hint": "操作提示",
        }
    )
    display_df = drop_uniform_display_columns(
        display_df,
        ["当前状态", "买点类型", "ATR风控价", "最近卖点原因", "盘中动作", "盘中提示"],
    )
    st.caption(
        f"当前已跟踪 {len(watchlist_symbols)} 只自选股。操作提示仍以最新完整日线为主；新增的盘中动作/盘中提示只作为快照辅助，不替代收盘确认。"
    )
    st.dataframe(style_watchlist_dataframe(display_df), width="stretch", hide_index=True)


def render_fund_section() -> None:
    st.subheader("基金涨幅看板")

    control_col1, control_col2 = st.columns([1.6, 1])
    search_text = control_col1.text_input("筛选基金", value="", placeholder="输入基金代码或名称")
    row_limit = int(control_col2.number_input("显示数量", min_value=20, max_value=500, value=100, step=20))

    with st.spinner("正在加载基金涨幅数据..."):
        fund_df, fund_source, fund_error = cached_fund_rank()

    if fund_df.empty:
        st.warning("当前没有可展示的基金数据。")
        if fund_error:
            st.caption(f"基金数据失败原因：{fund_error}")
        return

    if fund_source == "本地缓存":
        st.warning("基金实时接口本次不可用，当前展示的是本地缓存结果。")
    if fund_error and fund_source != "无可用数据":
        st.caption(f"实时接口失败原因：{fund_error}")

    full_fund_df = build_fund_scored_dataframe(fund_df)
    shortlist_size = min(max(row_limit, 20), 40)
    full_fund_df = enhance_fund_scores(full_fund_df, shortlist_size)
    full_fund_df = build_fund_action_signals(full_fund_df)

    working_df = full_fund_df.copy()
    if search_text.strip():
        keyword = search_text.strip()
        working_df = working_df[
            working_df["基金代码"].astype(str).str.contains(keyword, case=False, na=False)
            | working_df["基金简称"].astype(str).str.contains(keyword, case=False, na=False)
        ]

    latest_date = working_df["日期"].max() if "日期" in working_df.columns else None

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("基金数量", int(len(working_df)))
    metric2.metric("平均评分", format_two_decimals(working_df["基金评分"].mean()))
    metric3.metric("近6月均值", format_percent_value(working_df["近6月"].mean()))
    metric4.metric("近1年均值", format_percent_value(working_df["近1年"].mean()))
    st.caption(f"当前基金数据来源：{fund_source}")
    if latest_date is not None and pd.notna(latest_date):
        st.caption(f"基金数据日期：{latest_date:%Y-%m-%d}")
    category_counts = working_df["基金类别"].value_counts()
    count1, count2, count3, count4 = st.columns(4)
    count1.metric("偏股基金", int(category_counts.get("偏股基金", 0)))
    count2.metric("主动权益基金", int(category_counts.get("主动权益基金", 0)))
    count3.metric("ETF/指数基金", int(category_counts.get("ETF/指数基金", 0)))
    count4.metric("债基", int(category_counts.get("债基", 0)))

    st.caption("基金判断 V4：在原有评分基础上，额外结合近 1/3/6 月收益结构，细化为一次性买入、定投、观察、减仓、清仓。基金更适合周度/月度调仓，不像股票那样强调日线买卖点。")
    st.divider()
    render_fund_watchlist_section(full_fund_df)
    st.divider()
    category_tabs = st.tabs(["偏股基金", "主动权益基金", "ETF/指数基金", "债基"])
    with category_tabs[0]:
        render_fund_category_table(working_df, "偏股基金", row_limit)
    with category_tabs[1]:
        render_fund_category_table(working_df, "主动权益基金", row_limit)
    with category_tabs[2]:
        render_fund_category_table(working_df, "ETF/指数基金", row_limit)
    with category_tabs[3]:
        render_fund_category_table(working_df, "债基", row_limit)


def main() -> None:
    st.title("A股量化分析看板")
    st.caption("基于 AkShare 的日线扫描、信号识别与单股回测面板")

    config = load_config(CONFIG_PATH)
    run_startup_maintenance(CONFIG_PATH, int(config["scan"]["history_days"]))
    config_fingerprint = compute_config_fingerprint(CONFIG_PATH)
    st.sidebar.header("策略参数概览")
    st.sidebar.subheader("趋势框架")
    st.sidebar.write(f"均线组: MA{config['strategy']['ma_fast']} / MA{config['strategy']['ma_mid']} / MA{config['strategy']['ma_slow']}")
    st.sidebar.caption("用于定义主趋势。均线越长，噪音越少，但出手次数也会减少。")
    st.sidebar.write(f"突破窗口: {config['strategy']['breakout_window']} 日")
    st.sidebar.caption("窗口越长，越偏中期趋势突破；窗口越短，信号更多，但假突破也会增多。")
    st.sidebar.subheader("动量确认")
    st.sidebar.write(f"量能阈值: {config['strategy']['volume_ratio_min']:.1f} x 20日均量")
    st.sidebar.caption("当前阈值偏确认型，更像牺牲部分出手频率，换更强的突破质量。")
    st.sidebar.write(
        "RSI过滤: 关闭（仅作辅助观察）"
        if not config['strategy'].get('use_rsi_filter', False)
        else f"RSI过滤: {config['strategy']['rsi_min']} - {config['strategy']['rsi_max']}"
    )
    st.sidebar.caption("RSI 目前不做硬门槛，避免在强趋势里过早把高景气龙头排除掉。")
    st.sidebar.subheader("扫描池抽样")
    st.sidebar.write(f"抽样模式: {config['market'].get('candidate_selection_mode', 'turnover_amount')}")
    st.sidebar.caption("当前更偏活跃度和资金参与度，不是简单按成交额从高到低排序。")
    st.sidebar.write(f"最小换手率: {float(config['market'].get('min_turnover_rate', 0) or 0):.1f}%")
    st.sidebar.write(f"候选扫描数: {config['market']['max_symbols']}")
    st.sidebar.caption("扫描池越大越容易覆盖主线，但耗时和数据源不稳定性也会一起上升。")
    st.sidebar.subheader("风险控制")
    st.sidebar.write(f"ATR窗口: {config['strategy']['atr_window']} 日")
    st.sidebar.write(f"止损: {config['strategy']['atr_stop_multiple']:.1f} x ATR")
    st.sidebar.caption("ATR 止损越紧，胜率可能更平滑，但也更容易在强趋势回撤中被提前洗掉。")
    st.sidebar.subheader("结果质量")
    st.sidebar.write(f"严格收盘后模式: {'开启' if config['scan'].get('strict_post_close_mode', False) else '关闭'}")
    st.sidebar.write(f"单次扫描超时: {int(config['scan'].get('scan_timeout_seconds', 0) or 0)} 秒")
    st.sidebar.caption("如果数据源波动或超时，只会先得到 partial / stale 结果，交易上应降低信号信任度。")
    st.sidebar.subheader("盘中提示")
    intraday_preset = st.sidebar.radio(
        "盘中提示档位",
        options=["跟随配置", "保守", "积极", "超积极"],
        index=0,
        help="只影响当前看板会话里的盘中提示强弱，不会改写 YAML 配置。",
    )
    active_intraday_cfg = resolve_intraday_prompt_config(config, intraday_preset)
    st.sidebar.write(
        f"突破确认: +{active_intraday_cfg['breakout_confirm_pct']:.2%} / 试买评分 >= {int(active_intraday_cfg['trial_score_min'])}"
    )
    st.sidebar.write(
        f"最大追价: +{active_intraday_cfg['breakout_chase_limit_pct']:.1%} / 扫描试买最小涨幅: +{active_intraday_cfg['scan_trial_min_live_delta_pct']:.2%}"
    )
    sidebar_preset_warning = get_intraday_preset_warning(intraday_preset)
    if sidebar_preset_warning:
        st.sidebar.warning(sidebar_preset_warning)
    st.sidebar.subheader("市场环境")
    st.sidebar.write(f"指数过滤: {'开启' if config['market_timing']['use_index_filter'] else '关闭'}")
    st.sidebar.write(
        f"宽度阈值: 强势环境 >= {config['market_timing']['min_trend_breadth_risk_on']:.0%} / 中性环境 >= {config['market_timing']['min_trend_breadth_neutral']:.0%}"
    )
    st.sidebar.caption("指数过滤失败时会回退到宽度代理；这时环境标签可用，但应比平时更谨慎。")
    st.sidebar.info("首次扫描会稍慢，因为需要拉取并缓存全市场日线数据。")

    st.markdown(PAGE_TABS_CSS, unsafe_allow_html=True)
    active_page = st.radio(
        "页面",
        options=["股票", "市场", "基金", "黄金"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    if active_page == "股票":
        scan_df = render_scan_section(CONFIG_PATH, intraday_preset)
        st.divider()
        render_watchlist_section(CONFIG_PATH, config_fingerprint, intraday_preset)
        st.divider()
        render_symbol_section(CONFIG_PATH, config, config_fingerprint, scan_df, intraday_preset)
    elif active_page == "市场":
        market_scan_df = st.session_state.get("scan_df", pd.DataFrame())
        market_status_message = None
        if market_scan_df.empty:
            market_scan_df, market_status_message, _, _ = load_latest_scan_from_disk(config_fingerprint, None)
        if market_status_message:
            st.info(market_status_message)
        elif market_scan_df.empty:
            st.info("当前没有可用的扫描缓存，市场总览会先按空结果展示。你也可以先去“股票”页刷新一次扫描。")
        render_market_overview_section(CONFIG_PATH, config_fingerprint, market_scan_df)
    elif active_page == "基金":
        render_fund_section()
    else:
        render_gold_section()


if __name__ == "__main__":
    main()
