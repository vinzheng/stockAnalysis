from __future__ import annotations

import atexit
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from io import StringIO
import os
from pathlib import Path
import re
import time

import akshare as ak
import baostock as bs
import baostock.common.context as baostock_context
import pandas as pd
from pandas.tseries.offsets import BDay

from ashare_quant.config import load_config

ef = None
_efinance_import_attempted = False

try:
    import tushare as ts
except ImportError:  # pragma: no cover - optional dependency until installed
    ts = None

HISTORY_COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "turnover",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "振幅": "amplitude",
    "换手率": "turnover_rate",
}

DAILY_FALLBACK_COLUMN_MAP = {
    "date": "date",
    "open": "open",
    "close": "close",
    "high": "high",
    "low": "low",
    "volume": "volume",
    "amount": "turnover",
    "turnover": "turnover_rate",
}

BAOSTOCK_COLUMN_MAP = {
    "date": "date",
    "open": "open",
    "close": "close",
    "high": "high",
    "low": "low",
    "volume": "volume",
    "amount": "turnover",
    "turn": "turnover_rate",
}

INDEX_COLUMN_MAP = {
    "date": "date",
    "open": "open",
    "close": "close",
    "high": "high",
    "low": "low",
    "volume": "volume",
    "amount": "turnover",
}

SNAPSHOT_REQUIRED_COLUMNS = ["代码", "名称", "最新价", "成交额"]
SNAPSHOT_OPTIONAL_COLUMNS = ["换手率", "涨跌幅", "振幅"]
SNAPSHOT_CACHE_NAME = "latest_snapshot.csv"
ACTIVE_SNAPSHOT_CACHE_NAME = "latest_snapshot_active.csv"
EFINANCE_ADJUST_MAP = {"hfq": 2, "qfq": 1, "": 0, None: 0, "none": 0}
HISTORY_CACHE_FILE_PATTERN = re.compile(r"^(?:benchmark_[a-z0-9]+_\d{8}_\d{8}|(?:sh|sz|bj)?[a-z0-9]+_\d{8}_\d{8}_[^.]+)\.csv$", re.IGNORECASE)
SNAPSHOT_COLUMN_ALIASES = {
    "代码": ["代码", "股票代码"],
    "名称": ["名称", "股票名称"],
    "最新价": ["最新价"],
    "成交额": ["成交额"],
    "换手率": ["换手率"],
    "涨跌幅": ["涨跌幅"],
    "振幅": ["振幅"],
}


def _get_efinance_module():
    global ef, _efinance_import_attempted
    if _efinance_import_attempted:
        return ef

    _efinance_import_attempted = True
    try:
        import efinance as imported_ef
    except Exception:  # pragma: no cover - cloud/runtime-specific import failures should trigger fallback
        ef = None
    else:
        ef = imported_ef
    return ef


def _has_usable_snapshot_rows(snapshot: pd.DataFrame) -> bool:
    return not snapshot.empty and snapshot["symbol"].notna().any()


def _has_active_snapshot_activity(snapshot: pd.DataFrame) -> bool:
    if snapshot.empty or "turnover_amount" not in snapshot.columns:
        return False
    turnover_amount = pd.to_numeric(snapshot["turnover_amount"], errors="coerce").fillna(0.0)
    return bool((turnover_amount > 0).any())


def _latest_cache_reference_day(reference: datetime | None = None) -> date:
    current = pd.Timestamp(reference or datetime.now())
    if current.weekday() >= 5:
        return (current - BDay(1)).date()
    return current.date()


def _count_cache_trading_day_lag(cached_date: date | None, target_date: date) -> int | None:
    if cached_date is None:
        return None
    if cached_date >= target_date:
        return 0
    trading_days = pd.date_range(
        start=pd.Timestamp(cached_date) + BDay(1),
        end=pd.Timestamp(target_date),
        freq="B",
    )
    return len(trading_days)


def _is_history_cache_file(path: Path) -> bool:
    return path.is_file() and HISTORY_CACHE_FILE_PATTERN.match(path.name) is not None


def purge_expired_history_cache(
    cache_dir: Path,
    reference_date: date | None = None,
    retention_trading_days: int = 260,
) -> list[Path]:
    target_date = reference_date or _latest_cache_reference_day()
    deleted_paths: list[Path] = []
    for cache_path in cache_dir.glob("*.csv"):
        if not _is_history_cache_file(cache_path):
            continue
        try:
            history_dates = pd.read_csv(cache_path, usecols=["date"], parse_dates=["date"])
        except Exception:
            continue
        if history_dates.empty or "date" not in history_dates.columns:
            continue
        max_cached_timestamp = pd.to_datetime(history_dates["date"], errors="coerce").max()
        if pd.isna(max_cached_timestamp):
            continue
        trading_day_lag = _count_cache_trading_day_lag(max_cached_timestamp.date(), target_date)
        if trading_day_lag is None or trading_day_lag <= retention_trading_days:
            continue
        try:
            cache_path.unlink()
            deleted_paths.append(cache_path)
        except OSError:
            continue
    return deleted_paths


@dataclass(slots=True)
class MarketDataClient:
    cache_dir: Path
    config_path: Path = Path("config/universe.yaml")
    retries: int = 3
    retry_delay_seconds: float = 1.5
    _baostock_logged_in: bool = False
    _tushare_client: object | None = None
    _tushare_token: str | None = None
    _runtime_config: dict[str, object] | None = None
    _last_fetch_details: dict[str, dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.config_path = Path(self.config_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        atexit.register(self._reset_baostock_session)

    def _with_retry(self, operation_name: str, operation):
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                return operation()
            except Exception as error:  # pragma: no cover - network variability
                last_error = error
                if attempt == self.retries:
                    break
                time.sleep(self.retry_delay_seconds * attempt)
        raise RuntimeError(f"{operation_name} failed after {self.retries} attempts") from last_error

    def _run_baostock_call(self, operation):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return operation()

    def _reset_baostock_session(self) -> None:
        try:
            if self._baostock_logged_in:
                try:
                    self._run_baostock_call(bs.logout)
                except Exception:
                    pass
            default_socket = getattr(baostock_context, "default_socket", None)
            if default_socket is not None:
                try:
                    default_socket.close()
                except Exception:
                    pass
            setattr(baostock_context, "default_socket", None)
            if hasattr(baostock_context, "user_id"):
                setattr(baostock_context, "user_id", None)
        except Exception:
            pass
        self._baostock_logged_in = False

    def _snapshot_cache_path(self) -> Path:
        return self.cache_dir / SNAPSHOT_CACHE_NAME

    def _active_snapshot_cache_path(self) -> Path:
        return self.cache_dir / ACTIVE_SNAPSHOT_CACHE_NAME

    def _count_trading_day_lag(self, cached_date: date | None, target_date: date) -> int | None:
        if cached_date is None:
            return None
        if cached_date >= target_date:
            return 0
        trading_days = pd.date_range(
            start=pd.Timestamp(cached_date) + BDay(1),
            end=pd.Timestamp(target_date),
            freq="B",
        )
        return len(trading_days)

    def _set_fetch_detail(self, kind: str, source: str, note: str = "") -> None:
        self._last_fetch_details[kind] = {"source": source, "note": note}

    def get_last_fetch_detail(self, kind: str) -> dict[str, str]:
        return dict(self._last_fetch_details.get(kind, {}))

    def _load_runtime_config(self) -> dict[str, object]:
        if self._runtime_config is not None:
            return self._runtime_config
        if not self.config_path.exists():
            self._runtime_config = {}
            return self._runtime_config
        try:
            loaded_config = load_config(self.config_path)
        except Exception:
            loaded_config = {}
        self._runtime_config = loaded_config
        return self._runtime_config

    def _resolve_optional_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate
        cwd_path = Path.cwd() / candidate
        if cwd_path.exists():
            return cwd_path
        config_relative_path = self.config_path.parent / candidate
        if config_relative_path.exists():
            return config_relative_path
        return cwd_path

    def _load_env_file_value(self, env_path: Path, key: str) -> str:
        if not env_path.exists():
            return ""
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                env_key, env_value = line.split("=", 1)
                if env_key.strip() != key:
                    continue
                cleaned_value = env_value.strip().strip('"').strip("'")
                return cleaned_value
        except OSError:
            return ""
        return ""

    def _get_tushare_token(self) -> str:
        if self._tushare_token is not None:
            return self._tushare_token

        config = self._load_runtime_config()
        data_source_cfg = config.get("data_source", {}) if isinstance(config, dict) else {}
        if not isinstance(data_source_cfg, dict):
            data_source_cfg = {}

        env_keys: list[str] = []
        configured_env_key = str(data_source_cfg.get("tushare_token_env", "") or "").strip()
        if configured_env_key:
            env_keys.append(configured_env_key)
        env_keys.append("TUSHARE_TOKEN")

        unique_env_keys: list[str] = []
        for env_key in env_keys:
            if env_key and env_key not in unique_env_keys:
                unique_env_keys.append(env_key)

        for env_key in unique_env_keys:
            token = os.getenv(env_key, "").strip()
            if token:
                self._tushare_token = token
                return token

        configured_token = str(data_source_cfg.get("tushare_token", "") or "").strip()
        if configured_token:
            self._tushare_token = configured_token
            return configured_token

        dotenv_candidates: list[Path] = []
        configured_dotenv_path = str(data_source_cfg.get("tushare_dotenv_path", "") or "").strip()
        if configured_dotenv_path:
            dotenv_candidates.append(self._resolve_optional_path(configured_dotenv_path))
        for default_path in [".env.local", ".env"]:
            resolved_path = self._resolve_optional_path(default_path)
            if resolved_path not in dotenv_candidates:
                dotenv_candidates.append(resolved_path)

        for dotenv_path in dotenv_candidates:
            for env_key in unique_env_keys:
                token = self._load_env_file_value(dotenv_path, env_key)
                if token:
                    self._tushare_token = token
                    return token

        self._tushare_token = ""
        return self._tushare_token

    def _use_tushare_for_index(self) -> bool:
        config = self._load_runtime_config()
        data_source_cfg = config.get("data_source", {}) if isinstance(config, dict) else {}
        if not isinstance(data_source_cfg, dict):
            return False
        raw_value = data_source_cfg.get("use_tushare_for_index", False)
        if isinstance(raw_value, bool):
            return raw_value
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}

    def _load_cached_history(self, cache_path: Path) -> pd.DataFrame:
        return pd.read_csv(cache_path, parse_dates=["date"])

    def _find_reusable_history_cache(
        self,
        normalized_symbol: str,
        start_date: datetime,
        end_date: datetime,
        adjust: str,
        max_cache_trading_day_lag: int,
    ) -> pd.DataFrame | None:
        if max_cache_trading_day_lag < 0:
            return None

        cache_pattern = f"{normalized_symbol}_{start_date:%Y%m%d}_*_{adjust}.csv"
        best_candidate: tuple[pd.Timestamp, float, pd.DataFrame] | None = None
        for candidate_path in self.cache_dir.glob(cache_pattern):
            try:
                cached_history = self._load_cached_history(candidate_path)
            except Exception:
                continue
            if cached_history.empty:
                continue
            max_cached_timestamp = pd.to_datetime(cached_history["date"], errors="coerce").max()
            if pd.isna(max_cached_timestamp):
                continue
            trading_day_lag = self._count_trading_day_lag(max_cached_timestamp.date(), end_date.date())
            if trading_day_lag is None or trading_day_lag > max_cache_trading_day_lag:
                continue
            candidate_key = (max_cached_timestamp, candidate_path.stat().st_mtime, cached_history)
            if best_candidate is None or candidate_key[:2] > best_candidate[:2]:
                best_candidate = candidate_key

        if best_candidate is None:
            return None
        return best_candidate[2]

    def _find_reusable_index_cache(
        self,
        normalized_symbol: str,
        start_date: datetime,
        end_date: datetime,
        max_cache_trading_day_lag: int = 3,
    ) -> tuple[pd.DataFrame, Path] | None:
        if max_cache_trading_day_lag < 0:
            return None

        cache_pattern = f"benchmark_{normalized_symbol}_*.csv"
        best_candidate: tuple[pd.Timestamp, float, pd.DataFrame, Path] | None = None
        for candidate_path in self.cache_dir.glob(cache_pattern):
            try:
                cached_history = self._load_cached_history(candidate_path)
            except Exception:
                continue
            if cached_history.empty:
                continue
            min_cached_timestamp = pd.to_datetime(cached_history["date"], errors="coerce").min()
            max_cached_timestamp = pd.to_datetime(cached_history["date"], errors="coerce").max()
            if pd.isna(min_cached_timestamp) or pd.isna(max_cached_timestamp):
                continue
            if min_cached_timestamp.date() > start_date.date():
                continue
            trading_day_lag = self._count_trading_day_lag(max_cached_timestamp.date(), end_date.date())
            if trading_day_lag is None or trading_day_lag > max_cache_trading_day_lag:
                continue
            candidate_key = (max_cached_timestamp, candidate_path.stat().st_mtime, cached_history, candidate_path)
            if best_candidate is None or candidate_key[:2] > best_candidate[:2]:
                best_candidate = candidate_key

        if best_candidate is None:
            return None
        return best_candidate[2], best_candidate[3]

    def _index_cache_path(self, symbol: str, start_date: datetime, end_date: datetime) -> Path:
        normalized_symbol = symbol.replace(".", "")
        return self.cache_dir / f"benchmark_{normalized_symbol}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.csv"

    def _normalize_symbol(self, symbol: str) -> str:
        symbol = str(symbol).strip().lower()
        if symbol.startswith(("sh", "sz", "bj")):
            return symbol
        if symbol.startswith(("60", "68", "51", "56", "58")):
            return f"sh{symbol}"
        if symbol.startswith(("00", "30", "12", "15")):
            return f"sz{symbol}"
        if symbol.startswith(("4", "8", "92")):
            return f"bj{symbol}"
        return symbol

    def _normalize_snapshot_symbol(self, symbol: object) -> str:
        cleaned = str(symbol).strip().lower()
        if cleaned.startswith(("sh", "sz", "bj")):
            return cleaned
        digits_only = "".join(character for character in cleaned if character.isdigit())
        if digits_only:
            return digits_only.zfill(6)
        return cleaned

    def _load_cached_snapshot(self, cache_path: Path) -> pd.DataFrame:
        cached_snapshot = pd.read_csv(
            cache_path,
            dtype={"symbol": "string", "name": "string"},
        )
        if "symbol" in cached_snapshot.columns:
            cached_snapshot["symbol"] = cached_snapshot["symbol"].map(self._normalize_snapshot_symbol)
        for numeric_column in ["last_price", "turnover_amount", "turnover_rate", "pct_change", "amplitude"]:
            if numeric_column in cached_snapshot.columns:
                cached_snapshot[numeric_column] = pd.to_numeric(cached_snapshot[numeric_column], errors="coerce")
        return cached_snapshot

    def _normalize_snapshot(self, snapshot: pd.DataFrame) -> pd.DataFrame:
        selected_columns: dict[str, str] = {}
        for canonical_name, aliases in SNAPSHOT_COLUMN_ALIASES.items():
            matched_name = next((column for column in aliases if column in snapshot.columns), None)
            if matched_name is None and canonical_name in SNAPSHOT_REQUIRED_COLUMNS:
                raise KeyError(f"snapshot missing required column for {canonical_name}")
            if matched_name is not None:
                selected_columns[canonical_name] = matched_name

        ordered_columns = [selected_columns[column] for column in SNAPSHOT_REQUIRED_COLUMNS]
        ordered_columns.extend(
            selected_columns[column] for column in SNAPSHOT_OPTIONAL_COLUMNS if column in selected_columns
        )
        snapshot = snapshot[ordered_columns].copy()
        rename_map = {
            selected_columns["代码"]: "symbol",
            selected_columns["名称"]: "name",
            selected_columns["最新价"]: "last_price",
            selected_columns["成交额"]: "turnover_amount",
        }
        if "换手率" in selected_columns:
            rename_map[selected_columns["换手率"]] = "turnover_rate"
        if "涨跌幅" in selected_columns:
            rename_map[selected_columns["涨跌幅"]] = "pct_change"
        if "振幅" in selected_columns:
            rename_map[selected_columns["振幅"]] = "amplitude"
        snapshot = snapshot.rename(columns=rename_map)
        snapshot["symbol"] = snapshot["symbol"].map(self._normalize_snapshot_symbol)
        snapshot["last_price"] = pd.to_numeric(snapshot["last_price"], errors="coerce")
        snapshot["turnover_amount"] = pd.to_numeric(snapshot["turnover_amount"], errors="coerce")
        for column in ["turnover_rate", "pct_change", "amplitude"]:
            if column not in snapshot.columns:
                snapshot[column] = pd.NA
            snapshot[column] = pd.to_numeric(snapshot[column], errors="coerce")
        return snapshot.dropna(subset=["symbol", "last_price", "turnover_amount"])

    def _normalize_efinance_adjust(self, adjust: str) -> int:
        return EFINANCE_ADJUST_MAP.get(adjust, 1)

    def _get_snapshot_from_efinance(self) -> pd.DataFrame:
        ef_module = _get_efinance_module()
        if ef_module is None:
            raise RuntimeError("efinance is not installed")
        snapshot = self._with_retry("fetch efinance universe snapshot", ef_module.stock.get_realtime_quotes)
        return self._normalize_snapshot(snapshot)

    def _get_history_from_efinance(self, symbol: str, start_date: datetime, end_date: datetime, adjust: str) -> pd.DataFrame:
        ef_module = _get_efinance_module()
        if ef_module is None:
            raise RuntimeError("efinance is not installed")

        normalized_symbol = self._normalize_symbol(symbol)
        symbol_code = normalized_symbol[2:] if normalized_symbol.startswith(("sh", "sz", "bj")) else normalized_symbol
        history = self._with_retry(
            f"fetch efinance history for {normalized_symbol}",
            lambda: ef_module.stock.get_quote_history(
                stock_codes=symbol_code,
                beg=start_date.strftime("%Y%m%d"),
                end=end_date.strftime("%Y%m%d"),
                klt=101,
                fqt=self._normalize_efinance_adjust(adjust),
            ),
        )
        if isinstance(history, dict):
            history = history.get(symbol_code, pd.DataFrame())
        if history is None or history.empty:
            return pd.DataFrame()
        return self._normalize_history(history, HISTORY_COLUMN_MAP)

    def _normalize_baostock_symbol(self, symbol: str) -> str:
        normalized_symbol = self._normalize_symbol(symbol)
        if normalized_symbol.startswith(("sh", "sz", "bj")):
            return f"{normalized_symbol[:2]}.{normalized_symbol[2:]}"
        return normalized_symbol

    def _normalize_tushare_stock_symbol(self, symbol: str) -> str:
        normalized_symbol = self._normalize_symbol(symbol)
        if normalized_symbol.startswith("sh"):
            return f"{normalized_symbol[2:]}.SH"
        if normalized_symbol.startswith("sz"):
            return f"{normalized_symbol[2:]}.SZ"
        if normalized_symbol.startswith("bj"):
            return f"{normalized_symbol[2:]}.BJ"
        return normalized_symbol.upper()

    def _normalize_index_symbol(self, symbol: str) -> str:
        cleaned = symbol.strip().lower().replace(".", "")
        if cleaned.startswith(("sh", "sz")):
            return cleaned[2:]
        return cleaned

    def _normalize_baostock_index_symbol(self, symbol: str) -> str:
        cleaned = symbol.strip().lower().replace(".", "")
        if cleaned.startswith(("sh", "sz")):
            return f"{cleaned[:2]}.{cleaned[2:]}"
        prefix = "sh" if cleaned.startswith(("000", "880", "899")) else "sz"
        return f"{prefix}.{cleaned}"

    def _normalize_tushare_index_symbol(self, symbol: str) -> str:
        cleaned = symbol.strip().lower().replace(".", "")
        if cleaned.startswith("sh"):
            return f"{cleaned[2:]}.SH"
        if cleaned.startswith("sz"):
            return f"{cleaned[2:]}.SZ"
        suffix = ".SH" if cleaned.startswith(("000", "880", "899")) else ".SZ"
        return f"{cleaned.upper()}{suffix}"

    def _get_tushare_client(self):
        token = self._get_tushare_token()
        if not token:
            raise RuntimeError("Tushare token is not configured")
        if ts is None:
            raise RuntimeError("tushare is not installed")
        if self._tushare_client is None:
            ts.set_token(token)
            self._tushare_client = ts.pro_api(token)
        return self._tushare_client

    def _normalize_tushare_history(self, history: pd.DataFrame, *, is_index: bool = False) -> pd.DataFrame:
        if history is None or history.empty:
            return pd.DataFrame()
        rename_map = {
            "trade_date": "date",
            "vol": "volume",
            "amount": "turnover",
            "pct_chg": "pct_change",
        }
        result = history.rename(columns=rename_map).copy()
        expected_columns = ["date", "open", "close", "high", "low", "volume", "turnover"]
        for column in expected_columns:
            if column not in result.columns:
                result[column] = pd.NA
        result["date"] = pd.to_datetime(result["date"], format="%Y%m%d", errors="coerce")
        for column in ["open", "close", "high", "low", "volume", "turnover", "pct_change"]:
            if column in result.columns:
                result[column] = pd.to_numeric(result[column], errors="coerce")
        if not is_index:
            if "turnover_rate" not in result.columns:
                result["turnover_rate"] = pd.NA
            result["turnover_rate"] = pd.to_numeric(result["turnover_rate"], errors="coerce")
            ordered_columns = ["date", "open", "close", "high", "low", "volume", "turnover", "pct_change", "turnover_rate"]
            return result[ordered_columns].dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        ordered_columns = ["date", "open", "close", "high", "low", "volume", "turnover"]
        return result[ordered_columns].dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    def _get_history_from_tushare(self, symbol: str, start_date: datetime, end_date: datetime, adjust: str) -> pd.DataFrame:
        client = self._get_tushare_client()
        ts_code = self._normalize_tushare_stock_symbol(symbol)
        if adjust in {"qfq", "hfq"}:
            history = self._with_retry(
                f"fetch tushare adjusted history for {ts_code}",
                lambda: ts.pro_bar(
                    ts_code=ts_code,
                    asset="E",
                    adj=adjust,
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                ),
            )
        else:
            history = self._with_retry(
                f"fetch tushare history for {ts_code}",
                lambda: client.daily(
                    ts_code=ts_code,
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                ),
            )
        return self._normalize_tushare_history(history, is_index=False)

    def _get_index_history_from_tushare(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        client = self._get_tushare_client()
        ts_code = self._normalize_tushare_index_symbol(symbol)
        history = self._with_retry(
            f"fetch tushare index history for {ts_code}",
            lambda: client.index_daily(
                ts_code=ts_code,
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
            ),
        )
        return self._normalize_tushare_history(history, is_index=True)

    def _baostock_adjust_flag(self, adjust: str) -> str:
        adjust_flags = {"hfq": "1", "qfq": "2", "": "3", None: "3", "none": "3"}
        return adjust_flags.get(adjust, "2")

    def _ensure_baostock_session(self) -> None:
        if self._baostock_logged_in:
            return

        login_result = self._with_retry("login to baostock", lambda: self._run_baostock_call(bs.login))
        if getattr(login_result, "error_code", "0") != "0":
            self._reset_baostock_session()
            raise RuntimeError(f"login to baostock failed: {login_result.error_msg}")
        self._baostock_logged_in = True

    def _get_history_from_baostock(self, symbol: str, start_date: datetime, end_date: datetime, adjust: str) -> pd.DataFrame:
        try:
            self._ensure_baostock_session()
            query_result = self._with_retry(
                f"fetch baostock history for {symbol}",
                lambda: self._run_baostock_call(
                    lambda: bs.query_history_k_data_plus(
                        self._normalize_baostock_symbol(symbol),
                        "date,open,high,low,close,volume,amount,turn",
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d"),
                        frequency="d",
                        adjustflag=self._baostock_adjust_flag(adjust),
                    )
                ),
            )
            if getattr(query_result, "error_code", "0") != "0":
                raise RuntimeError(f"fetch baostock history for {symbol} failed: {query_result.error_msg}")

            rows: list[list[str]] = []
            while query_result.next():
                rows.append(query_result.get_row_data())

            history = pd.DataFrame(rows, columns=query_result.fields)
            if history.empty:
                return history
            return self._normalize_history(history, BAOSTOCK_COLUMN_MAP)
        except Exception:
            self._reset_baostock_session()
            raise
        finally:
            self._reset_baostock_session()

    def _get_index_history_from_baostock(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            self._ensure_baostock_session()
            query_result = self._with_retry(
                f"fetch baostock index history for {symbol}",
                lambda: self._run_baostock_call(
                    lambda: bs.query_history_k_data_plus(
                        self._normalize_baostock_index_symbol(symbol),
                        "date,open,high,low,close,volume,amount",
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d"),
                        frequency="d",
                        adjustflag="3",
                    )
                ),
            )
            if getattr(query_result, "error_code", "0") != "0":
                raise RuntimeError(f"fetch baostock index history for {symbol} failed: {query_result.error_msg}")

            rows: list[list[str]] = []
            while query_result.next():
                rows.append(query_result.get_row_data())

            history = pd.DataFrame(rows, columns=query_result.fields)
            if history.empty:
                return history
            return self._normalize_history(history, INDEX_COLUMN_MAP)
        except Exception:
            self._reset_baostock_session()
            raise
        finally:
            self._reset_baostock_session()

    def _normalize_history(self, history: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
        history = history.rename(columns=column_map)
        history = history[list(column_map.values())].copy()
        history["date"] = pd.to_datetime(history["date"])
        for column in history.columns:
            if column != "date":
                history[column] = pd.to_numeric(history[column], errors="coerce")
        return history.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

    def get_universe_snapshot(self) -> pd.DataFrame:
        cache_path = self._snapshot_cache_path()
        active_cache_path = self._active_snapshot_cache_path()
        # Priority: efinance → cache → akshare_spot legacy fallback → akshare_em last-resort
        try:
            snapshot = self._get_snapshot_from_efinance()
            if not _has_usable_snapshot_rows(snapshot):
                raise RuntimeError("efinance snapshot returned no rows")
            self._set_fetch_detail("snapshot", "efinance")
        except Exception:
            if active_cache_path.exists():
                active_snapshot = self._load_cached_snapshot(active_cache_path)
                if _has_usable_snapshot_rows(active_snapshot) and _has_active_snapshot_activity(active_snapshot):
                    self._set_fetch_detail("snapshot", "cache", active_cache_path.name)
                    return active_snapshot
            if cache_path.exists():
                cached_snapshot = self._load_cached_snapshot(cache_path)
                if _has_usable_snapshot_rows(cached_snapshot) and _has_active_snapshot_activity(cached_snapshot):
                    self._set_fetch_detail("snapshot", "cache", cache_path.name)
                    return cached_snapshot
            try:
                snapshot = self._with_retry("fetch fallback universe snapshot", ak.stock_zh_a_spot)
                snapshot = self._normalize_snapshot(snapshot)
                if not _has_usable_snapshot_rows(snapshot):
                    raise RuntimeError("akshare legacy snapshot returned no rows")
                self._set_fetch_detail("snapshot", "akshare_spot")
            except Exception:
                snapshot = self._with_retry("fetch universe snapshot", ak.stock_zh_a_spot_em)
                snapshot = self._normalize_snapshot(snapshot)
                if not _has_usable_snapshot_rows(snapshot):
                    raise RuntimeError("akshare_em snapshot returned no rows")
                self._set_fetch_detail("snapshot", "akshare_em")

        if _has_active_snapshot_activity(snapshot):
            snapshot.to_csv(cache_path, index=False, encoding="utf-8-sig")
            snapshot.to_csv(active_cache_path, index=False, encoding="utf-8-sig")
        elif cache_path.exists():
            cached_snapshot = self._load_cached_snapshot(cache_path)
            if _has_usable_snapshot_rows(cached_snapshot):
                self._set_fetch_detail("snapshot", "cache", cache_path.name)
                return cached_snapshot
        return snapshot

    def get_symbol_snapshot(self, symbol: str) -> dict[str, object] | None:
        normalized_symbol = self._normalize_symbol(symbol)
        snapshot = self.get_universe_snapshot()
        if snapshot.empty:
            return None

        matched = snapshot.loc[snapshot["symbol"].astype(str).str.lower() == normalized_symbol]
        if matched.empty and normalized_symbol.startswith(("sh", "sz", "bj")):
            raw_symbol = normalized_symbol[2:]
            matched = snapshot.loc[snapshot["symbol"].astype(str).isin([raw_symbol, normalized_symbol])]
        if matched.empty:
            return None

        row = matched.iloc[0]
        return {
            "symbol": str(row["symbol"]),
            "name": str(row.get("name", "") or ""),
            "last_price": float(row["last_price"]),
            "turnover_amount": float(row["turnover_amount"]),
        }

    def get_index_history(self, symbol: str, start_date: datetime, end_date: datetime, use_cache: bool = True) -> pd.DataFrame:
        cache_path = self._index_cache_path(symbol, start_date, end_date)
        if use_cache and cache_path.exists():
            cached_history = pd.read_csv(cache_path, parse_dates=["date"])
            if not cached_history.empty and cached_history["date"].max() >= pd.Timestamp(end_date.date()):
                self._set_fetch_detail("index", "cache", cache_path.name)
                return cached_history
        elif use_cache:
            reusable_cache = self._find_reusable_index_cache(
                self._normalize_index_symbol(symbol),
                start_date,
                end_date,
            )
            if reusable_cache is not None:
                cached_history, cached_path = reusable_cache
                self._set_fetch_detail("index", "cache", cached_path.name)
                return cached_history

        normalized_symbol = self._normalize_index_symbol(symbol)
        # Priority: optional tushare -> akshare index endpoint -> baostock fallback -> cache
        try:
            try:
                if self._use_tushare_for_index():
                    history = self._get_index_history_from_tushare(normalized_symbol, start_date, end_date)
                    self._set_fetch_detail("index", "tushare")
                else:
                    raise RuntimeError("tushare index source is disabled by config")
            except Exception:
                history = self._with_retry(
                    f"fetch index history for {normalized_symbol}",
                    lambda: ak.index_zh_a_hist(
                        symbol=normalized_symbol,
                        period="daily",
                        start_date=start_date.strftime("%Y%m%d"),
                        end_date=end_date.strftime("%Y%m%d"),
                    ),
                )
                history = self._normalize_history(history, HISTORY_COLUMN_MAP)
                self._set_fetch_detail("index", "akshare_index")
        except Exception:
            try:
                history = self._get_index_history_from_baostock(normalized_symbol, start_date, end_date)
                self._set_fetch_detail("index", "baostock")
            except Exception:
                if cache_path.exists():
                    self._set_fetch_detail("index", "cache", cache_path.name)
                    return pd.read_csv(cache_path, parse_dates=["date"])
                raise RuntimeError(f"fetch benchmark index history for {normalized_symbol} failed")

        if history.empty:
            return history
        history.to_csv(cache_path, index=False, encoding="utf-8-sig")
        return history

    def get_history(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        adjust: str = "qfq",
        use_cache: bool = True,
        max_cache_trading_day_lag: int = 0,
    ) -> pd.DataFrame:
        normalized_symbol = self._normalize_symbol(symbol)
        cache_path = self.cache_dir / f"{normalized_symbol}_{start_date:%Y%m%d}_{end_date:%Y%m%d}_{adjust}.csv"
        if use_cache and cache_path.exists():
            cached_history = self._load_cached_history(cache_path)
            if not cached_history.empty and cached_history["date"].max() >= pd.Timestamp(end_date.date()):
                self._set_fetch_detail("history", "cache", cache_path.name)
                return cached_history
            if not cached_history.empty and max_cache_trading_day_lag > 0:
                trading_day_lag = self._count_trading_day_lag(cached_history["date"].max().date(), end_date.date())
                if trading_day_lag is not None and trading_day_lag <= max_cache_trading_day_lag:
                    self._set_fetch_detail("history", "cache", cache_path.name)
                    return cached_history
        elif use_cache and max_cache_trading_day_lag > 0:
            reusable_cache = self._find_reusable_history_cache(
                normalized_symbol,
                start_date,
                end_date,
                adjust,
                max_cache_trading_day_lag,
            )
            if reusable_cache is not None:
                self._set_fetch_detail("history", "cache", "reusable")
                return reusable_cache

        # Priority: tushare -> efinance -> akshare_daily -> akshare_hist -> baostock last-resort fallback
        try:
            try:
                history = self._get_history_from_tushare(normalized_symbol, start_date, end_date, adjust)
                self._set_fetch_detail("history", "tushare")
            except Exception:
                history = self._get_history_from_efinance(normalized_symbol, start_date, end_date, adjust)
                self._set_fetch_detail("history", "efinance")
        except Exception:
            try:
                history = self._with_retry(
                    f"fetch fallback history for {normalized_symbol}",
                    lambda: ak.stock_zh_a_daily(symbol=normalized_symbol, adjust=adjust),
                )
                history = self._normalize_history(history, DAILY_FALLBACK_COLUMN_MAP)
                history = history[
                    (history["date"] >= pd.Timestamp(start_date)) & (history["date"] <= pd.Timestamp(end_date))
                ].reset_index(drop=True)
                self._set_fetch_detail("history", "akshare_daily")
            except Exception:
                try:
                    history = self._with_retry(
                        f"fetch history for {normalized_symbol}",
                        lambda: ak.stock_zh_a_hist(
                            symbol=normalized_symbol,
                            period="daily",
                            start_date=start_date.strftime("%Y%m%d"),
                            end_date=end_date.strftime("%Y%m%d"),
                            adjust=adjust,
                        ),
                    )
                    history = self._normalize_history(history, HISTORY_COLUMN_MAP)
                    self._set_fetch_detail("history", "akshare_hist")
                except Exception:
                    history = self._get_history_from_baostock(normalized_symbol, start_date, end_date, adjust)
                    self._set_fetch_detail("history", "baostock")

        if history.empty:
            return history
        history.to_csv(cache_path, index=False, encoding="utf-8-sig")
        return history


def resolve_date_window(history_days: int, as_of: str | None) -> tuple[datetime, datetime]:
    end_date = datetime.strptime(as_of, "%Y-%m-%d") if as_of else datetime.now()
    start_date = end_date - timedelta(days=history_days * 2)
    return start_date, end_date
