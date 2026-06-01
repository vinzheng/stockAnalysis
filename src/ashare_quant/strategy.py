from __future__ import annotations

import numpy as np
import pandas as pd


def get_board_limit_ratio(symbol: str) -> float:
    """
    根据 A 股代码前缀识别板块，返回标准涨跌幅限制比率：
    - 科创板 (sh688... / 688...): 0.20
    - 创业板 (sz30... / 300... / 301...): 0.20
    - 北交所 (bj... / 43... / 83... / 87... / 88... / 92...): 0.30
    - 沪深主板: 0.10
    """
    symbol = str(symbol).lower().strip()
    code = "".join(filter(str.isdigit, symbol))
    if not code:
        return 0.10

    if "sh688" in symbol or code.startswith("688"):
        return 0.20
    if "sz300" in symbol or "sz301" in symbol or code.startswith(("300", "301")):
        return 0.20
    if "bj" in symbol or code.startswith(("43", "83", "87", "88", "92")):
        return 0.30
    return 0.10


def get_breakout_chase_limit_pct(config: dict | None) -> float:
    if not config:
        return 0.04
    prompt_cfg = config.get("intraday_prompt", {}) or {}
    return float(prompt_cfg.get("breakout_chase_limit_pct", 0.04))


def get_entry_weak_open_limit_pct(config: dict | None) -> float:
    if not config:
        return 0.005
    prompt_cfg = config.get("intraday_prompt", {}) or {}
    return float(prompt_cfg.get("entry_weak_open_limit_pct", 0.005))


def get_entry_score_thresholds(config: dict | None) -> tuple[float, float]:
    if not config:
        return 70.0, 65.0
    prompt_cfg = config.get("intraday_prompt", {}) or {}
    return float(prompt_cfg.get("trial_score_min", 70.0)), float(prompt_cfg.get("pullback_score_min", 65.0))


def compute_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(history: pd.DataFrame, window: int) -> pd.Series:
    previous_close = history["close"].shift(1)
    true_range = pd.concat(
        [
            history["high"] - history["low"],
            (history["high"] - previous_close).abs(),
            (history["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=window, min_periods=window).mean()


def enrich_indicators(history: pd.DataFrame, config: dict) -> pd.DataFrame:
    result = history.copy()
    strategy = config["strategy"]
    result["ma_fast"] = result["close"].rolling(strategy["ma_fast"]).mean()
    result["ma_mid"] = result["close"].rolling(strategy["ma_mid"]).mean()
    result["ma_slow"] = result["close"].rolling(strategy["ma_slow"]).mean()
    result["volume_ma"] = result["volume"].rolling(strategy["volume_window"]).mean()
    result["volume_ratio"] = result["volume"] / result["volume_ma"]
    result["breakout_high"] = result["high"].shift(1).rolling(strategy["breakout_window"]).max()
    result["rsi"] = compute_rsi(result["close"], strategy["rsi_window"])
    result["atr"] = compute_atr(result, strategy["atr_window"])
    result["momentum_20"] = result["close"].pct_change(20)
    result["distance_to_high"] = result["close"] / result["high"].rolling(120).max()
    result["trend_spread"] = (result["ma_fast"] / result["ma_mid"]) - 1
    return result


def apply_position_state_machine(history: pd.DataFrame, config: dict, symbol: str | None = None) -> pd.DataFrame:
    if symbol is None:
        symbol = history.attrs.get("symbol", "")
    limit_ratio = get_board_limit_ratio(symbol)
    scale_factor = limit_ratio / 0.10

    atr_stop_multiple = float(config["strategy"].get("atr_stop_multiple", 0))
    enable_trailing_stop = bool(config["strategy"].get("enable_trailing_stop", True))
    atr_profit_activation_multiple = float(config["strategy"].get("atr_profit_activation_multiple", 0))
    atr_profit_retracement_multiple = float(config["strategy"].get("atr_profit_retracement_multiple", 0))

    breakout_chase_limit_pct = get_breakout_chase_limit_pct(config) * scale_factor
    entry_weak_open_limit_pct = get_entry_weak_open_limit_pct(config) * scale_factor
    result = history.copy()

    effective_buy_signal: list[bool] = []
    effective_add_on_signal: list[bool] = []
    effective_sell_signal: list[bool] = []
    atr_stop_signal: list[bool] = []
    atr_stop_price: list[float] = []
    position_state: list[str] = []
    next_action: list[str] = []
    holding_days: list[float] = []
    position_active: list[bool] = []
    signal_context: list[str] = []
    signal_reason: list[str] = []
    entry_signal_type: list[str] = []

    in_position = False
    pending_entry = False
    pending_exit = False
    entry_price = np.nan
    entry_atr = np.nan
    entry_index: int | None = None
    active_entry_type = ""
    pending_entry_type = ""
    pending_breakout_high = np.nan
    pending_signal_close = np.nan

    # === Phase 3 dynamic trackers ===
    max_close_since_entry = np.nan
    profit_protect_activated = False

    prev_close = np.nan

    for index, row in enumerate(result.itertuples(index=False)):
        row_open = float(row.open)
        row_close = float(row.close)
        row_atr = float(row.atr) if pd.notna(row.atr) else np.nan

        # 估算涨停和跌停边界价格（基于昨日收盘价）
        limit_up_price = np.nan
        limit_down_price = np.nan
        if pd.notna(prev_close):
            limit_up_price = round(prev_close * (1 + limit_ratio), 2)
            limit_down_price = round(prev_close * (1 - limit_ratio), 2)

        if pending_exit and in_position:
            # 跌停限制：如果开盘价锁死在跌停板或更低，则今日无法完成卖出平仓，继续持有和等待
            if pd.notna(limit_down_price) and row_open <= limit_down_price:
                pass
            else:
                in_position = False
                pending_exit = False
                entry_price = np.nan
                entry_atr = np.nan
                entry_index = None
                active_entry_type = ""
                # === Reset Phase 3 trackers ===
                max_close_since_entry = np.nan
                profit_protect_activated = False

        if pending_entry and not in_position:
            # 涨停限制：如果开盘价锁死在涨停板或更高，则今日无法买入建仓，继续保持等待
            if pd.notna(limit_up_price) and row_open >= limit_up_price:
                pass
            else:
                skip_breakout_entry = (
                    pending_entry_type == "突破买入"
                    and pd.notna(pending_breakout_high)
                    and row_open > float(pending_breakout_high) * (1 + breakout_chase_limit_pct)
                )
                skip_weak_open_entry = (
                    pd.notna(pending_signal_close)
                    and (
                        row_open < float(pending_signal_close) * (1 - entry_weak_open_limit_pct)
                        or (
                            pending_entry_type == "突破买入"
                            and pd.notna(pending_breakout_high)
                            and row_open < float(pending_breakout_high) * (1 - entry_weak_open_limit_pct)
                        )
                    )
                )
                pending_entry = False
                if not (skip_breakout_entry or skip_weak_open_entry):
                    in_position = True
                    entry_price = row_open
                    entry_atr = row_atr
                    entry_index = index
                    active_entry_type = pending_entry_type or str(getattr(row, "raw_buy_signal_type", "") or "")
                    
                    # === Init Phase 3 trackers ===
                    max_close_since_entry = row_close
                    profit_protect_activated = False
                pending_entry_type = ""
                pending_breakout_high = np.nan
                pending_signal_close = np.nan

        # === Update peak close tracking when in position ===
        if in_position:
            if pd.isna(max_close_since_entry):
                max_close_since_entry = row_close
            else:
                max_close_since_entry = max(max_close_since_entry, row_close)

        current_stop_price = np.nan
        current_atr_stop_signal = False
        current_atr_profit_signal = False

        if in_position and pd.notna(entry_price) and pd.notna(entry_atr):
            # 1. 动态移动止损
            if atr_stop_multiple > 0:
                if enable_trailing_stop and pd.notna(max_close_since_entry):
                    current_stop_price = max_close_since_entry - (atr_stop_multiple * entry_atr)
                else:
                    current_stop_price = entry_price - (atr_stop_multiple * entry_atr)
                
                # 止损线只升不降
                if index > 0 and len(atr_stop_price) > 0:
                    prev_stop = atr_stop_price[-1]
                    if pd.notna(prev_stop) and pd.notna(current_stop_price):
                        current_stop_price = max(current_stop_price, prev_stop)
                
                if row_close <= current_stop_price:
                    current_atr_stop_signal = True

            # 2. ATR自适应保护 / 移动止盈
            if atr_profit_activation_multiple > 0 and atr_profit_retracement_multiple > 0:
                if not profit_protect_activated and row_close >= entry_price + (atr_profit_activation_multiple * entry_atr):
                    profit_protect_activated = True
                
                if profit_protect_activated and pd.notna(max_close_since_entry):
                    profit_protect_price = max_close_since_entry - (atr_profit_retracement_multiple * entry_atr)
                    if row_close <= profit_protect_price:
                        current_atr_profit_signal = True

        raw_buy_signal = bool(row.buy_signal)
        raw_buy_type = str(row.raw_buy_signal_type or "")
        raw_add_on_signal = bool(row.raw_add_on_signal)
        raw_sell_signal = bool(row.base_sell_signal)

        buy_signal = (not in_position) and raw_buy_signal
        sell_signal = in_position and (raw_sell_signal or current_atr_stop_signal or current_atr_profit_signal)
        add_on_signal = in_position and raw_add_on_signal and not sell_signal

        current_signal_reason = ""
        if sell_signal:
            if current_atr_stop_signal:
                current_signal_reason = "动态移动止损" if enable_trailing_stop else "固定ATR止损"
            elif current_atr_profit_signal:
                current_signal_reason = "ATR自适应止盈"
            else:
                current_signal_reason = str(row.base_sell_reason or "")

        if sell_signal:
            pending_exit = True
            current_position_state = "待卖出"
            current_next_action = "卖出"
            current_signal_context = "离场"
        elif add_on_signal:
            current_position_state = "持仓待加仓"
            current_next_action = "补仓"
            current_signal_context = "加仓"
        elif buy_signal:
            pending_entry = True
            pending_entry_type = raw_buy_type
            pending_breakout_high = float(row.breakout_high) if pd.notna(getattr(row, "breakout_high", np.nan)) else np.nan
            pending_signal_close = row_close
            current_position_state = "待买入"
            current_next_action = "买入"
            current_signal_context = "建仓"
        elif in_position:
            current_position_state = "持仓"
            current_next_action = "持仓"
            current_signal_context = "持有"
        else:
            current_position_state = "空仓"
            current_next_action = "观察"
            current_signal_context = "空仓观察"

        effective_buy_signal.append(buy_signal)
        effective_add_on_signal.append(add_on_signal)
        effective_sell_signal.append(sell_signal)
        atr_stop_signal.append(current_atr_stop_signal)
        atr_stop_price.append(current_stop_price)
        position_state.append(current_position_state)
        next_action.append(current_next_action)
        position_active.append(in_position)
        signal_context.append(current_signal_context)
        signal_reason.append(current_signal_reason)
        entry_signal_type.append(raw_buy_type if buy_signal else active_entry_type)
        holding_days.append(float(index - entry_index) if in_position and entry_index is not None else np.nan)

        prev_close = row_close

    result["buy_signal"] = pd.Series(effective_buy_signal, index=result.index, dtype=bool)
    result["add_on_signal"] = pd.Series(effective_add_on_signal, index=result.index, dtype=bool)
    result["sell_signal"] = pd.Series(effective_sell_signal, index=result.index, dtype=bool)
    result["atr_stop_signal"] = pd.Series(atr_stop_signal, index=result.index, dtype=bool)
    result["atr_stop_price"] = pd.Series(atr_stop_price, index=result.index, dtype=float)
    result["position_state"] = pd.Series(position_state, index=result.index, dtype="object")
    result["next_action"] = pd.Series(next_action, index=result.index, dtype="object")
    result["position_active"] = pd.Series(position_active, index=result.index, dtype=bool)
    result["signal_context"] = pd.Series(signal_context, index=result.index, dtype="object")
    result["sell_reason"] = pd.Series(signal_reason, index=result.index, dtype="object")
    result["entry_signal_type"] = pd.Series(entry_signal_type, index=result.index, dtype="object")
    result["holding_days"] = pd.Series(holding_days, index=result.index, dtype=float)
    return result


def add_signal_columns(history: pd.DataFrame, config: dict, symbol: str | None = None) -> pd.DataFrame:
    if symbol is None:
        symbol = history.attrs.get("symbol", "")
    strategy = config["strategy"]
    result = enrich_indicators(history, config)
    breakout_score_min, pullback_score_min = get_entry_score_thresholds(config)
    use_rsi_filter = bool(strategy.get("use_rsi_filter", False))
    rsi_buy_filter = result["rsi"].between(strategy["rsi_min"], strategy["rsi_max"]) if use_rsi_filter else True
    rsi_add_on_filter = (
        result["rsi"].between(max(strategy["rsi_min"] - 2, 50), strategy["rsi_max"]) if use_rsi_filter else True
    )
    base_score = (
        result["momentum_20"].fillna(0) * 40
        + (result["volume_ratio"].fillna(0).clip(0, 3) * 20)
        + ((result["distance_to_high"].fillna(0) - 0.85).clip(lower=0) * 100)
        + (result["trend_spread"].fillna(0).clip(lower=0) * 400)
    )
    breakout_entry_signal = (
        (result["close"] > result["ma_fast"])
        & (result["ma_fast"] > result["ma_mid"])
        & (result["ma_mid"] > result["ma_slow"])
        & (result["close"] >= result["breakout_high"])
        & (result["volume_ratio"] >= strategy["volume_ratio_min"])
        & (base_score >= breakout_score_min)
        & rsi_buy_filter
    )
    recent_breakout_setup = breakout_entry_signal.shift(1).rolling(8, min_periods=1).max().fillna(False)
    pullback_entry_signal = (
        recent_breakout_setup.astype(bool)
        & (result["close"] > result["ma_fast"])
        & (result["ma_fast"] > result["ma_mid"])
        & (result["ma_mid"] > result["ma_slow"])
        & (result["low"] <= result["ma_fast"] * 1.01)
        & (result["close"] >= result["ma_fast"])
        & (result["close"] >= result["breakout_high"] * 0.98)
        & (result["close"] < result["breakout_high"])
        & (result["volume_ratio"] >= 1.0)
        & (base_score >= pullback_score_min)
        & rsi_add_on_filter
        & ~breakout_entry_signal
    )
    result["breakout_entry_signal"] = breakout_entry_signal
    result["pullback_entry_signal"] = pullback_entry_signal
    result["buy_signal"] = breakout_entry_signal | pullback_entry_signal
    result["raw_buy_signal_type"] = np.select(
        [breakout_entry_signal, pullback_entry_signal],
        ["突破买入", "回踩买入"],
        default="",
    )
    result["trend_sell_signal"] = (
        (result["close"] < result["ma_fast"])
        | (result["ma_fast"] < result["ma_mid"])
    )
    recent_breakout_context = breakout_entry_signal.shift(1).rolling(3, min_periods=1).max().fillna(False)
    result["breakout_failure_signal"] = (
        recent_breakout_context.astype(bool)
        & (result["close"] < result["breakout_high"])
        & ((result["close"] < result["open"]) | (result["close"] < result["close"].shift(1)))
        & ~result["buy_signal"]
    )
    result["base_sell_signal"] = result["trend_sell_signal"] | result["breakout_failure_signal"]
    result["base_sell_reason"] = np.select(
        [
            result["breakout_failure_signal"],
            result["close"] < result["ma_fast"],
            result["ma_fast"] < result["ma_mid"],
        ],
        ["突破失败快退", "跌破MA20", "趋势走坏"],
        default="",
    )
    recent_buy_context = result["buy_signal"].shift(1).rolling(5, min_periods=1).max().fillna(False)
    result["raw_add_on_signal"] = (
        recent_buy_context.astype(bool)
        & (result["close"] > result["ma_fast"])
        & (result["ma_fast"] > result["ma_mid"])
        & (result["ma_mid"] > result["ma_slow"])
        & (result["low"] <= result["ma_fast"] * 1.02)
        & (result["close"] >= result["ma_fast"])
        & (result["close"] < result["breakout_high"])
        & (result["close"] >= result["breakout_high"] * 0.97)
        & (result["volume_ratio"] >= 1.0)
        & rsi_add_on_filter
        & ~result["buy_signal"]
        & ~result["base_sell_signal"]
    )
    result["score"] = base_score
    final_result = apply_position_state_machine(result, config, symbol=symbol)
    
    limit_ratio = get_board_limit_ratio(symbol)
    scale_factor = limit_ratio / 0.10
    final_result.attrs["breakout_chase_limit_pct"] = get_breakout_chase_limit_pct(config) * scale_factor
    final_result.attrs["entry_weak_open_limit_pct"] = get_entry_weak_open_limit_pct(config) * scale_factor
    final_result.attrs["symbol"] = symbol
    return final_result


def _latest_action_row(history: pd.DataFrame) -> tuple[pd.Series | None, str]:
    latest = history.iloc[-1]
    action = str(latest.get("next_action", "观察"))
    if action in {"买入", "补仓", "卖出", "持仓"}:
        return latest, action
    return None, "观察"


def latest_signal_summary(history: pd.DataFrame, config: dict | None = None) -> dict[str, object] | None:
    if history.empty:
        return None
    latest = history.iloc[-1]
    action_row, action = _latest_action_row(history)
    signal_age = None
    holding_days = int(latest["holding_days"]) if pd.notna(latest.get("holding_days")) else None
    position_state = str(latest.get("position_state", "空仓"))
    atr_stop_price = round(float(latest["atr_stop_price"]), 2) if pd.notna(latest.get("atr_stop_price")) else None
    atr_stop_signal = bool(latest.get("atr_stop_signal", False))
    entry_signal_type = str(latest.get("entry_signal_type", "") or "")
    execution_advice = "当前没有新的执行信号，继续观察。"
    sell_reason = ""
    breakout_chase_limit_pct = float(history.attrs.get("breakout_chase_limit_pct", get_breakout_chase_limit_pct(config)))
    entry_weak_open_limit_pct = float(history.attrs.get("entry_weak_open_limit_pct", get_entry_weak_open_limit_pct(config)))
    if action_row is not None:
        if action in {"买入", "补仓", "卖出"}:
            signal_age = int(len(history) - history.index.get_loc(action_row.name) - 1)
        sell_reason = str(latest.get("sell_reason", "") or "") if action == "卖出" else ""
        if action == "买入":
            chase_suffix = ""
            if entry_signal_type == "突破买入":
                chase_suffix = (
                    f" 若次日开盘高于突破位约 {breakout_chase_limit_pct:.1%} 以上，或重新低于突破位，不建议机械追价。"
                )
            execution_advice = (
                f"今天收盘确认{entry_signal_type or '买点'}，默认按下一交易日开盘再执行。"
                f" 若次日开盘低于当日收盘约 {entry_weak_open_limit_pct:.1%} 以上，建议取消机械买入。{chase_suffix}"
                if signal_age == 0
                else f"最近一次{entry_signal_type or '买点'}已过去 {signal_age} 个交易日，不适合机械追价，优先等待重新突破或回踩确认。"
            )
        elif action == "补仓":
            execution_advice = (
                "今天收盘出现补仓点，默认按下一交易日开盘再执行。"
                if signal_age == 0
                else f"最近一次补仓点已过去 {signal_age} 个交易日，如未执行，优先等待新的回踩确认。"
            )
        elif action == "卖出":
            execution_advice = (
                f"今天收盘出现卖点（{sell_reason or '触发离场条件'}），默认按下一交易日开盘执行离场。"
                if signal_age == 0
                else f"最近一次卖点已过去 {signal_age} 个交易日（{sell_reason or '触发离场条件'}），如果仍持有，需要先检查是否已经偏离原始风控。"
            )
        elif action == "持仓":
            execution_advice = "当前按系统仍处于持仓状态，继续持有并跟踪风控价。"
            if holding_days is not None:
                execution_advice = f"当前按系统仍处于持仓状态，已持有 {holding_days} 个交易日，继续持有并跟踪风控价。"
            if atr_stop_price is not None:
                execution_advice += f" 当前 ATR 风控价约为 {atr_stop_price:.2f}。"

    return {
        "date": latest["date"].strftime("%Y-%m-%d"),
        "close": round(float(latest["close"]), 2),
        "buy_signal": bool(latest["buy_signal"]),
        "add_on_signal": bool(latest["add_on_signal"]),
        "sell_signal": bool(latest["sell_signal"]),
        "sell_reason": sell_reason,
        "action": action,
        "signal_age": signal_age,
        "holding_days": holding_days,
        "position_state": position_state,
        "atr_stop_price": atr_stop_price,
        "atr_stop_signal": atr_stop_signal,
        "entry_signal_type": entry_signal_type,
        "execution_advice": execution_advice,
        "score": round(float(latest["score"]), 2),
        "volume_ratio": round(float(latest["volume_ratio"]), 2) if pd.notna(latest["volume_ratio"]) else None,
        "rsi": round(float(latest["rsi"]), 2) if pd.notna(latest["rsi"]) else None,
    }
