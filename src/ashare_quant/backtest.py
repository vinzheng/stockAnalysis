from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from ashare_quant.strategy import get_breakout_chase_limit_pct, get_entry_weak_open_limit_pct, get_board_limit_ratio


@dataclass(slots=True)
class BacktestResult:
    symbol: str
    trades: int
    win_rate: float
    total_return: float
    max_drawdown: float
    avg_win_return: float
    avg_loss_return: float
    expectancy: float
    profit_factor: float
    execution_basis: str


def analyze_signal_statistics(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()

    signal_specs = [
        ("buy_signal", "买点信号", "long"),
        ("add_on_signal", "补仓信号", "long"),
        ("sell_signal", "卖点信号", "short"),
        ("breakout_failure_signal", "快退信号", "short"),
    ]
    horizons = (3, 5, 10)
    rows: list[dict[str, object]] = []

    for signal_column, label, direction in signal_specs:
        if signal_column not in history.columns:
            continue

        signal_rows = history.loc[history[signal_column]].copy()
        if signal_rows.empty:
            continue

        row: dict[str, object] = {"信号类型": label, "样本数": int(len(signal_rows))}
        next_open = history["open"].shift(-1)
        for horizon in horizons:
            future_close = history["close"].shift(-horizon)
            forward_return = (future_close / next_open) - 1
            valid_returns = forward_return.loc[history[signal_column] & forward_return.notna() & next_open.notna()]
            if valid_returns.empty:
                row[f"{horizon}日平均收益"] = None
                row[f"{horizon}日命中率"] = None
                continue

            row[f"{horizon}日平均收益"] = float(valid_returns.mean())
            if direction == "long":
                row[f"{horizon}日命中率"] = float((valid_returns > 0).mean())
            else:
                row[f"{horizon}日命中率"] = float((valid_returns < 0).mean())
        rows.append(row)

    return pd.DataFrame(rows)


def analyze_entry_slices(history: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    if history.empty or "buy_signal" not in history.columns:
        return pd.DataFrame()

    signal_mask = history["buy_signal"].fillna(False) | history.get("add_on_signal", pd.Series(False, index=history.index)).fillna(False)
    if not signal_mask.any():
        return pd.DataFrame()

    next_open = history["open"].shift(-1)
    future_close = history["close"].shift(-horizon)
    forward_return = (future_close / next_open) - 1

    sliced = history.loc[signal_mask, [column for column in ["entry_signal_type", "market_regime", "buy_signal", "add_on_signal"] if column in history.columns]].copy()
    sliced["signal_kind"] = sliced["buy_signal"].map(lambda value: "买点" if bool(value) else "补仓")
    sliced["entry_signal_type"] = sliced.get(
        "entry_signal_type",
        pd.Series("", index=sliced.index, dtype="object"),
    ).fillna("").replace("", "未标记")
    sliced["market_regime"] = sliced.get(
        "market_regime",
        pd.Series("未知", index=sliced.index, dtype="object"),
    ).fillna("未知")
    sliced["forward_return"] = forward_return.loc[sliced.index]
    sliced = sliced.loc[sliced["forward_return"].notna()].copy()
    if sliced.empty:
        return pd.DataFrame()

    grouped = (
        sliced.groupby(["signal_kind", "entry_signal_type", "market_regime"], dropna=False)["forward_return"]
        .agg([("样本数", "size"), (f"{horizon}日平均收益", "mean"), (f"{horizon}日命中率", lambda values: (values > 0).mean())])
        .reset_index()
        .sort_values(["样本数", f"{horizon}日平均收益"], ascending=[False, False])
    )
    return grouped


def analyze_exit_slices(history: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    if history.empty or "sell_signal" not in history.columns:
        return pd.DataFrame()

    signal_mask = history["sell_signal"].fillna(False)
    if not signal_mask.any():
        return pd.DataFrame()

    next_open = history["open"].shift(-1)
    future_close = history["close"].shift(-horizon)
    forward_return = (future_close / next_open) - 1

    sliced = history.loc[signal_mask, [column for column in ["sell_reason", "market_regime"] if column in history.columns]].copy()
    sliced["sell_reason"] = sliced.get(
        "sell_reason",
        pd.Series("", index=sliced.index, dtype="object"),
    ).fillna("").replace("", "未标记")
    sliced["market_regime"] = sliced.get(
        "market_regime",
        pd.Series("未知", index=sliced.index, dtype="object"),
    ).fillna("未知")
    sliced["forward_return"] = forward_return.loc[sliced.index]
    sliced = sliced.loc[sliced["forward_return"].notna()].copy()
    if sliced.empty:
        return pd.DataFrame()

    grouped = (
        sliced.groupby(["sell_reason", "market_regime"], dropna=False)["forward_return"]
        .agg([("样本数", "size"), (f"{horizon}日平均跌幅", "mean"), (f"{horizon}日有效率", lambda values: (values < 0).mean())])
        .reset_index()
        .sort_values(["样本数", f"{horizon}日平均跌幅"], ascending=[False, True])
    )
    return grouped


def run_single_symbol_backtest(history: pd.DataFrame, symbol: str, atr_stop_multiple: float) -> BacktestResult:
    del atr_stop_multiple
    if history.empty:
        return BacktestResult(
            symbol=symbol,
            trades=0,
            win_rate=0.0,
            total_return=0.0,
            max_drawdown=0.0,
            avg_win_return=0.0,
            avg_loss_return=0.0,
            expectancy=0.0,
            profit_factor=0.0,
            execution_basis="next_open",
        )

    # 计算板块涨跌幅限制和相关比例调整系数
    limit_ratio = get_board_limit_ratio(symbol)
    scale_factor = limit_ratio / 0.10

    cash = 1.0
    position = 0.0
    entry_price = math.nan
    cash_before_entry = math.nan
    trade_returns: list[float] = []
    equity_curve: list[float] = []
    pending_entry = False
    pending_exit = False
    pending_entry_type = ""
    pending_breakout_high = math.nan
    pending_signal_close = math.nan
    
    breakout_chase_limit_pct = 0.04 * scale_factor
    entry_weak_open_limit_pct = 0.005 * scale_factor
    if "breakout_chase_limit_pct" in history.attrs:
        breakout_chase_limit_pct = float(history.attrs["breakout_chase_limit_pct"])
    elif "config" in history.attrs:
        breakout_chase_limit_pct = get_breakout_chase_limit_pct(history.attrs["config"]) * scale_factor
    if "entry_weak_open_limit_pct" in history.attrs:
        entry_weak_open_limit_pct = float(history.attrs["entry_weak_open_limit_pct"])
    elif "config" in history.attrs:
        entry_weak_open_limit_pct = get_entry_weak_open_limit_pct(history.attrs["config"]) * scale_factor

    # 从配置信息中动态抽取交易摩擦与成本(滑点、佣金、印花税)
    strategy_cfg = {}
    if "config" in history.attrs and history.attrs["config"]:
        strategy_cfg = history.attrs["config"].get("strategy", {}) or {}
    elif hasattr(history, "attrs") and "strategy" in history.attrs:
        strategy_cfg = history.attrs["strategy"] or {}

    slippage_pct = float(strategy_cfg.get("slippage_pct", 0.001))        # 默认 0.1% 滑点成本
    commission_rate = float(strategy_cfg.get("commission_rate", 0.0003)) # 默认万分之三券商佣金
    stamp_duty_rate = float(strategy_cfg.get("stamp_duty_rate", 0.0005)) # 默认万分之五印花税 (卖方单边收取)

    prev_close = math.nan

    for row in history.itertuples(index=False):
        open_price = float(row.open)
        close = float(row.close)
        buy_signal = bool(row.buy_signal)
        sell_signal = bool(row.sell_signal)

        # 估算涨停和跌停边界价格（基于昨日收盘价）
        limit_up_price = math.nan
        limit_down_price = math.nan
        if not math.isnan(prev_close):
            limit_up_price = round(prev_close * (1 + limit_ratio), 2)
            limit_down_price = round(prev_close * (1 - limit_ratio), 2)

        if pending_exit and position > 0:
            # 跌停限制：如果当天开盘价锁死在跌停价或更低，则今日无法成交卖出，保留 pending_exit 继续持有
            if not math.isnan(limit_down_price) and open_price <= limit_down_price:
                pass
            else:
                sell_exec_price = open_price * (1 - slippage_pct)
                cash = position * sell_exec_price * (1 - commission_rate - stamp_duty_rate)
                trade_returns.append((cash / cash_before_entry) - 1 if not math.isnan(cash_before_entry) else (sell_exec_price / entry_price) - 1)
                position = 0.0
                entry_price = math.nan
                cash_before_entry = math.nan
                pending_exit = False

        if pending_entry and position == 0:
            # 涨停限制：如果当天开盘价锁死在涨停价或更高，则今日无法买入成交，保留 pending_entry 继续观望
            if not math.isnan(limit_up_price) and open_price >= limit_up_price:
                pass
            else:
                skip_breakout_entry = (
                    pending_entry_type == "突破买入"
                    and not math.isnan(pending_breakout_high)
                    and open_price > pending_breakout_high * (1 + breakout_chase_limit_pct)
                )
                skip_weak_open_entry = (
                    not math.isnan(pending_signal_close)
                    and (
                        open_price < pending_signal_close * (1 - entry_weak_open_limit_pct)
                        or (
                            pending_entry_type == "突破买入"
                            and not math.isnan(pending_breakout_high)
                            and open_price < pending_breakout_high * (1 - entry_weak_open_limit_pct)
                        )
                    )
                )
                pending_entry = False
                if not (skip_breakout_entry or skip_weak_open_entry):
                    cash_before_entry = cash
                    buy_exec_price = open_price * (1 + slippage_pct)
                    position = (cash / (1 + commission_rate)) / buy_exec_price
                    cash = 0.0
                    entry_price = buy_exec_price
                pending_entry_type = ""
                pending_breakout_high = math.nan
                pending_signal_close = math.nan

        if position > 0:
            if sell_signal:
                pending_exit = True
        elif buy_signal:
            pending_entry = True
            pending_entry_type = str(getattr(row, "entry_signal_type", "") or getattr(row, "raw_buy_signal_type", "") or "")
            breakout_high = getattr(row, "breakout_high", math.nan)
            pending_breakout_high = float(breakout_high) if pd.notna(breakout_high) else math.nan
            pending_signal_close = close

        equity = cash if position == 0 else position * close
        equity_curve.append(equity)
        prev_close = close

    if position > 0:
        final_close = float(history.iloc[-1]["close"])
        sell_exec_price = final_close * (1 - slippage_pct)
        cash = position * sell_exec_price * (1 - commission_rate - stamp_duty_rate)
        trade_returns.append((cash / cash_before_entry) - 1 if not math.isnan(cash_before_entry) else (sell_exec_price / entry_price) - 1)
        equity_curve[-1] = cash

    equity_series = pd.Series(equity_curve, dtype=float)
    rolling_peak = equity_series.cummax()
    drawdown = (equity_series / rolling_peak) - 1

    wins = sum(1 for value in trade_returns if value > 0)
    trades = len(trade_returns)
    win_rate = wins / trades if trades else 0.0
    total_return = cash - 1.0
    max_drawdown = abs(float(drawdown.min())) if not drawdown.empty else 0.0
    positive_returns = [value for value in trade_returns if value > 0]
    negative_returns = [value for value in trade_returns if value <= 0]
    avg_win_return = sum(positive_returns) / len(positive_returns) if positive_returns else 0.0
    avg_loss_return = sum(negative_returns) / len(negative_returns) if negative_returns else 0.0
    expectancy = sum(trade_returns) / trades if trades else 0.0
    gross_profit = sum(positive_returns)
    gross_loss = abs(sum(negative_returns))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    return BacktestResult(
        symbol=symbol,
        trades=trades,
        win_rate=win_rate,
        total_return=total_return,
        max_drawdown=max_drawdown,
        avg_win_return=avg_win_return,
        avg_loss_return=avg_loss_return,
        expectancy=expectancy,
        profit_factor=profit_factor,
        execution_basis="next_open",
    )
