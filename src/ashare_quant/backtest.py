from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from ashare_quant.strategy import get_breakout_chase_limit_pct


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

    cash = 1.0
    position = 0.0
    entry_price = math.nan
    trade_returns: list[float] = []
    equity_curve: list[float] = []
    pending_entry = False
    pending_exit = False
    pending_entry_type = ""
    pending_breakout_high = math.nan
    breakout_chase_limit_pct = 0.04
    if "breakout_chase_limit_pct" in history.attrs:
        breakout_chase_limit_pct = float(history.attrs["breakout_chase_limit_pct"])

    for row in history.itertuples(index=False):
        open_price = float(row.open)
        close = float(row.close)
        buy_signal = bool(row.buy_signal)
        sell_signal = bool(row.sell_signal)

        if pending_exit and position > 0:
            cash = position * open_price
            trade_returns.append((open_price / entry_price) - 1)
            position = 0.0
            entry_price = math.nan
            pending_exit = False

        if pending_entry and position == 0:
            skip_breakout_entry = (
                pending_entry_type == "突破买入"
                and not math.isnan(pending_breakout_high)
                and open_price > pending_breakout_high * (1 + breakout_chase_limit_pct)
            )
            pending_entry = False
            if not skip_breakout_entry:
                position = cash / open_price
                cash = 0.0
                entry_price = open_price
            pending_entry_type = ""
            pending_breakout_high = math.nan

        if position > 0:
            if sell_signal:
                pending_exit = True
        elif buy_signal:
            pending_entry = True
            pending_entry_type = str(getattr(row, "entry_signal_type", "") or getattr(row, "raw_buy_signal_type", "") or "")
            breakout_high = getattr(row, "breakout_high", math.nan)
            pending_breakout_high = float(breakout_high) if pd.notna(breakout_high) else math.nan

        equity = cash if position == 0 else position * close
        equity_curve.append(equity)

    if position > 0:
        final_close = float(history.iloc[-1]["close"])
        cash = position * final_close
        trade_returns.append((final_close / entry_price) - 1)
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
