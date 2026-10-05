"""Own backtest engine: weekly rebalance, decisions from data ≤ t-1, execution at the open of t,
costs in basis points of traded notional, multi-ticker, fractional shares.

Positions are signed: a negative target weight is a short (sale proceeds go to cash). Flipping long <-> short
is executed as a close followed by an open. Shorts pay a borrow fee of `borrow_bps` per year on their market
value, accrued every session. Interest on cash and short proceeds is ignored (as for long positions).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from jevbt.features.technical import compute_technical
from jevbt.strategy import Strategy


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    round_trip_pnl: list[float] = field(default_factory=list)


def rebalance_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First session of each calendar week."""
    weeks = calendar.to_period("W")
    return calendar[~weeks.duplicated()]


def apply_gross_cap(decisions: list[list], held: float, max_gross: float) -> None:
    """Scale new entries (and flips) so that gross exposure stays ≤ max_gross; shared with paper trading.

    `decisions` rows are [ticker, price, current_weight, target_weight, record] (mutated in place); `held` is the
    gross weight of open positions that are not in `decisions`. Held/closing positions are never trimmed."""
    is_entry = [dec[3] != 0 and dec[3] * dec[2] <= 0 for dec in decisions]
    committed = held + sum(abs(dec[3]) for dec, e in zip(decisions, is_entry) if not e)
    wanted = sum(abs(dec[3]) for dec, e in zip(decisions, is_entry) if e)
    room = max(0.0, max_gross - committed)
    if wanted > room:
        for dec, e in zip(decisions, is_entry):
            if e:
                dec[3] *= room / wanted
                dec[4] = {**dec[4], "scaled_by_gross_cap": room / wanted}


def _json_default(o):
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def run_backtest(
    prices: dict[str, pd.DataFrame],
    strategy: Strategy,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    initial_cash: float = 100_000.0,
    cost_bps: float = 10.0,
    log_path: Path | None = None,
    liquidate_at_end: bool = False,
    borrow_bps: float = 30.0,
    trailing_stop: float | None = None,
    take_profit: float | None = None,
    trailing_stop_atr: float | None = None,
    stop_rearm: bool = False,
    stop_cooldown_weeks: int = 0,
    max_gross: float | None = None,
    rebalance: str = "weekly",
) -> BacktestResult:
    """`liquidate_at_end` closes every open position at the last session's close (with costs),
    so a window's result and round trips are complete (used by walk-forward test windows).

    Exits checked every session, between the weekly rebalances (both optional, fractions such as 0.03):
      trailing_stop — long: exit when the low touches (1 − x) × the highest high since entry;
                      short: cover when the high touches (1 + x) × the lowest low since entry.
      take_profit   — long: exit when the high touches (1 + x) × entry price; short: (1 − x) × entry.
    Fills at the level, or at the open when the session gaps through it. If both trigger in one session the
    stop is assumed first (conservative). The trailing extreme is updated after the check with the session's
    high/low, so a level never uses later prices of its own session.
      trailing_stop_atr — volatility-adjusted trailing stop: distance = x × ATR14/price of the previous session
                          (e.g. 3 → 3 ATRs). The level only tightens (it never moves back when volatility rises).
                          If both trailing stops are given, the tighter level applies.
    Re-entry after a stop (same direction only; the opposite side is never blocked):
      stop_rearm          — wait until the strategy's target has left that direction at a rebalance (fresh signal).
      stop_cooldown_weeks — wait at least this many weeks.
    Without either, the strategy may re-enter at the next rebalance.

    A strategy may choose the exits of each new position: if it has `position_exits(t, sign)`, the keys of the
    dict it returns (names of the five exit arguments above) override those arguments for that position;
    None or a missing key keeps the argument (strategy.NO_EXITS disables every exit). A strategy with
    `release_stop_block(t, stopped_at)` can lift a re-entry block early (e.g. after a regime change).

    max_gross — cap on gross exposure (sum of |weights|, e.g. 1.0 = no leverage) at each rebalance: if held
    positions plus new entries would exceed it, the new entries (and flips) are scaled down proportionally.
    Held positions are never trimmed (no resizing), so exposure can drift above the cap with price moves.

    rebalance — "weekly" (first session of each week, the default) or "daily" (every session). Either way decisions
    use data up to the previous session and orders fill at the session's open."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    features = {tk: compute_technical(df) for tk, df in prices.items()}
    calendar = pd.DatetimeIndex(sorted(set().union(*(df.index for df in prices.values()))))
    calendar = calendar[(calendar >= start) & (calendar <= end)]
    if calendar.empty:
        raise ValueError("no sessions in the backtest range")
    closes = pd.DataFrame({tk: df["close"] for tk, df in prices.items()}).sort_index().ffill()
    if rebalance not in ("weekly", "daily"):
        raise ValueError(f"rebalance must be weekly or daily, not {rebalance!r}")
    rebalances = set(calendar) if rebalance == "daily" else set(rebalance_dates(calendar))
    # Row position of each session per ticker: the previous row (data ≤ t-1) is an O(1) lookup, which keeps
    # daily rebalancing fast (same result as features_asof).
    positions_of = {tk: {d: i for i, d in enumerate(f.index)} for tk, f in features.items()}
    global_exits = {"trailing_stop": trailing_stop, "take_profit": take_profit, "trailing_stop_atr": trailing_stop_atr,
                    "stop_rearm": stop_rearm, "stop_cooldown_weeks": stop_cooldown_weeks}
    exit_hook = getattr(strategy, "position_exits", None)
    release_hook = getattr(strategy, "release_stop_block", None)  # (t, stopped_at) -> True lifts a re-entry block
    use_exits = bool(trailing_stop or take_profit or trailing_stop_atr or exit_hook)
    # Per-session bars as plain dicts: the exit checks run every session for every ticker.
    ohlc = {tk: dict(zip(df.index, df[["open", "high", "low"]].itertuples(index=False, name=None)))
            for tk, df in prices.items()} if use_exits else {}
    prev_atr = ({tk: features[tk]["atr_pct"].shift(1).to_dict() for tk in prices}
                if trailing_stop_atr or exit_hook else {})

    cash = initial_cash
    shares = {tk: 0.0 for tk in prices}
    basis = {tk: 0.0 for tk in prices}     # signed cash paid to open (negative for shorts: proceeds net of costs)
    realized = {tk: 0.0 for tk in prices}  # P&L of partial reductions, booked into the round trip when it closes
    entry = {tk: np.nan for tk in prices}    # entry price of the open position
    extreme = {tk: np.nan for tk in prices}  # highest high (long) / lowest low (short) since entry
    stop_level = {tk: np.nan for tk in prices}  # current trailing stop level (ratchets)
    pos_exits = {tk: global_exits for tk in prices}  # exit settings of the open position
    blocked: dict[str, dict] = {}  # ticker -> {"sign", "until", "reset"}: re-entry block after a stop
    trades, round_trips, equity = [], [], {}
    log = log_path.open("w", encoding="utf-8") if log_path else None

    def execute(d, tk, delta, price, reason="signal") -> dict:
        """Trade `delta` shares at `price` without crossing zero; books costs, basis and round trips."""
        nonlocal cash
        notional = delta * price
        cost = abs(notional) * cost_bps / 1e4
        cash -= notional + cost
        if shares[tk] == 0:
            entry[tk] = extreme[tk] = price
            stop_level[tk] = np.nan
            chosen = exit_hook(d, 1 if delta > 0 else -1) if exit_hook else None
            pos_exits[tk] = {**global_exits, **chosen} if chosen is not None else global_exits
        if shares[tk] == 0 or (delta > 0) == (shares[tk] > 0):   # open or add
            basis[tk] += notional + cost
        else:                                                    # reduce or close
            released = basis[tk] * (-delta / shares[tk])
            basis[tk] -= released
            realized[tk] += -(notional + cost) - released
        shares[tk] += delta
        if abs(shares[tk]) < 1e-9:
            shares[tk], basis[tk] = 0.0, 0.0
            round_trips.append(realized[tk])
            realized[tk] = 0.0
        order = {"side": "buy" if delta > 0 else "sell", "shares": delta, "price": price, "cost": cost,
                 "reason": reason}
        trades.append({"date": d, "ticker": tk, **order})
        return order

    def check_exits(d, tk, bar) -> None:
        """Trailing stops / take-profit for an open position during session d."""
        o, h, lo = bar
        long_ = shares[tk] > 0
        sign = 1 if long_ else -1
        ex = pos_exits[tk]
        dists = []
        if ex["trailing_stop"]:
            dists.append(ex["trailing_stop"])
        if ex["trailing_stop_atr"]:
            atr = prev_atr[tk].get(d, np.nan)
            if np.isfinite(atr):
                dists.append(ex["trailing_stop_atr"] * atr)
        stop = tp = None
        if dists:
            candidate = extreme[tk] * (1 - sign * min(dists))
            prev = stop_level[tk]
            stop = candidate if not np.isfinite(prev) else (max(prev, candidate) if long_ else min(prev, candidate))
            stop_level[tk] = stop
        if ex["take_profit"]:
            tp = entry[tk] * (1 + sign * ex["take_profit"])
        hit_stop = stop is not None and (lo <= stop if long_ else h >= stop)
        hit_tp = tp is not None and (h >= tp if long_ else lo <= tp)
        if hit_stop:
            gap = o <= stop if long_ else o >= stop
            execute(d, tk, -shares[tk], o if gap else stop, "stop")
            if ex["stop_rearm"] or ex["stop_cooldown_weeks"]:
                blocked[tk] = {"sign": sign, "until": d + pd.Timedelta(weeks=ex["stop_cooldown_weeks"]),
                               "reset": not ex["stop_rearm"], "at": d}
        elif hit_tp:
            gap = o >= tp if long_ else o <= tp
            execute(d, tk, -shares[tk], o if gap else tp, "take_profit")
        else:
            extreme[tk] = max(extreme[tk], h) if long_ else min(extreme[tk], lo)

    def apply_block(d, tk, target_w) -> tuple[float, str | None]:
        """Re-entry block after a stop: returns the allowed target and a note for the decision log."""
        b = blocked.get(tk)
        if b is None:
            return target_w, None
        if release_hook and release_hook(d, b["at"]):
            del blocked[tk]
            return target_w, None
        same_side = target_w * b["sign"] > 0
        if not same_side:
            b["reset"] = True  # the signal left the stopped direction: fresh signal from now on
        if b["reset"] and d >= b["until"]:
            del blocked[tk]
            return target_w, None
        if same_side:
            return 0.0, "re-entry blocked after stop"
        return target_w, None

    try:
        for d in calendar:
            if d in rebalances:
                # Mark at today's open where available, else yesterday's close.
                marks = {}
                for tk, df in prices.items():
                    if d in df.index:
                        marks[tk] = df.at[d, "open"]
                    else:
                        prev = closes.loc[:d].iloc[:-1][tk].dropna()
                        marks[tk] = prev.iloc[-1] if len(prev) else np.nan
                eq_open = cash + sum(shares[tk] * marks[tk] for tk in prices if shares[tk])
                # 1) Decide every ticker from the same opening equity.
                decisions = []
                for tk, df in prices.items():
                    if d not in df.index:
                        continue
                    pos = positions_of[tk][d]
                    if pos == 0:
                        continue  # no data before t
                    technical = features[tk].iloc[pos - 1]
                    price = marks[tk]
                    current_w = shares[tk] * price / eq_open if eq_open > 0 else 0.0
                    target_w, record = strategy.target(tk, d, technical, current_w)
                    if current_w == 0:
                        target_w, note = apply_block(d, tk, target_w)
                        if note:
                            record = {**record, "blocked": note}
                    decisions.append([tk, price, current_w, target_w, record])
                # 2) Gross exposure cap: new entries (and flips) share what the held positions leave free.
                if max_gross is not None and eq_open > 0:
                    decided = {dec[0] for dec in decisions}
                    held = sum(abs(shares[tk] * marks[tk]) / eq_open for tk in prices if shares[tk] and tk not in decided)
                    apply_gross_cap(decisions, held, max_gross)
                # 3) Execute.
                for tk, price, current_w, target_w, record in decisions:
                    order = None
                    if target_w != current_w:
                        target_shares = target_w * eq_open / price
                        if shares[tk] and target_shares * shares[tk] <= 0:   # close (then open if flipping)
                            order = execute(d, tk, -shares[tk], price)
                        if abs(target_shares - shares[tk]) > 1e-9:
                            order = execute(d, tk, target_shares - shares[tk], price)
                    if log:
                        log.write(json.dumps({"date": d, "ticker": tk, "strategy": strategy.name,
                                              "current_weight": current_w, "target_weight": target_w,
                                              "order": order, **record}, default=_json_default) + "\n")
            if use_exits:
                for tk in prices:
                    if shares[tk] and d in ohlc[tk]:
                        check_exits(d, tk, ohlc[tk][d])
            row = closes.loc[d]
            cash -= sum(-shares[tk] * row[tk] for tk in prices if shares[tk] < 0) * borrow_bps / 1e4 / 252
            if liquidate_at_end and d == calendar[-1]:
                for tk in prices:
                    if shares[tk]:
                        execute(d, tk, -shares[tk], row[tk], "liquidation")
            equity[d] = cash + sum(shares[tk] * row[tk] for tk in prices if shares[tk])
    finally:
        if log:
            log.close()

    return BacktestResult(
        equity=pd.Series(equity, name="equity"),
        trades=pd.DataFrame(trades, columns=["date", "ticker", "side", "shares", "price", "cost", "reason"]),
        round_trip_pnl=round_trips,
    )


def buy_and_hold(prices: dict[str, pd.DataFrame], start, end, initial_cash: float = 100_000.0,
                 cost_bps: float = 10.0) -> pd.Series:
    """Equal-weight buy & hold benchmark, bought at the first session's open."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    closes = pd.DataFrame({tk: df["close"] for tk, df in prices.items()}).sort_index().ffill()
    closes = closes.loc[start:end]
    first = closes.index[0]
    alloc = initial_cash * (1 - cost_bps / 1e4) / len(prices)
    units = {tk: alloc / prices[tk].at[first, "open"] for tk in prices if first in prices[tk].index}
    cash = initial_cash - alloc * len(units) / (1 - cost_bps / 1e4)
    return (closes[list(units)] * pd.Series(units)).sum(axis=1).rename("equity") + cash
