"""Walk-forward validation of the Jev rules (CLAUDE.md §8: avoid overfitting the thresholds).

Rolling folds: pick the rules with the best Sharpe on a train window, then apply them unchanged to the
following test window. Only the test windows are reported (out-of-sample). Each test window starts flat
with the equity carried over from the previous one (open positions are sold at the close of the
window's last session, with costs); the baseline is chained over the same windows.

Jev answers do not depend on the rules, so the grid search re-uses the memoized decisions of
`JevStrategy`: each (ticker, week) costs at most one Jev call for the whole walk-forward.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field

import pandas as pd

from jevbt.backtest.engine import run_backtest
from jevbt.backtest.metrics import summarize
from jevbt.strategy import BaselineStrategy, JevRules, JevStrategy


GRID: dict[str, list] = {
    "entry_mode": ["action", "signals"],
    "min_trend_up": [0.60, 0.75],
    "max_overbought": [0.40, 0.60],
    "min_quality": [1.0, 1.5],
    "exit_trend_up": [0.30, 0.50],
}


def default_grid(**fixed) -> list[JevRules]:
    """32 rule sets; `fixed` sets JevRules fields for all of them (e.g. use_overbought=False, min_quality=0.0),
    replacing that grid dimension if it is one. Duplicates are dropped (max_overbought is irrelevant without
    the overbought gate)."""
    dims = {k: [fixed[k]] if k in fixed else v for k, v in GRID.items()}
    if not fixed.get("use_overbought", True):
        dims["max_overbought"] = [1.0]
    rest = {k: v for k, v in fixed.items() if k not in GRID}
    grid = [JevRules(**dict(zip(dims, combo)), **rest) for combo in itertools.product(*dims.values())]
    return list(dict.fromkeys(grid))


def make_folds(start, end, train_months: int, test_months: int) -> list[tuple[pd.Timestamp, ...]]:
    """(train_start, train_end, test_start, test_end), rolling by `test_months`."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    folds = []
    test_start = start + pd.DateOffset(months=train_months)
    while test_start <= end:
        test_end = min(test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1), end)
        train_start = test_start - pd.DateOffset(months=train_months)
        folds.append((train_start, test_start - pd.Timedelta(days=1), test_start, test_end))
        test_start = test_start + pd.DateOffset(months=test_months)
    return folds


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    rules: JevRules
    train_sharpe: float
    train_trades: int
    test: dict


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    equity: pd.Series
    baseline_equity: pd.Series
    trades: pd.DataFrame
    round_trip_pnl: list[float] = field(default_factory=list)

    def folds_table(self) -> pd.DataFrame:
        rows = []
        for f in self.folds:
            rows.append({"test": f"{f.test_start:%Y-%m-%d}..{f.test_end:%Y-%m-%d}",
                         **{k: v for k, v in asdict(f.rules).items()
                            if k not in ("min_confidence", "valuation_penalty", "use_overbought", "sell_veto", "exit_on_sell")},
                         "train_sharpe": f.train_sharpe, "train_trades": f.train_trades, "test_return": f.test["total_return"],
                         "test_trades": f.test["trades"]})
        return pd.DataFrame(rows)


def _rank(metrics: dict, min_trades: int) -> tuple:
    """Rule sets with at least `min_trades` train trades always rank above the others; among them the best
    Sharpe wins. A high Sharpe from one or two lucky trades is not evidence (the rules then sit flat
    out-of-sample). If no rule set reaches the minimum, the most active one wins."""
    s = metrics["sharpe"]
    sharpe = -math.inf if s is None or math.isnan(s) or metrics["trades"] == 0 else s
    eligible = metrics["trades"] >= min_trades
    return (eligible, sharpe) if eligible else (False, metrics["trades"], sharpe)


def walk_forward(prices: dict[str, pd.DataFrame], strategy: JevStrategy, start, end, grid: list[JevRules] | None = None,
                 train_months: int = 12, test_months: int = 3, initial_cash: float = 100_000.0,
                 cost_bps: float = 10.0, min_trades: int | None = None) -> WalkForwardResult:
    """`min_trades` in each train window; default 2 per ticker (≈ one round trip per ticker)."""
    grid = grid or default_grid()
    min_trades = 2 * len(prices) if min_trades is None else min_trades
    baseline = BaselineStrategy(max_alloc=strategy.max_alloc)
    folds, equities, base_equities, trades, round_trips = [], [], [], [], []
    cash, base_cash = initial_cash, initial_cash
    for train_start, train_end, test_start, test_end in make_folds(start, end, train_months, test_months):
        best = None
        for rules in grid:
            strategy.rules = rules
            res = run_backtest(prices, strategy, train_start, train_end, initial_cash, cost_bps)
            m = summarize(res.equity, res.trades)
            rank = _rank(m, min_trades)
            if best is None or rank > best[0]:
                best = (rank, rules, m)
        _, best_rules, train_m = best
        strategy.rules = best_rules
        test = run_backtest(prices, strategy, test_start, test_end, cash, cost_bps, liquidate_at_end=True)
        base = run_backtest(prices, baseline, test_start, test_end, base_cash, cost_bps, liquidate_at_end=True)
        folds.append(Fold(train_start, train_end, test_start, test_end, best_rules, train_m["sharpe"], train_m["trades"],
                          summarize(test.equity, test.trades, test.round_trip_pnl)))
        equities.append(test.equity)
        base_equities.append(base.equity)
        trades.append(test.trades)
        round_trips += test.round_trip_pnl
        cash, base_cash = test.equity.iloc[-1], base.equity.iloc[-1]
    if not folds:
        raise ValueError("period too short for one train + test window")
    return WalkForwardResult(folds, pd.concat(equities), pd.concat(base_equities),
                             pd.concat(trades, ignore_index=True), round_trips)
