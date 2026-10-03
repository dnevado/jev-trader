"""Entry/exit/sizing rules. Strategies return a target weight per ticker at each rebalance date.

Weights are fractions of portfolio equity, capped at `max_alloc` per position. A held position that
triggers no exit keeps its current weight (no resizing) to limit turnover.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from jevbt.features.technical import compute_technical, features_asof


class Strategy(Protocol):
    name: str

    def target(self, ticker: str, t: pd.Timestamp, technical: pd.Series, current_weight: float) -> tuple[float, dict]:
        """(target weight, log record) for `ticker` at rebalance date t, from data ≤ t-1."""


Direction = Literal["long", "short", "both"]


@dataclass
class BaselineStrategy:
    """No-LLM baseline (CLAUDE.md §1.9).

    Long:  enter when close > SMA200 and RSI14 < 70; exit when close < SMA200.
    Short (mirror): enter when close < SMA200 and RSI14 > 30; cover when close > SMA200.
    direction "both" flips sides when the exit of one side is the entry of the other.
    """

    max_alloc: float = 0.1
    direction: Direction = "long"
    name: str = "baseline"

    def target(self, ticker, t, technical, current_weight):
        close, sma200, rsi = technical["close"], technical["sma200"], technical["rsi14"]
        if not np.isfinite(sma200):
            return 0.0, {"reason": "warm-up"}
        long_entry = self.direction != "short" and close > sma200 and rsi < 70
        short_entry = self.direction != "long" and close < sma200 and rsi > 30
        if current_weight > 0:
            if close < sma200:
                if short_entry:
                    return -self.max_alloc, {"reason": "flip to short: close < sma200 and rsi > 30"}
                return 0.0, {"reason": "exit: close < sma200"}
            return current_weight, {"reason": "hold"}
        if current_weight < 0:
            if close > sma200:
                if long_entry:
                    return self.max_alloc, {"reason": "flip to long: close > sma200 and rsi < 70"}
                return 0.0, {"reason": "cover: close > sma200"}
            return current_weight, {"reason": "hold short"}
        if long_entry:
            return self.max_alloc, {"reason": "entry: close > sma200 and rsi < 70"}
        if short_entry:
            return -self.max_alloc, {"reason": "short: close < sma200 and rsi > 30"}
        return 0.0, {"reason": "no entry"}


@dataclass(frozen=True)
class JevRules:
    """CLAUDE.md §5 thresholds (optimized with walk-forward validation, see backtest/walkforward.py).

    entry_mode:
      "action"  — original §5 rule: requires Jev's action == buy with confidence ≥ min_confidence;
                  size = p(buy) × (1 − penalty × valuation_risk).
      "signals" — entry from the Noul/Score answers only (trend, overbought, quality), vetoed by action == sell;
                  size = trend_up × (1 − penalty × valuation_risk). Jev's action is still used to exit.
    use_overbought — require overbought < max_overbought to enter (an RSI-like gate that blocks strong trends).
    sell_veto      — "signals" mode: no entry while action == sell.
    exit_on_sell   — exit a held position when action == sell (besides trend_up < exit_trend_up).
    direction      — "long", "short" or "both". Shorts mirror every long rule: trend_up ≤ 1 − min_trend_up,
                     quality ≤ 2 − min_quality, action == sell with p(sell) sizing (action mode), buy veto and
                     cover on buy (sell_veto / exit_on_sell), cover when trend_up > 1 − exit_trend_up, size based
                     on 1 − trend_up and × (1 − penalty × (1 − valuation_risk)) (high valuation risk supports a short).
                     Jev has no "oversold" question, so shorts have no overbought-style gate.
    """

    entry_mode: Literal["action", "signals"] = "action"
    min_confidence: float = 0.70
    min_trend_up: float = 0.75
    max_overbought: float = 0.50
    min_quality: float = 1.5      # expected Score level: 0 weak, 1 medium, 2 strong
    exit_trend_up: float = 0.40
    valuation_penalty: float = 0.5
    use_overbought: bool = True
    sell_veto: bool = True
    exit_on_sell: bool = True
    direction: Direction = "long"


class JevStrategy:
    """Jev decisions do not depend on the position or the rules, so they are memoized per (ticker, t):
    changing `rules` (walk-forward grid search) re-evaluates the rules without calling the graph again."""

    name = "jev"

    def __init__(self, graph, ratios: dict[str, pd.DataFrame], rules: JevRules | None = None,
                 max_alloc: float = 0.1) -> None:
        self.graph = graph
        self.ratios = ratios
        self.rules = rules or JevRules()
        self.max_alloc = max_alloc
        self._memo: dict[tuple[str, pd.Timestamp], dict] = {}

    def decision(self, ticker: str, t: pd.Timestamp, technical: pd.Series) -> dict:
        key = (ticker, t)
        if key not in self._memo:
            self._memo[key] = self.graph.invoke(
                {"ticker": ticker, "t": t, "technical": technical, "ratios": self.ratios[ticker]})
        return self._memo[key]

    def target(self, ticker, t, technical, current_weight):
        if not np.isfinite(technical["sma200"]):
            return 0.0, {"reason": "warm-up"}
        out = self.decision(ticker, t, technical)
        d = out["decision"]
        r = self.rules
        log = {"state": out["state_text"], "jev": d.model_dump(), "rules": asdict(r)}
        long_size = self._long_size(d) if r.direction != "short" else 0.0
        short_size = self._short_size(d) if r.direction != "long" else 0.0
        if current_weight > 0:
            if (r.exit_on_sell and d.action == "sell") or d.trend_up < r.exit_trend_up:
                if short_size:
                    return -short_size * self.max_alloc, {**log, "reason": "flip to short"}
                return 0.0, {**log, "reason": "exit"}
            return current_weight, {**log, "reason": "hold"}
        if current_weight < 0:
            if (r.exit_on_sell and d.action == "buy") or d.trend_up > 1 - r.exit_trend_up:
                if long_size:
                    return long_size * self.max_alloc, {**log, "reason": "flip to long"}
                return 0.0, {**log, "reason": "cover"}
            return current_weight, {**log, "reason": "hold short"}
        if long_size:
            return long_size * self.max_alloc, {**log, "reason": "entry"}
        if short_size:
            return -short_size * self.max_alloc, {**log, "reason": "short entry"}
        return 0.0, {**log, "reason": "no entry"}

    def _long_size(self, d) -> float:
        """Long entry size as a fraction of max_alloc (0 = no entry)."""
        r = self.rules
        if not (d.trend_up >= r.min_trend_up
                and (not r.use_overbought or d.overbought < r.max_overbought)
                and d.fundamental_quality >= r.min_quality):
            return 0.0
        if r.entry_mode == "action":
            if d.action == "buy" and d.action_confidence >= r.min_confidence:
                return d.p_buy * (1 - r.valuation_penalty * d.valuation_risk)
            return 0.0
        if r.sell_veto and d.action == "sell":
            return 0.0
        return d.trend_up * (1 - r.valuation_penalty * d.valuation_risk)

    def _short_size(self, d) -> float:
        """Mirror of _long_size for shorts (0 = no entry)."""
        r = self.rules
        if not (d.trend_up <= 1 - r.min_trend_up and d.fundamental_quality <= 2 - r.min_quality):
            return 0.0
        if r.entry_mode == "action":
            if d.action == "sell" and d.action_confidence >= r.min_confidence:
                return d.p_sell * (1 - r.valuation_penalty * (1 - d.valuation_risk))
            return 0.0
        if r.sell_veto and d.action == "buy":
            return 0.0
        return (1 - d.trend_up) * (1 - r.valuation_penalty * (1 - d.valuation_risk))


NO_EXITS = {"trailing_stop": None, "take_profit": None, "trailing_stop_atr": None,
            "stop_rearm": False, "stop_cooldown_weeks": 0}

# Jev rule presets found in the 2024–26 experiments (CLAUDE.md §8): loose long-only rules for rising
# markets, the original strict rules (long + short) for falling ones.
JEV_LOOSE_LONG = JevRules(entry_mode="signals", min_trend_up=0.60, exit_trend_up=0.30, min_quality=0.0,
                          valuation_penalty=0.0, use_overbought=False, sell_veto=False, exit_on_sell=False)
JEV_STRICT_BOTH = JevRules(entry_mode="signals", min_trend_up=0.60, max_overbought=0.40, min_quality=1.0,
                           exit_trend_up=0.50, direction="both")


class RegimeSwitchStrategy:
    """Market-regime switch: the regime at rebalance date t is "bull" when the index closed above its SMA200
    on the last session before t, else "bear" (bull during the index warm-up). Each regime delegates to its
    own strategy; an open position is handed over to the new regime's rules (not force-closed).

    `exits` maps (regime, side) → engine exit settings for positions opened in that regime and side
    (see run_backtest), e.g. {("bear", 1): {"trailing_stop_atr": 2, "stop_rearm": True}}; anything not set
    there has no exit (global exit arguments of run_backtest do not apply to a regime strategy).
    """

    name = "regime"

    def __init__(self, index_prices: pd.DataFrame, bull: Strategy, bear: Strategy,
                 exits: dict[tuple[str, int], dict] | None = None) -> None:
        self.index_features = compute_technical(index_prices)
        self.bull, self.bear = bull, bear
        self.exits = exits or {}

    def regime(self, t: pd.Timestamp) -> str:
        try:
            row = features_asof(self.index_features, t)
        except ValueError:
            return "bull"
        return "bear" if np.isfinite(row["sma200"]) and row["close"] < row["sma200"] else "bull"

    def target(self, ticker, t, technical, current_weight):
        regime = self.regime(t)
        weight, record = (self.bull if regime == "bull" else self.bear).target(ticker, t, technical, current_weight)
        return weight, {**record, "regime": regime}

    def release_stop_block(self, t: pd.Timestamp, stopped_at: pd.Timestamp) -> bool:
        """A re-entry block set by a stop in one regime does not carry over into the other regime."""
        return self.regime(t) != self.regime(stopped_at)

    def position_exits(self, t: pd.Timestamp, sign: int) -> dict:
        # The regime at entry decides the exits for the whole life of the position.
        return {**NO_EXITS, **self.exits.get((self.regime(t), sign), {})}
