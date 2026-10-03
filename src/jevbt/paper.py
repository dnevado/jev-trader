"""Forward paper trading: turn this week's strategy targets into orders on an Alpaca PAPER account.

Same decision path as the backtest: indicators from daily bars up to the previous session (features_asof),
the same Strategy.target, the same gross-exposure cap (engine.apply_gross_cap), and market-on-open orders
("opg", like the backtest's execution at the next open). Dry run by default: orders are only printed and
logged unless `submit=True`. Whole shares only (Alpaca does not short fractional shares); a long ↔ short flip
is sent as a close order followed by an open order. Every run is logged to data/paper/<timestamp>.json.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta

import pandas as pd

from jevbt.backtest.engine import apply_gross_cap
from jevbt.broker.alpaca_paper import AlpacaPaperBroker
from jevbt.config import Settings
from jevbt.features.technical import compute_technical, features_asof
from jevbt.ingest.alpaca import AlpacaClient
from jevbt.strategy import Strategy

HISTORY_DAYS = 600  # calendar days of bars: > 252 sessions for 12-month momentum and the SMA200


@dataclass
class Order:
    ticker: str
    side: str        # "buy" | "sell"
    qty: int
    reason: str      # "close" | "open" | "adjust"
    ref_price: float  # last close used for sizing (the fill will be at the open)


def plan_orders(targets: dict[str, float], positions: dict[str, float], ref_prices: dict[str, float],
                equity: float) -> list[Order]:
    """Orders that move `positions` (signed shares) to the target weights, in whole shares (rounded toward
    zero). Closing orders come first; a flip is a close followed by an open."""
    orders: list[Order] = []
    for tk, w in targets.items():
        price = ref_prices[tk]
        target_qty = int(math.trunc(w * equity / price)) if price > 0 else 0
        current = positions.get(tk, 0.0)
        if current and target_qty * current <= 0:
            orders.append(Order(tk, "sell" if current > 0 else "buy", int(round(abs(current))), "close", price))
            current = 0.0
        delta = target_qty - int(round(current))
        if delta:
            orders.append(Order(tk, "buy" if delta > 0 else "sell", abs(delta), "open" if not current else "adjust", price))
    return sorted(orders, key=lambda o: o.reason != "close")


def decide(prices: dict[str, pd.DataFrame], strategy: Strategy, t: pd.Timestamp, positions: dict[str, float],
           equity: float, max_gross: float | None, held_outside: float = 0.0
           ) -> tuple[dict[str, float], dict[str, float], list[dict]]:
    """Target weights at session t from data ≤ t-1. `positions` are signed shares; `held_outside` is the gross
    weight of open positions outside the universe. Returns (targets, reference prices, decision records)."""
    decisions, ref_prices = [], {}
    for tk, df in prices.items():
        technical = features_asof(compute_technical(df), t)
        price = float(technical["close"])
        ref_prices[tk] = price
        current_w = positions.get(tk, 0.0) * price / equity if equity > 0 else 0.0
        target_w, record = strategy.target(tk, t, technical, current_w)
        decisions.append([tk, price, current_w, target_w, {**record, "data_until": str(technical.name.date())}])
    if max_gross is not None and equity > 0:
        apply_gross_cap(decisions, held_outside, max_gross)
    targets = {d[0]: d[3] for d in decisions}
    records = [{"ticker": d[0], "ref_price": d[1], "current_weight": d[2], "target_weight": d[3], **d[4]}
               for d in decisions]
    return targets, ref_prices, records


def first_session_of_week(sessions: list[str], today: date) -> bool:
    week = [d for d in sessions if date.fromisoformat(d).isocalendar()[:2] == today.isocalendar()[:2]]
    return bool(week) and week[0] == today.isoformat()


def run_paper(settings: Settings, tickers: list[str], strategy: Strategy, max_gross: float | None = 1.0,
              submit: bool = False, time_in_force: str = "opg", force: bool = False, equity: float | None = None,
              today: date | None = None, broker: AlpacaPaperBroker | None = None,
              prices_client: AlpacaClient | None = None) -> dict:
    """One weekly paper-trading step. Returns (and logs) the decisions and orders."""
    today = today or date.today()
    t = pd.Timestamp(today)
    notes = []
    broker = broker if broker is not None else (AlpacaPaperBroker(settings) if (submit or equity is None) else None)
    if broker is not None:
        account = broker.account()
        detail = broker.positions()
        if equity is None:
            equity = account["equity"]
        positions = {s: p["qty"] for s, p in detail.items()}
        held_outside = sum(abs(p["market_value"]) for s, p in detail.items() if s not in tickers) / equity
        if held_outside:
            notes.append(f"positions outside the universe use {held_outside:.0%} of equity (counted in the cap)")
        sessions = broker.calendar((today - timedelta(days=7)).isoformat(), (today + timedelta(days=7)).isoformat())
        if today.isoformat() not in sessions:
            notes.append(f"{today} is not a trading session")
        elif not first_session_of_week(sessions, today):
            notes.append(f"{today} is not the first session of the week (the backtest rebalances weekly)")
    else:
        account, positions, held_outside = None, {}, 0.0
    if submit and notes and not force:
        raise SystemExit("not submitting: " + "; ".join(notes) + " (use --force to override)")
    if submit and account and (account["trading_blocked"] or account["status"] != "ACTIVE"):
        raise SystemExit(f"not submitting: paper account status {account['status']}, blocked={account['trading_blocked']}")

    prices_client = prices_client or AlpacaClient(settings)
    start = (today - timedelta(days=HISTORY_DAYS)).isoformat()
    prices = {tk: prices_client.prices(tk, start, today) for tk in tickers}  # bars up to yesterday
    targets, ref_prices, records = decide(prices, strategy, t, positions, equity, max_gross, held_outside)
    orders = plan_orders(targets, positions, ref_prices, equity)

    results = []
    for o in orders:
        if o.side == "sell" and o.reason != "close" and positions.get(o.ticker, 0.0) <= 0 and broker is not None:
            asset = broker.asset(o.ticker)
            if not (asset.get("shortable") and asset.get("easy_to_borrow") and account and account["shorting_enabled"]):
                results.append({**asdict(o), "status": "skipped: not shortable / shorting disabled"})
                continue
        if not submit:
            results.append({**asdict(o), "status": "dry-run"})
            continue
        coid = f"jevbt-{today:%Y%m%d}-{o.ticker}-{o.reason}-{o.side}"
        resp = broker.submit_market_order(o.ticker, o.qty, o.side, time_in_force, coid)
        results.append({**asdict(o), "status": resp.get("status"), "order_id": resp.get("id"), "client_order_id": coid})

    run = {"date": today.isoformat(), "submitted": submit, "time_in_force": time_in_force, "strategy": repr(strategy),
           "tickers": tickers, "equity": equity, "max_gross": max_gross, "positions_before": positions,
           "notes": notes, "decisions": records, "orders": results}
    out_dir = settings.data_dir / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}{'' if submit else '_dry-run'}.json"
    path.write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    run["log_path"] = str(path)
    return run
