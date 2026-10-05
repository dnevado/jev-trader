"""Command line:
  python -m jevbt ingest      --tickers AAPL MSFT --start 2021-01-01 --end 2025-12-31
  python -m jevbt baseline    --tickers AAPL MSFT --start 2023-01-01 --end 2026-09-25
  python -m jevbt backtest    --tickers AAPL MSFT --start 2023-01-01 --end 2026-09-25 [--offline] [--entry-mode signals]
  python -m jevbt walkforward --tickers AAPL MSFT --start 2023-01-01 --end 2026-09-25 [--offline]
  python -m jevbt research "profitable US large caps with accelerating revenue" [--seed AMZN MSFT] [--max-tickers 10]
  python -m jevbt serve [--port 8000]      # API for the React UI in ui/
  python -m jevbt paper --tickers AMD NKE ... [--direction long] [--submit]   # Alpaca PAPER account, dry run by default
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from jevbt.config import Settings, load_settings
from jevbt.ingest.alpaca import AlpacaClient, AlpacaError, AlpacaPlanError
from jevbt.ingest.fmp import STATEMENT_PATHS, FMPClient, FMPError, FMPPlanError

WARMUP_DAYS = 400  # > 252 sessions for 12-month momentum and SMA200


def _ingest(args: argparse.Namespace) -> int:
    settings = load_settings()
    client, prices_client = FMPClient(settings), AlpacaClient(settings)
    status = 0
    for ticker in args.tickers:
        try:
            px = prices_client.prices(ticker, args.start, args.end, refresh=args.refresh)
            print(f"{ticker}: {len(px)} price rows ({px.index.min():%Y-%m-%d} .. {px.index.max():%Y-%m-%d})"
                  if len(px) else f"{ticker}: no price rows")
            for kind in STATEMENT_PATHS:
                st = client.statements(ticker, kind, args.period, refresh=args.refresh)
                print(f"  {kind}_{args.period}: {len(st)} periods")
        except FMPPlanError as e:
            print(f"{ticker}: not available on the current FMP plan: {e}", file=sys.stderr)
            status = 1
        except AlpacaPlanError as e:
            print(f"{ticker}: Alpaca credentials or plan rejected the request: {e}", file=sys.stderr)
            status = 1
        except (FMPError, AlpacaError) as e:
            print(f"{ticker}: {e}", file=sys.stderr)
            status = 1
    return status


def _exit_options(args: argparse.Namespace) -> dict:
    """Engine exit options (trailing stops, take-profit, re-entry after a stop) from the CLI."""
    return {"trailing_stop": args.trailing_stop, "take_profit": args.take_profit,
            "trailing_stop_atr": args.trailing_stop_atr, "stop_rearm": args.stop_rearm,
            "stop_cooldown_weeks": args.stop_cooldown, "max_gross": args.max_gross, "rebalance": args.rebalance}


def _rule_flags(args: argparse.Namespace) -> dict:
    """JevRules overrides from --no-overbought / --no-sell-veto / --no-exit-on-sell / --min-quality /
    --valuation-penalty (the last two fix the value for every rule set of the walk-forward grid)."""
    flags = {"use_overbought": not getattr(args, "no_overbought", False),
             "sell_veto": not getattr(args, "no_sell_veto", False),
             "exit_on_sell": not getattr(args, "no_exit_on_sell", False)}
    for name in ("min_quality", "valuation_penalty"):
        if getattr(args, name, None) is not None:
            flags[name] = getattr(args, name)
    flags["direction"] = getattr(args, "direction", "long")
    return flags


def _jev_strategy(settings: Settings, client: FMPClient, tickers: list[str], offline: bool, max_alloc: float,
                  entry_mode: str = "action", flags: dict | None = None):
    from jevbt.decision.jev import JevDecider, MockJevClassifier, typesafe_classifier
    from jevbt.features.fundamentals import compute_ratios
    from jevbt.graph import build_graph
    from jevbt.llm.summarizer import MockSummaryLLM, Summarizer, openai_summarizer_llm
    from jevbt.strategy import JevRules, JevStrategy

    ratios = {tk: compute_ratios(*(client.statements(tk, k, "annual") for k in ("income", "balance", "cashflow")))
              for tk in tickers}
    if offline:
        cache = settings.cache_dir / "mock"
        summarizer = Summarizer(MockSummaryLLM(), "mock/summary", cache)
        decider = JevDecider(MockJevClassifier(), cache)
    else:
        if not settings.typesafe_api_key:
            raise SystemExit("TYPESAFE_API_KEY / OPENROUTER_API_KEY not set")
        summarizer = Summarizer(openai_summarizer_llm(settings.openai_model_fast), settings.openai_model_fast,
                                settings.cache_dir)
        decider = JevDecider(typesafe_classifier(settings.jev_model, settings.typesafe_api_key,
                                                 settings.typesafe_base_url), settings.cache_dir)
    strategy = JevStrategy(build_graph(summarizer, decider), ratios, JevRules(entry_mode=entry_mode, **(flags or {})),
                           max_alloc)
    return strategy, summarizer, decider


def _load(args: argparse.Namespace, jev: bool):
    """Prices (Alpaca) with warm-up (+ Jev strategy, FMP statements) for the requested tickers, cached when possible."""
    settings = load_settings()
    client, prices_client = FMPClient(settings), AlpacaClient(settings)
    tickers = [t.upper() for t in args.tickers]
    max_alloc = args.max_alloc or 1 / len(tickers)
    warm_start = (pd.Timestamp(args.start) - pd.Timedelta(days=WARMUP_DAYS)).date()
    prices = {}
    for tk in tickers:
        try:
            prices[tk] = prices_client.prices(tk, warm_start, args.end)
        except AlpacaError as e:
            raise type(e)(f"{tk}: {e}") from None
    parts = (_jev_strategy(settings, client, tickers, args.offline, max_alloc, getattr(args, "entry_mode", "action"),
                           _rule_flags(args))
             if jev else (None, None, None))
    return settings, tickers, max_alloc, prices, parts


def _run_dir(settings: Settings, name: str, offline: bool) -> Path:
    run_dir = settings.runs_dir / f"{datetime.now():%Y%m%d-%H%M%S}_{name}{'_offline' if offline else ''}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _fmt(x: float) -> str:
    return f"{x:,.4f}"


def _backtest(args: argparse.Namespace) -> int:
    from jevbt.backtest.engine import buy_and_hold, run_backtest
    from jevbt.backtest.metrics import summarize
    from jevbt.strategy import (JEV_LOOSE_LONG, JEV_STRICT_BOTH, BaselineStrategy, JevStrategy, RegimeSwitchStrategy,
                                TrendConfidenceStrategy, jev_short_confirmation)

    regime = args.strategy == "regime"
    needs_jev = (args.strategy == "jev" or (regime and "jev" in (args.bull, args.bear))
                 or (args.strategy == "trend" and args.jev_confirm_short))
    try:
        settings, tickers, max_alloc, prices, (strategy, summarizer, decider) = _load(args, needs_jev)
        if regime:
            warm_start = (pd.Timestamp(args.start) - pd.Timedelta(days=WARMUP_DAYS)).date()
            index_prices = AlpacaClient(settings).prices(args.regime_index, warm_start, args.end)
    except (FMPError, AlpacaError) as e:
        print(e, file=sys.stderr)
        return 1
    if regime:
        jev = strategy

        def side(kind, preset, direction):
            if kind == "baseline":
                return BaselineStrategy(max_alloc=max_alloc, direction=direction)
            return JevStrategy(jev.graph, jev.ratios, preset, max_alloc)

        exits = ({("bear", 1): {"trailing_stop_atr": args.bear_long_stop_atr, "stop_rearm": True}}
                 if args.bear_long_stop_atr else {})
        strategy = RegimeSwitchStrategy(index_prices, side(args.bull, JEV_LOOSE_LONG, "long"),
                                        side(args.bear, JEV_STRICT_BOTH, "both"), exits)
        name = f"regime-{args.regime_index}-{args.bull}-{args.bear}"
    elif args.strategy == "trend":
        confirm = None
        if args.jev_confirm_short:
            confirm = jev_short_confirmation(JevStrategy(strategy.graph, strategy.ratios, max_alloc=max_alloc))
        strategy = TrendConfidenceStrategy(max_alloc=max_alloc, direction=args.direction,
                                           vol_sizing=args.vol_sizing, confirm_short=confirm)
        name = "trend"
    elif strategy is None:
        strategy = BaselineStrategy(max_alloc=max_alloc, direction=args.direction)
        name = "baseline"
    else:
        name = f"jev-{args.entry_mode}"
    if args.direction != "long" and not regime:
        name += f"-{args.direction}"
    run_dir = _run_dir(settings, name, args.offline)
    result = run_backtest(prices, strategy, args.start, args.end, cost_bps=args.cost_bps,
                          log_path=run_dir / "decisions.jsonl", borrow_bps=args.borrow_bps, **_exit_options(args))
    bh = buy_and_hold(prices, args.start, args.end, cost_bps=args.cost_bps)
    metrics = {
        name: summarize(result.equity, result.trades, result.round_trip_pnl),
        "buy_and_hold": summarize(bh),
    }
    result.equity.to_csv(run_dir / "equity.csv")
    result.trades.to_csv(run_dir / "trades.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(
        {"tickers": tickers, "start": args.start, "end": args.end, "max_alloc": max_alloc,
         "cost_bps": args.cost_bps, "offline": args.offline, "entry_mode": getattr(args, "entry_mode", None),
         "direction": args.direction, "borrow_bps": args.borrow_bps,
         "exits": _exit_options(args),
         "regime": {"index": args.regime_index, "bull": args.bull, "bear": args.bear,
                    "bear_long_stop_atr": args.bear_long_stop_atr} if regime else None,
         "metrics": metrics}, indent=2))

    print(pd.DataFrame(metrics).T.to_string(float_format=_fmt))
    if summarizer is not None:
        print(f"new OpenAI summaries: {summarizer.calls}, new Jev calls: {decider.calls}")
    print(f"run saved to {run_dir}")
    return 0


def _walkforward(args: argparse.Namespace) -> int:
    from jevbt.backtest.engine import buy_and_hold
    from jevbt.backtest.metrics import summarize
    from jevbt.backtest.walkforward import default_grid, walk_forward

    try:
        settings, tickers, max_alloc, prices, (strategy, summarizer, decider) = _load(args, True)
    except (FMPError, AlpacaError) as e:
        print(e, file=sys.stderr)
        return 1
    flags = _rule_flags(args)
    wf = walk_forward(prices, strategy, args.start, args.end, grid=default_grid(**flags), train_months=args.train_months,
                      test_months=args.test_months, cost_bps=args.cost_bps, min_trades=args.min_trades,
                      borrow_bps=args.borrow_bps, **_exit_options(args))
    oos_start, oos_end = wf.equity.index[0], wf.equity.index[-1]
    metrics = {
        "jev_walkforward": summarize(wf.equity, wf.trades, wf.round_trip_pnl),
        "baseline": summarize(wf.baseline_equity),
        "buy_and_hold": summarize(buy_and_hold(prices, oos_start, oos_end, cost_bps=args.cost_bps)),
    }
    run_dir = _run_dir(settings, "walkforward", args.offline)
    table = wf.folds_table()
    table.to_csv(run_dir / "folds.csv", index=False)
    wf.equity.to_csv(run_dir / "equity.csv")
    wf.trades.to_csv(run_dir / "trades.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(
        {"tickers": tickers, "start": args.start, "end": args.end,
         "out_of_sample": [str(oos_start.date()), str(oos_end.date())],
         "train_months": args.train_months, "test_months": args.test_months, "min_trades": args.min_trades,
         "rule_flags": flags, "borrow_bps": args.borrow_bps,
         "exits": _exit_options(args),
         "max_alloc": max_alloc,
         "cost_bps": args.cost_bps, "offline": args.offline, "metrics": metrics}, indent=2))
    with pd.option_context("display.width", 220):
        print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print(f"\nOut-of-sample {oos_start:%Y-%m-%d} .. {oos_end:%Y-%m-%d}:")
        print(pd.DataFrame(metrics).T.to_string(float_format=_fmt))
    print(f"new OpenAI summaries: {summarizer.calls}, new Jev calls: {decider.calls}")
    print(f"run saved to {run_dir}")
    return 0


def _paper(args: argparse.Namespace) -> int:
    from jevbt.broker.alpaca_paper import BrokerError
    from jevbt.paper import run_paper
    from jevbt.strategy import BaselineStrategy, TrendConfidenceStrategy

    tickers = [t.upper() for t in args.tickers]
    if args.strategy == "trend":
        strategy = TrendConfidenceStrategy(max_alloc=args.max_alloc, direction=args.direction, vol_sizing=args.vol_sizing)
    else:
        strategy = BaselineStrategy(max_alloc=args.max_alloc, direction=args.direction)
    try:
        run = run_paper(load_settings(), tickers, strategy, max_gross=args.max_gross, submit=args.submit,
                        time_in_force=args.tif, force=args.force, equity=args.equity, rebalance=args.rebalance)
    except (BrokerError, AlpacaError) as e:
        print(e, file=sys.stderr)
        return 1
    print(f"{'SUBMITTED' if run['submitted'] else 'DRY RUN (nothing sent; add --submit)'}  {run['date']}  "
          f"equity {run['equity']:,.2f}  max gross {run['max_gross']}")
    for note in run["notes"]:
        print(f"note: {note}")
    for d in run["decisions"]:
        print(f"  {d['ticker']:6} data to {d['data_until']}  weight {d['current_weight']:+.3f} -> {d['target_weight']:+.3f}"
              f"  {d.get('reason', '')}")
    print("orders:" if run["orders"] else "orders: none")
    for o in run["orders"]:
        print(f"  {o['side']:4} {o['qty']:6d} {o['ticker']:6} ({o['reason']}, ref {o['ref_price']:.2f})  {o['status']}")
    print(f"log: {run['log_path']}")
    return 0


def _research(args: argparse.Namespace) -> int:
    import asyncio

    from jevbt.api import research_dir, save_research
    from jevbt.research.mcp_agent import run_research

    settings = load_settings()
    result = asyncio.run(run_research(settings, args.criteria, args.max_tickers, args.seed))
    run = save_research(settings, result)
    for c in result.candidates:
        print(f"{c.ticker:6} {c.rationale}")
    print(f"\ntickers: {' '.join(c.ticker for c in result.candidates)}")
    print(f"saved to {research_dir(settings) / (run.id + '.json')}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("jevbt.api:create_app", factory=True, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jevbt")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="download prices (Alpaca) and statements (FMP) into the Parquet cache")
    ingest.add_argument("--tickers", nargs="+", required=True)
    ingest.add_argument("--start", required=True)
    ingest.add_argument("--end", required=True)
    ingest.add_argument("--period", choices=["quarter", "annual"], default="quarter")
    ingest.add_argument("--refresh", action="store_true", help="refetch statements and prices even if cached")

    for name, help_ in (("backtest", "run a strategy backtest (default: jev)"),
                        ("baseline", "run the no-LLM baseline (same as backtest --strategy baseline)"),
                        ("walkforward", "walk-forward validation of the Jev rules (out-of-sample)")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--tickers", nargs="+", required=True)
        p.add_argument("--start", required=True)
        p.add_argument("--end", required=True)
        p.add_argument("--max-alloc", type=float, default=None, help="max weight per position (default 1/N)")
        p.add_argument("--cost-bps", type=float, default=10.0)
        p.add_argument("--direction", choices=["long", "short", "both"], default="long",
                       help="long only (default), short only (mirror rules) or both (flip on reversal)")
        p.add_argument("--borrow-bps", type=float, default=30.0, help="annual borrow fee on short positions")
        p.add_argument("--trailing-stop", type=float, default=None,
                       help="trailing stop as a fraction, e.g. 0.03 = exit 3%% from the best price since entry")
        p.add_argument("--take-profit", type=float, default=None,
                       help="take-profit as a fraction of the entry price, e.g. 0.10")
        p.add_argument("--trailing-stop-atr", type=float, default=None,
                       help="volatility-adjusted trailing stop in ATRs, e.g. 3 = 3 x ATR14 from the best price")
        p.add_argument("--stop-rearm", action="store_true",
                       help="after a stop, re-enter the same side only after the signal has switched off once")
        p.add_argument("--stop-cooldown", type=int, default=0, help="after a stop, wait at least N weeks to re-enter")
        p.add_argument("--rebalance", choices=["weekly", "daily"], default="weekly",
                       help="decide and trade on the first session of each week (default) or every session")
        p.add_argument("--max-gross", type=float, default=None,
                       help="cap on gross exposure at each rebalance (1.0 = fully invested, no leverage)")
        if name == "backtest":
            p.add_argument("--strategy", choices=["jev", "baseline", "regime", "trend"], default="jev",
                           help="trend = trade only confident up/down trends (use --direction both for long+short)")
            p.add_argument("--vol-sizing", action="store_true",
                           help="trend: size entries by 2%% / ATR%% (0.5-2x --max-alloc); combine with --max-gross")
            p.add_argument("--jev-confirm-short", action="store_true",
                           help="trend: open shorts only when Jev confirms (paid calls unless cached)")
            p.add_argument("--regime-index", default="SPY", help="regime: index whose close vs SMA200 sets bull/bear")
            p.add_argument("--bull", choices=["baseline", "jev"], default="baseline",
                           help="regime: long-only strategy above the index SMA200 (jev = loose long preset)")
            p.add_argument("--bear", choices=["baseline", "jev"], default="jev",
                           help="regime: long+short strategy below the index SMA200 (jev = strict long+short preset)")
            p.add_argument("--bear-long-stop-atr", type=float, default=None,
                           help="regime: ATR trailing stop (+ fresh-signal re-entry) for longs opened in the bear regime")
            p.add_argument("--entry-mode", choices=["action", "signals"], default="action")
        if name == "walkforward":
            p.add_argument("--train-months", type=int, default=12)
            p.add_argument("--test-months", type=int, default=3)
            p.add_argument("--min-trades", type=int, default=None,
                           help="min trades in a train window for a rule set to be eligible (default 2 per ticker)")
        if name != "baseline":
            p.add_argument("--offline", action="store_true", help="mock OpenAI and Jev (no paid calls)")
            p.add_argument("--no-overbought", action="store_true", help="ignore Jev's overbought answer on entry")
            p.add_argument("--no-sell-veto", action="store_true", help="allow entries while Jev's action is sell")
            p.add_argument("--no-exit-on-sell", action="store_true", help="do not exit on Jev's action == sell")
            p.add_argument("--min-quality", type=float, default=None,
                           help="min expected fundamental_quality to enter (0 = no quality filter)")
            p.add_argument("--valuation-penalty", type=float, default=None,
                           help="size × (1 − penalty × valuation_risk); 0 = size by trend_up / p(buy) only")

    research = sub.add_parser("research", help="research agent (OpenAI + FMP MCP) that proposes tickers")
    research.add_argument("criteria", help="what kind of companies to look for")
    research.add_argument("--max-tickers", type=int, default=10)
    research.add_argument("--seed", nargs="*", default=None, help="seed tickers to expand with peers")

    paper = sub.add_parser("paper", help="weekly step on an Alpaca PAPER account (dry run unless --submit)")
    paper.add_argument("--tickers", nargs="+", required=True)
    paper.add_argument("--strategy", choices=["trend", "baseline"], default="trend")
    paper.add_argument("--direction", choices=["long", "short", "both"], default="long")
    paper.add_argument("--max-alloc", type=float, default=1 / 12, help="weight per position (default 1/12)")
    paper.add_argument("--max-gross", type=float, default=1.0, help="cap on gross exposure (default 1.0)")
    paper.add_argument("--vol-sizing", action="store_true")
    paper.add_argument("--submit", action="store_true", help="send the orders to the paper account")
    paper.add_argument("--tif", choices=["opg", "day"], default="opg",
                       help="opg = market-on-open (submit before 09:28 ET, like the backtest); day = now")
    paper.add_argument("--rebalance", choices=["weekly", "daily"], default="weekly",
                       help="weekly: submit only on the first session of the week; daily: any trading session")
    paper.add_argument("--force", action="store_true", help="submit even if today is not a rebalance session")
    paper.add_argument("--equity", type=float, default=None,
                       help="dry run without paper keys: size orders for this equity (assumes no positions)")

    serve = sub.add_parser("serve", help="HTTP API for the React UI (ui/); JEVBT_RESEARCH_MOCK=1 for mock runs")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    if args.command == "ingest":
        return _ingest(args)
    if args.command == "paper":
        return _paper(args)
    if args.command == "research":
        return _research(args)
    if args.command == "serve":
        return _serve(args)
    if args.command == "walkforward":
        return _walkforward(args)
    if args.command == "baseline":
        args.strategy, args.offline = "baseline", False
    return _backtest(args)
