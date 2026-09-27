"""Command line:
  python -m jevbt ingest      --tickers AAPL MSFT --start 2021-01-01 --end 2025-12-31
  python -m jevbt baseline    --tickers AAPL MSFT --start 2023-01-01 --end 2026-09-25
  python -m jevbt backtest    --tickers AAPL MSFT --start 2023-01-01 --end 2026-09-25 [--offline] [--entry-mode signals]
  python -m jevbt walkforward --tickers AAPL MSFT --start 2023-01-01 --end 2026-09-25 [--offline]
  python -m jevbt research "profitable US large caps with accelerating revenue" [--seed AMZN MSFT] [--max-tickers 10]
  python -m jevbt serve [--port 8000]      # API for the React UI in ui/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from jevbt.config import Settings, load_settings
from jevbt.ingest.fmp import STATEMENT_PATHS, FMPClient, FMPError, FMPPlanError

WARMUP_DAYS = 400  # > 252 sessions for 12-month momentum and SMA200


def _ingest(args: argparse.Namespace) -> int:
    client = FMPClient(load_settings())
    status = 0
    for ticker in args.tickers:
        try:
            px = client.prices(ticker, args.start, args.end, refresh=args.refresh)
            print(f"{ticker}: {len(px)} price rows ({px.index.min():%Y-%m-%d} .. {px.index.max():%Y-%m-%d})"
                  if len(px) else f"{ticker}: no price rows")
            for kind in STATEMENT_PATHS:
                st = client.statements(ticker, kind, args.period, refresh=args.refresh)
                print(f"  {kind}_{args.period}: {len(st)} periods")
        except FMPPlanError as e:
            print(f"{ticker}: not available on the current FMP plan: {e}", file=sys.stderr)
            status = 1
        except FMPError as e:
            print(f"{ticker}: {e}", file=sys.stderr)
            status = 1
    return status


def _jev_strategy(settings: Settings, client: FMPClient, tickers: list[str], offline: bool, max_alloc: float,
                  entry_mode: str = "action"):
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
    strategy = JevStrategy(build_graph(summarizer, decider), ratios, JevRules(entry_mode=entry_mode), max_alloc)
    return strategy, summarizer, decider


def _load(args: argparse.Namespace, jev: bool):
    """Prices with warm-up (+ Jev strategy) for the requested tickers, from the FMP cache when possible."""
    settings = load_settings()
    client = FMPClient(settings)
    tickers = [t.upper() for t in args.tickers]
    max_alloc = args.max_alloc or 1 / len(tickers)
    warm_start = (pd.Timestamp(args.start) - pd.Timedelta(days=WARMUP_DAYS)).date()
    prices = {}
    for tk in tickers:
        try:
            prices[tk] = client.prices(tk, warm_start, args.end)
        except FMPError as e:
            raise type(e)(f"{tk}: {e}") from None
    parts =(_jev_strategy(settings, client, tickers, args.offline, max_alloc, getattr(args, "entry_mode", "action"))
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
    from jevbt.strategy import BaselineStrategy

    try:
        settings, tickers, max_alloc, prices, (strategy, summarizer, decider) = _load(args, args.strategy == "jev")
    except FMPError as e:
        print(e, file=sys.stderr)
        return 1
    if strategy is None:
        strategy = BaselineStrategy(max_alloc=max_alloc)
        name = "baseline"
    else:
        name = f"jev-{args.entry_mode}"
    run_dir = _run_dir(settings, name, args.offline)
    result = run_backtest(prices, strategy, args.start, args.end, cost_bps=args.cost_bps,
                          log_path=run_dir / "decisions.jsonl")
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
         "metrics": metrics}, indent=2))

    print(pd.DataFrame(metrics).T.to_string(float_format=_fmt))
    if summarizer is not None:
        print(f"new OpenAI summaries: {summarizer.calls}, new Jev calls: {decider.calls}")
    print(f"run saved to {run_dir}")
    return 0


def _walkforward(args: argparse.Namespace) -> int:
    from jevbt.backtest.engine import buy_and_hold
    from jevbt.backtest.metrics import summarize
    from jevbt.backtest.walkforward import walk_forward

    try:
        settings, tickers, max_alloc, prices, (strategy, summarizer, decider) = _load(args, True)
    except FMPError as e:
        print(e, file=sys.stderr)
        return 1
    wf = walk_forward(prices, strategy, args.start, args.end, train_months=args.train_months,
                      test_months=args.test_months, cost_bps=args.cost_bps, min_trades=args.min_trades)
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
         "max_alloc": max_alloc,
         "cost_bps": args.cost_bps, "offline": args.offline, "metrics": metrics}, indent=2))
    with pd.option_context("display.width", 220):
        print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print(f"\nOut-of-sample {oos_start:%Y-%m-%d} .. {oos_end:%Y-%m-%d}:")
        print(pd.DataFrame(metrics).T.to_string(float_format=_fmt))
    print(f"new OpenAI summaries: {summarizer.calls}, new Jev calls: {decider.calls}")
    print(f"run saved to {run_dir}")
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

    ingest = sub.add_parser("ingest", help="download prices and statements from FMP into the Parquet cache")
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
        if name == "backtest":
            p.add_argument("--strategy", choices=["jev", "baseline"], default="jev")
            p.add_argument("--entry-mode", choices=["action", "signals"], default="action")
        if name == "walkforward":
            p.add_argument("--train-months", type=int, default=12)
            p.add_argument("--test-months", type=int, default=3)
            p.add_argument("--min-trades", type=int, default=None,
                           help="min trades in a train window for a rule set to be eligible (default 2 per ticker)")
        if name != "baseline":
            p.add_argument("--offline", action="store_true", help="mock OpenAI and Jev (no paid calls)")

    research = sub.add_parser("research", help="research agent (OpenAI + FMP MCP) that proposes tickers")
    research.add_argument("criteria", help="what kind of companies to look for")
    research.add_argument("--max-tickers", type=int, default=10)
    research.add_argument("--seed", nargs="*", default=None, help="seed tickers to expand with peers")

    serve = sub.add_parser("serve", help="HTTP API for the React UI (ui/); JEVBT_RESEARCH_MOCK=1 for mock runs")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    if args.command == "ingest":
        return _ingest(args)
    if args.command == "research":
        return _research(args)
    if args.command == "serve":
        return _serve(args)
    if args.command == "walkforward":
        return _walkforward(args)
    if args.command == "baseline":
        args.strategy, args.offline = "baseline", False
    return _backtest(args)
