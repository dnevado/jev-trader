# Project: jev-backtest

Simple Python trading system to **validate strategies via backtesting** on US stocks,
combining technical indicators (pandas), fundamental analysis with OpenAI (GPT mini/full models) and
typed decisions with **Jev (TypeSafe AI)**.

This file summarizes the previous design session (claude.ai, 27-Sep-2026). Read it in full before writing code.
Working language with the user: **English**. User environment: **Windows** (PowerShell).

---

## 1. Architecture decisions (already made)

1. **Data ingestion: direct REST APIs, NOT MCP.** The backtest needs bulk, deterministic and reproducible
   downloads. Everything is cached locally as Parquet.
   - **Stock prices (daily OHLCV bars): Alpaca Market Data API v2** (`https://data.alpaca.markets/v2/stocks/bars`,
     `timeframe=1Day`, `adjustment=split`, paginate with `next_page_token`). Auth headers
     `APCA-API-KEY-ID` / `APCA-API-SECRET-KEY`. Free (Basic) plan: 200 requests/min, history since 2016.
     Default `feed=sip` (consolidated tape; the free plan only blocks SIP data < 15 min old, and we never ask
     for today's bar); `iex` is IEX-only volume/prices. Keep the feed fixed for reproducibility (it is part of
     the cache path).
   - **Fundamentals (financial statements, company profile/peers): FMP REST** (Alpaca has no fundamentals).
     FMP free plan: 250 requests/day, 5 years of data, US exchanges only → caching is mandatory.
2. **FMP MCP only in "research" mode** (separate from the backtest): LangGraph + OpenAI (full model) agent
   with the official MCP tools (`https://financialmodelingprep.com/mcp?apikey=...`)
   via `langchain-mcp-adapters`, to propose the ticker universe. Never inside the backtest
   (it would not be reproducible). Motto: *MCP to explore and choose, REST to measure.*
3. **Technical indicators: pandas**, never an LLM (SMA50/200, RSI14, 3m/12m momentum, ATR/price, drawdown).
4. **OpenAI (mini model by default, full model optional) only for fundamentals**: summarizes ratios + an excerpt of the
   latest earnings call into structured JSON (`with_structured_output`, Pydantic).
   - Via **OpenAI API key** with `langchain-openai` (`ChatOpenAI`), NOT via OpenRouter.
   - Models (configurable via `.env`): `OPENAI_MODEL_FAST=gpt-4.1-mini`, `OPENAI_MODEL_FULL=gpt-4.1`. `temperature=0`
     (reasoning models such as the GPT-5 family ignore/reject `temperature` → verify before switching).
   - Cache the summary **per ticker + annual filing** (free FMP plan → annual statements, see §8), not per day
     → ~5 calls per ticker.
   - OpenAI prompt caching is automatic (prompts ≥1024 tokens): keep the fixed system prompt first;
     Batch API (−50%) for the historical backfill.
5. **Jev = the decision-maker, NOT the backtest engine.** The backtest is done by our own Python.
6. **Jev via OpenRouter** with `langchain-typesafe` (`TypeSafeClassifier`), redirecting the base URL.
   Pinned model **`typesafe/jev-1.13`** (reproducibility). Do NOT use `~typesafe/jev-latest` in backtests.
   Do NOT use `typesafe/jev-router` (it is a chat-model router, something else).
   Jev does NOT go through chat completions: the SDK calls `https://openrouter.ai/api/v1/systemone`.
7. **Orchestration: LangGraph** for the per-(ticker, date) graph. The backtest loop is plain Python.
8. **Anti look-ahead (critical):**
   - Filter financial statements by **publication date** (filing/accepted date) < t, not by quarter-end date.
   - Compute ratios ourselves with pandas from annual statements; do NOT use FMP's "current" TTM endpoints.
   - A fiscal year is usable at t only when all three statements have `acceptedDate < t`.
   - Previous day's price for P/E; execute orders at the next session's open.
   - Summarizer prompt: "use only the given documents, no external knowledge".
   - The Jev state and the summarizer prompt are **anonymized** ("the company", no ticker, no rebalance date).
   - Residual risk: LLMs know the past from training → validate mainly on recent
     periods / forward paper testing.
9. **Mandatory baseline**: strategy without LLM (price > SMA200 and RSI < 70). If LLM+Jev do not improve
   Sharpe / max drawdown versus the baseline, they add no value.

## 2. Architecture

```
┌─ Research (MCP, optional) ──────┐
│ GPT (full) + FMP MCP → tickers  │
└─────────────────────────────────┘
                ▼
Alpaca REST (prices) + FMP REST (statements) ──► data/raw/*.parquet (cache)
                │
   for each (ticker, rebalance date t):
   ├─ technical indicators (pandas, data ≤ t-1)
   ├─ point-in-time annual ratios (pandas, acceptedDate < t) + P/E with close t-1
   │      └─► GPT mini (ChatOpenAI) → FundSummary JSON (cached per annual filing)
   ├─ build_state → compact text (< 32K tokens, Jev 1.13 limit)
   └─ Jev (TypeSafeClassifier, OpenRouter) → Noul/Score/Choice + confidence (disk cache)
                ▼
   Python rules (thresholds, confidence-based sizing) → own backtest engine → metrics
```

## 3. Configuration (.env)

```
ALPACA_API_KEY_ID=...
ALPACA_API_SECRET_KEY=...
ALPACA_DATA_FEED=sip              # sip (consolidated, >15 min old on free plan) | iex
FMP_API_KEY=...                   # fundamentals only
OPENAI_API_KEY=sk-...
OPENAI_MODEL_FAST=gpt-4.1-mini
OPENAI_MODEL_FULL=gpt-4.1
TYPESAFE_BASE_URL=https://openrouter.ai/api
TYPESAFE_API_KEY=sk-or-...        # OpenRouter key, only for Jev (falls back to OPENROUTER_API_KEY)
JEV_MODEL=typesafe/jev-1.13
```

Dependencies: `pandas pyarrow requests python-dotenv pydantic langchain-openai langchain-typesafe langgraph`
(+ `langchain-mcp-adapters` for research, `pytest` for tests).
`langchain-typesafe` is pre-release (0.0.1a2) and its API has already changed → **pin the version** in requirements.
Fallback if it fails: official SDK `typesafe-sdk` (`TypeSafeClient(...).system_one(state=..., model=..., questions=...)`).

## 4. langchain-typesafe API (verified against the installed 0.0.1a2 source + a live call, 27-Sep-2026)

```python
from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier
clf = TypeSafeClassifier(
    questions={
        "trend_up": Noul(instructions="..."),
        "action":   Choice(instructions="...", criteria={"buy": "...", "hold": "...", "sell": "..."}),
        "quality":  Score(instructions="...", criteria=["weak", "medium", "strong"]),
    },
    model="typesafe/jev-1.13",
    api_key="sk-or-...",                  # default: TYPESAFE_API_KEY
    base_url="https://openrouter.ai/api", # default: TYPESAFE_BASE_URL or https://api.typesafe.ai; posts to /v1/systemone
)
r = clf.invoke("...state text...")       # state: str, JSON object/array or LangChain messages
r.nouls["trend_up"].noul                 # prob. 0-1
r.choices["action"].choice               # "buy" | "hold" | "sell"
r.choices["action"].confidence           # concentration of the distribution, NOT p(choice)
r.choices["action"].probabilities        # dict label -> prob
r.scores["quality"].score                # EXPECTED level (float, e.g. 1.35), not an index
r.scores["quality"].legend               # {0: "weak", 1: "medium", 2: "strong"}
r.scores["quality"].probabilities, .confidence
r.model, r.usage, r.request_id           # model comes back dated, e.g. "typesafe/jev-1.13-20260917"
```
`TypeSafeClassifier` is marked beta (LangChainBetaWarning). Tests inject `clf.client = httpx2.Client(transport=httpx2.MockTransport(...))`.

## 5. Initial example strategy: "Momentum + quality" (4-week horizon)

Questions for Jev (atomic, a single call):
- `trend_up` Noul: Is the medium-term trend clearly bullish?
- `overbought` Noul: Is it overbought with risk of a short-term correction?
- `fundamental_quality` Score: weak / medium / strong
- `valuation_risk` Noul: Does the valuation pose a relevant risk within 4 weeks?
- `action` Choice: buy / hold / sell

Entry rule (parameters to optimize), `JevRules.entry_mode`:
- `action` (original): `action==buy and conf>=0.70 and trend_up>=0.75 and overbought<0.50 and quality>=1.5`;
  size `p(buy) * (1 - 0.5*valuation_risk)`. In practice Jev almost never answers `buy` (AMZN 2026: max p(buy) 0.37).
- `signals`: same thresholds on trend_up / overbought / quality, vetoed by `action==sell`;
  size `trend_up * (1 - 0.5*valuation_risk)`.
(`quality` is the expected Score level: 0 weak, 1 medium, 2 strong → 1.5 ≈ "strong"). Code: `strategy.JevRules`.
Size × maximum allocation per position.
Exit: `action==sell or trend_up<0.40`. A held position with no exit keeps its weight (no resizing).
Baseline (`strategy.BaselineStrategy`): enter when close > SMA200 and RSI14 < 70; exit when close < SMA200.

Extensions (all off by default, so earlier runs reproduce; CLI flags on `backtest` / `baseline` / `walkforward`):
- Rule flags (`JevRules`): `use_overbought`, `sell_veto`, `exit_on_sell`, fixed `min_quality` / `valuation_penalty`
  (`--no-overbought --no-sell-veto --no-exit-on-sell --min-quality 0 --valuation-penalty 0` = preset
  `JEV_LOOSE_LONG`; the original signals rules with direction both = `JEV_STRICT_BOTH`).
- `--direction long|short|both`: shorts mirror the long rules (baseline: close < SMA200 and RSI > 30; Jev:
  trend_up ≤ 1 − threshold, quality ≤ 2 − min_quality, buy veto/cover on buy, size from 1 − trend_up and
  (1 − valuation_risk)); `both` flips close-then-open. Borrow fee `--borrow-bps` (default 30/yr).
- Engine exits, checked every session: `--trailing-stop` (fraction), `--trailing-stop-atr` (× ATR14 of t-1,
  ratchets), `--take-profit`; re-entry after a stop `--stop-rearm` (fresh signal) / `--stop-cooldown N` weeks.
  A strategy may set exits per position (`position_exits`) and lift re-entry blocks (`release_stop_block`).
- `backtest --strategy trend` (`TrendConfidenceStrategy` + `TrendRules`): trades only confident trends using
  SMA200 slope, persistence above/below the SMA200, efficiency ratio and ADX (indicators in `features/technical.py`,
  not part of the Jev state); `--vol-sizing`, `--jev-confirm-short`. Engine `--max-gross` caps gross exposure.
- `backtest --strategy regime`: `RegimeSwitchStrategy` (index close vs SMA200 at t-1: bull → long-only strategy,
  bear → long+short strategy), `--regime-index SPY|QQQ --bull baseline|jev --bear baseline|jev --bear-long-stop-atr`.

## 6. Summarizer prompt (fundamentals, OpenAI)

System (fixed, cached): fundamental analyst; use ONLY the documents; do not use own knowledge
or facts after the date; "no data" if something is missing; factual, no recommendations.

User (built by Python from a template):
```
Company: the company                      (anonymized)
Analysis date: {acceptedDate of the latest fiscal year}
(All the following documents predate this date.)
<ratios_annual>
table of the last ≤3 published fiscal years: revenue, revenue_growth_yoy, gross/operating/net margin,
debt_to_equity, fcf_margin, eps_diluted
</ratios_annual>
<earnings_call_extract>not available</earnings_call_extract>   (transcripts: 402 on the free plan)
Generate the structured summary.
```
Output (Pydantic `FundSummary`): growth, margins, balance_sheet, catalysts[], risks[], data_quality.
Valuation is NOT in the summary: P/E changes daily, so it goes straight into the Jev state (close t-1 / eps_diluted).

## 7. Planned code structure

```
jev-backtest/
  CLAUDE.md  README.md  requirements.txt  .env.example
  src/jevbt/
    config.py
    ingest/cache.py          # shared Parquet helpers (date-range cache with covered-range meta)
    ingest/alpaca.py         # daily price bars (Alpaca REST) + Parquet cache in data/raw/prices_alpaca/<feed>/
    ingest/fmp.py            # statements/profile (FMP REST) + Parquet cache; normalizes column names
    features/technical.py    # pandas indicators
    features/fundamentals.py # point_in_time(), compute_ratios(), build_user_prompt()
    llm/summarizer.py        # ChatOpenAI (mini), cached per (ticker, filing)
    decision/jev.py          # questions, client, disk cache by hash(state+questions+model), offline mock
    graph.py                 # LangGraph: fundamentals → build_state → decide
    strategy.py              # entry/exit/sizing rules
    backtest/engine.py       # weekly rebalance, execution at t+1 open, costs in bps, multi-ticker
    backtest/metrics.py      # CAGR, Sharpe, max DD, # trades, hit rate
    research/mcp_agent.py    # ChatOpenAI (full) + FMP MCP agent
    broker/alpaca_paper.py   # Alpaca Trading API, PAPER endpoint only (refuses any other host)
    paper.py                 # weekly forward paper-trading step (same strategy + gross cap as the backtest)
    aws_job.py               # scheduled AWS job (python -m jevbt.aws_job trade|report): orders + SNS emails
    cli.py                   # python -m jevbt ingest | backtest | baseline | walkforward | research | serve | paper
    api.py                   # FastAPI for the UI: POST/GET /api/research, GET /api/budget
  ui/                        # React + Vite stock-discovery UI (npm run dev → :5173, proxies /api → :8000)
  docker/                    # Dockerfile + requirements of the scheduled job image
  infra/terraform/           # ECS Fargate + EventBridge Scheduler + SNS + S3 + SSM (one workspace per AWS account)
  tests/                     # synthetic data + OpenAI and Jev mocks (no network)
```

## 8. Pending / next steps

1. ~~Verify FMP "stable" endpoints and field names~~ **Done 2026-09-27** → see `docs/fmp_endpoints.md`.
   Free-plan limits found:
   - Statements: only the **latest 5 periods** (`limit > 5` → HTTP 402; `page` is ignored) → 5 quarters
     (currently from mid-2025) or 5 fiscal years. The cache merges refreshes, so quarters accumulate over time.
   - Earnings-call transcripts (`earning-call-transcript`, `-dates`): **not available** (HTTP 402).
   - EOD prices were split-adjusted (not dividend-adjusted). **Prices now come from Alpaca** with
     `adjustment=split` to keep the same convention: statements are restated for splits too, so
     `close / eps_diluted` stays consistent. `acceptedDate` (with time) is the publication date (`acceptedDate < t`).
   - **Done 2026-10-02: price ingestion migrated to `ingest/alpaca.py`** (CLI `ingest`/`baseline`/`backtest`/
     `walkforward`). `FMPClient.prices` removed (FMP is fundamentals only). Results in item 5 below were computed with
     FMP prices; re-runs with Alpaca bars change the technical features in the Jev state → Jev cache misses
     → those calls are paid again.
2. ~~Implement the structure in §7, with offline tests~~ **Phase 1 done**: `ingest/fmp.py` (Parquet cache,
   daily budget counter), `features/technical.py`, offline tests; other modules are stubs.
3. ~~Decide the fundamentals source~~ **Decided: annual statements** (free FMP plan, last 5 fiscal years).
   First usable YoY growth ≈ late 2022 → backtest from **2023-01-01**. No transcripts.
   **Phase 2 done**: fundamentals, summarizer, Jev client (OpenRouter), LangGraph, strategies, engine, metrics,
   CLI `baseline` / `backtest [--offline]`, decision log (`data/runs/*/decisions.jsonl`), offline tests.
   Smoke test OK: 1 real OpenAI summary + 1 real Jev call (AAPL, 2025-06-02).
4. ~~First real run~~ Done 2026-09-27: AMZN, 2026-04-01..2026-09-25 (user's choice).
   Jev `action` rule: 0 trades; `signals` rule: −8.2%; baseline: −23.4%; buy & hold: +18.6%.
   Cost: ~1 Jev call per (ticker, week) → ~195 per ticker for 2023-01..2026-09 (cached afterwards).
5. ~~Walk-forward~~ Done: `backtest/walkforward.py`, CLI `walkforward` (train 12m / test 3m, grid of 32 `JevRules`,
   best train Sharpe; open positions sold at each test window's close). Jev answers are memoized per (ticker, t),
   so the grid search costs no extra calls.
   AMZN out-of-sample 2024-01..2026-09: Jev −15.4% (Sharpe −0.62, max DD −24%), baseline −19.8% (Sharpe −0.18,
   max DD −40%), buy & hold +66.5% (Sharpe 0.74). The grid picks `signals` in 10/11 folds.
   → On AMZN, Jev does not beat the baseline on Sharpe (only on drawdown) and both lose to buy & hold.
   5 tickers (AMZN MSFT AAPL GOOGL NVDA; ORCL prices were 402 on the FMP free plan — no longer an issue with Alpaca), OOS 2024-01..2026-09:
   Jev +14.9% (Sharpe 1.00, max DD −5.2%, 102 trades, hit 43%), baseline +61.7% (Sharpe 1.05, max DD −18.2%),
   buy & hold +140.5% (Sharpe 1.34, max DD −28.7%). Tickers picked today → hindsight bias favours buy & hold.
   Weakness found: the last 4 folds picked `action` rules with train Sharpe 2.70 from very few trades and then
   made 0 trades out-of-sample → fixed: `walk_forward(min_trades=...)`, default 2 per ticker in each train window
   (CLI `--min-trades`). Re-run (all from cache): Jev +13.1% (Sharpe 0.72, max DD −7.0%, 146 trades, hit 40%);
   `signals` chosen in all 11 folds; the 4 formerly flat folds now trade but return −3%..+1%.
   **Conclusion so far: Jev rules do not beat the no-LLM baseline on Sharpe or return; they only cut drawdown,
   mostly through low exposure.** Next ideas: reword the `action` Choice
   (changes the Jev cache key → all calls are paid again).
6. ~~Log every decision~~ Done: `decisions.jsonl` per run (state sent, Jev answers, resulting order).
7. ~~Research agent~~ **Phase 3 done**: `research/mcp_agent.py`, CLI `research "<criteria>" [--seed ...]`.
   Free plan: screener (`search`) and index constituents (`indexes`) are denied → the agent expands seed tickers
   with `company` peers and checks them with `statements`; only allowed tools are exposed, max 25 tool calls,
   every call counted in the FMP daily budget. Output in `data/research/*.json`.
   Caveat: tickers chosen with today's data → backtesting them over the past has hindsight/survivorship bias.
8. ~~Discovery UI~~ Done: React sample in `ui/` + `api.py` (`python -m jevbt serve`; `JEVBT_RESEARCH_MOCK=1` for canned runs without paid calls).
9. **Experiments on Alpaca prices, 2026-10-02/03** (tickers AMD AMZN AAPL AAL; Jev answers cached for 2023-01..2026-09).
   All results below assume the cached Jev answers; a backtest only hits the cache when the price history starts
   at 2023-01-01 − 400 days (recursive RSI/ATR change the state text otherwise → new paid calls).
   - Walk-forward OOS 2024-01..2026-09: AAPL Jev +11% / baseline +69% / B&H +84%; AMD Jev +27% / baseline +419%
     / B&H +355%. AMD fundamentals look weak (GAAP margins hit by ~$3-4B/yr Xilinx amortization → quality < 1,
     GAAP P/E 160-260x → valuation_risk ~0.75, `sell` ~50% of weeks), which blocked longs through the rally.
   - Removing gates step by step improves Jev every time (AMD +27% → +132%, AAPL +11% → +35% with
     `JEV_LOOSE_LONG`), i.e. the fundamentals answers hurt longs; Jev then acts as a noisier trend filter.
     4-ticker portfolio long-only: Jev loose Sharpe 1.01 vs baseline 0.94 (return +42% vs +64%).
   - Shorts / long+short: hurt in the 2024-26 rise (baseline both: whipsaw, AAL −84%, AMZN −56%); pay off in
     declines (2022, Dec-2024..Apr-2025 correction, AMD 2024-25: baseline short +24..41%; Jev strict long+short
     +12% / +32% with Sharpe 3.3 / 1.1, beating baseline long+short in declines).
   - Stops: fixed 3% destroys returns (85% of stop-outs re-entered within a week, costs 12.6% of capital);
     ATR 2-3× and fresh-signal re-entry are better but still below no-stop in rises; in declines stops help the
     long side and wipe out short gains. Stops make sense only for long+short (baseline both 2×ATR+rearm:
     +17% Sharpe 0.51 DD −12% vs +12% / 0.29 / −28% without).
   - Market-regime switch (SPY/QQQ vs SMA200) fails: the index regime turns bear after the drop and bull after the
     rebound (every 2023-26 bear span ended higher than it started), and misses stock-specific declines (AMD fell
     62% with the market 93% bull). 2023-26: regime variants +30% at best vs baseline long +109%.
   - **Conclusion**: nothing beat buy & hold (+312% 2023-26). Baseline long is the best active strategy in rises;
     Jev's value is per-stock and defensive (strict long+short in declines, smallest drawdowns long-only).
     Rules were chosen after seeing 2024-26 → partly in-sample; 4 tickers. Next: forward paper testing; if
     anything, a per-stock regime from Jev's bearish answers; better fundamentals inputs (EBITDA-like margin,
     non-GAAP valuation) would change the Jev state → all calls paid again.

## 9. Conventions

- Windows: use `pathlib`, no hard-coded `/` paths; PowerShell commands in the README.
- Never commit `.env` or `data/`.
- Every paid/rate-limited call (OpenAI, Jev, FMP, Alpaca) goes through the disk cache.
