# jev-backtest

Backtesting of US stock strategies (prices from Alpaca, fundamentals from FMP): pandas technical indicators, OpenAI fundamentals summaries and Jev (TypeSafe AI) decisions.
Design and decisions: [CLAUDE.md](CLAUDE.md). Verified FMP endpoints and free-plan limits: [docs/fmp_endpoints.md](docs/fmp_endpoints.md).

## Setup (PowerShell)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt -e .
Copy-Item .env.example .env   # fill ALPACA_API_KEY_ID/SECRET_KEY and FMP_API_KEY (a parent-folder .env is also picked up)
```

## Usage

```powershell
.venv\Scripts\python -m pytest -q                      # offline tests, no network
.venv\Scripts\python scripts\probe_fmp.py              # re-verify FMP endpoints (~7 requests)
.venv\Scripts\python -m jevbt ingest --tickers AAPL MSFT --start 2021-01-01 --end 2025-12-31
.venv\Scripts\python -m jevbt baseline --tickers AAPL MSFT --start 2023-01-01 --end 2026-09-25
.venv\Scripts\python -m jevbt backtest --tickers AAPL MSFT --start 2023-01-01 --end 2026-09-25 --offline  # mocks, no paid calls
.venv\Scripts\python -m jevbt backtest --tickers AAPL MSFT --start 2023-01-01 --end 2026-09-25            # OpenAI + Jev
.venv\Scripts\python -m jevbt backtest --tickers AMZN --start 2026-04-01 --end 2026-09-25 --entry-mode signals
.venv\Scripts\python -m jevbt walkforward --tickers AMZN --start 2023-01-01 --end 2026-09-25        # out-of-sample
.venv\Scripts\python -m jevbt walkforward --tickers AMD AAPL --start 2023-01-01 --end 2026-09-25 --direction both  # long+short
.venv\Scripts\python -m jevbt baseline --tickers AMD --start 2023-01-01 --end 2026-09-25 --direction both --trailing-stop-atr 2 --stop-rearm
.venv\Scripts\python -m jevbt backtest --strategy regime --regime-index QQQ --bull baseline --bear jev --tickers AMD AAPL --start 2023-01-01 --end 2026-09-25
.venv\Scripts\python -m jevbt research "profitable US large caps with revenue growth" --seed AMZN MSFT  # FMP MCP
.venv\Scripts\python scripts\probe_fmp_mcp.py                                                        # list MCP tools
```

Each backtest writes `data/runs/<timestamp>_<strategy>/`: `metrics.json`, `equity.csv`, `trades.csv` and
`decisions.jsonl` (state sent to Jev, its answers and the resulting order). OpenAI summaries and Jev answers
are cached in `data/cache/`, so re-running the same backtest costs nothing.

Data is cached in `data/raw/` (Parquet). Cached ranges never hit the network.
- Prices: Alpaca daily bars, split-adjusted, in `data/raw/prices_alpaca/<feed>/` (free plan: 200 requests/min).
- Statements: FMP; the daily request counter lives in `data/raw/_budget.json` (`FMP_DAILY_BUDGET`, default 240
  of the 250 free requests). The old FMP price cache (`data/raw/prices/`) is no longer used and can be deleted.

## Forward paper trading (Alpaca paper account)

Weekly step with the same strategy code as the backtest; only the paper endpoint is ever used.
Run it on the first session of the week before 09:28 ET (market-on-open orders, like the backtest):

```powershell
.venv\Scripts\python -m jevbt paper --tickers AMD NKE XOM KO                  # dry run: prints and logs orders
.venv\Scripts\python -m jevbt paper --tickers AMD NKE XOM KO --submit         # sends them to the paper account
```

Defaults: `--strategy trend --direction long --max-alloc 1/12 --max-gross 1.0 --tif opg`. Whole shares only;
a long/short flip is a close order plus an open order; shorts need a shortable, easy-to-borrow asset. Each run
is logged in `data/paper/`. The account starts flat, so a stock needs a full entry signal to be bought (the
backtest may already hold positions entered earlier).

## Stock discovery UI (React)

A small React + Vite app in `ui/` to prompt the research agent (OpenAI + FMP MCP) and browse past runs.
It talks to a thin FastAPI layer (`src/jevbt/api.py`).

```powershell
# terminal 1 — API on :8000 (add $env:JEVBT_RESEARCH_MOCK="1" first for canned answers, no FMP/OpenAI calls)
.venv\Scripts\python -m jevbt serve
# terminal 2 — UI dev server on http://localhost:5173 (proxies /api to :8000)
cd ui; npm install; npm run dev
```

Or build once (`cd ui; npm run build`) and open http://localhost:8000: the API serves `ui/dist` itself.

Endpoints: `POST /api/research` `{criteria, seeds, max_tickers}`, `GET /api/research`, `GET /api/research/{id}`,
`GET /api/budget`. Runs are saved to `data/research/` (same files as the `research` CLI command); one agent run
at a time (409 otherwise), 429 when the FMP daily budget is exhausted.
