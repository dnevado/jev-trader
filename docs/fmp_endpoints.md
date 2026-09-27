# FMP "stable" endpoints — verified 2026-09-27 (free plan, symbol AAPL)

Base URL: `https://financialmodelingprep.com/stable`. Auth: `apikey` query parameter.
Raw responses: `data/probe/*.json` (not committed). Probe: `scripts/probe_fmp.py`.

| Data | Endpoint | Params | Free plan result |
|---|---|---|---|
| EOD prices | `historical-price-eod/full` | `symbol, from, to` | 200 OK |
| EOD prices (div-adjusted) | `historical-price-eod/dividend-adjusted` | `symbol, from, to` | 200 OK |
| Income statement | `income-statement` | `symbol, period=quarter\|annual, limit` | 200 only with `limit <= 5`; `limit > 5` → 402 |
| Balance sheet | `balance-sheet-statement` | same | same |
| Cash flow | `cash-flow-statement` | same | same |
| Transcript dates | `earning-call-transcript-dates` | `symbol` | **402 Restricted Endpoint** |
| Transcript | `earning-call-transcript` | `symbol, year, quarter` | **402 Restricted Endpoint** |

Free plan limits observed:
- Statements: at most the **latest 5 periods** (5 quarters ≈ 15 months, or 5 fiscal years). `page=1` is ignored
  (it returns the same latest rows), so there is no way to go further back.
- Transcripts: not available at all.
- Plan errors come back as HTTP 402 with a plain-text body (not JSON).

## Fields

EOD prices: `symbol, date, open, high, low, close, volume, change, changePercent, vwap`.
Prices are **split-adjusted** (verified on NVDA's 10:1 split, 2024-06-10: no jump), not dividend-adjusted.
`historical-price-eod/dividend-adjusted` (`adjOpen, adjHigh, adjLow, adjClose, volume`) also works on the free plan,
if total-return prices are ever needed.
Statements are **restated for splits** too (NVDA FY2024 `epsDiluted` = 1.19, filed as 11.93 before the 10:1 split),
so `close / epsDiluted` is consistent with the split-adjusted prices.

Statements (all three) share these header fields:
`date` (period end), `symbol, reportedCurrency, cik,`
**`filingDate`** (`YYYY-MM-DD`), **`acceptedDate`** (`YYYY-MM-DD HH:MM:SS`), `fiscalYear` (string), `period` (`Q1..Q4` / `FY`).
The legacy typo `fillingDate` does **not** appear in the stable API (still renamed defensively on ingest).

Fields needed for the ratios in CLAUDE.md §6:
- Income: `revenue, grossProfit, operatingIncome, netIncome, eps, epsDiluted, weightedAverageShsOutDil`
- Balance: `totalDebt, totalStockholdersEquity, totalEquity, netDebt, cashAndCashEquivalents`
- Cash flow: `operatingCashFlow, capitalExpenditure, freeCashFlow`

Anti look-ahead: use `acceptedDate` (has a timestamp, often before market open on `filingDate`) → a statement
is usable at rebalance date `t` only if `acceptedDate < t`.

## Official MCP server (research agent) — verified 2026-09-27, free plan

URL `https://financialmodelingprep.com/mcp?apikey=...`, transport `streamable_http`. 28 tools, each with an
`endpoint` argument (enum listed in the tool description). Probe: `scripts/probe_fmp_mcp.py`.
Errors and plan denials come back as text results (`MCP error -32602 ...`, `ACCESS DENIED ...`), not exceptions.

| Tool | Free plan |
|---|---|
| `company` (`profile-symbol`, `peers`, ...) | OK |
| `statements` (`key-metrics`, `income-statement-growth`, ...) | OK |
| `marketPerformance` (`sector-performance-snapshot`, `most-active`, ...) | OK |
| `search` (incl. `search-company-screener`) | **ACCESS DENIED** (Starter+) |
| `indexes` (S&P 500 / Nasdaq / Dow constituents) | **ACCESS DENIED** (Premium+) |
| `earningsTranscript`, `ESG`, `form13F` | Ultimate+ (per description) |
| `technicalIndicators` | Starter+ (per description) |

Each tool call presumably spends the same 250/day quota; the research agent counts it in `data/raw/_budget.json`.
