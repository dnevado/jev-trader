import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { api, type Budget, type ResearchRun, type ResearchSummary } from "./api";

const EXAMPLES = [
  "Profitable US large caps (market cap > 100B USD) with positive annual revenue growth and positive free cash flow",
  "US semiconductor companies with revenue growth above 20% and improving operating margins",
  "Low-debt US consumer staples with stable margins and positive free cash flow",
];

const BACKTEST_START = "2023-01-01"; // first date with annual YoY growth on the free FMP plan (CLAUDE.md §8)

function yesterday(): string {
  const d = new Date();
  d.setDate(d.getDate() - 1);
  return d.toISOString().slice(0, 10);
}

function parseSeeds(text: string): string[] {
  return text
    .split(/[\s,;]+/)
    .map((s) => s.trim().toUpperCase())
    .filter(Boolean);
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export default function App() {
  const [criteria, setCriteria] = useState(EXAMPLES[0]);
  const [seedsText, setSeedsText] = useState("AMZN MSFT");
  const [maxTickers, setMaxTickers] = useState(8);
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [run, setRun] = useState<ResearchRun | null>(null);
  const [history, setHistory] = useState<ResearchSummary[]>([]);
  const [budget, setBudget] = useState<Budget | null>(null);
  const timer = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [b, h] = await Promise.all([api.budget(), api.listResearch()]);
      setBudget(b);
      setHistory(h);
    } catch (e) {
      setError(`API not reachable: ${(e as Error).message}. Is "python -m jevbt serve" running?`);
    }
  }, []);

  useEffect(() => {
    void refresh();
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [refresh]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setRunning(true);
    setElapsed(0);
    const started = Date.now();
    timer.current = window.setInterval(() => setElapsed(Math.round((Date.now() - started) / 1000)), 1000);
    try {
      setRun(await api.runResearch({ criteria: criteria.trim(), seeds: parseSeeds(seedsText), max_tickers: maxTickers }));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      if (timer.current) window.clearInterval(timer.current);
      setRunning(false);
      void refresh();
    }
  }

  async function openRun(id: string) {
    setError(null);
    try {
      setRun(await api.getResearch(id));
    } catch (err) {
      setError((err as Error).message);
    }
  }

  const budgetPct = budget ? Math.min(100, (budget.count / budget.limit) * 100) : 0;

  return (
    <div className="page">
      <header className="topbar">
        <div>
          <h1>Stock discovery</h1>
          <p className="muted">Research agent · OpenAI + FMP MCP · proposes a ticker universe for the backtest</p>
        </div>
        {budget && (
          <div className="budget" title="FMP requests used today (REST + MCP), shared with the backtest">
            <span>
              FMP today: <strong>{budget.count}</strong> / {budget.limit}
            </span>
            <div className="meter">
              <div className={budgetPct > 80 ? "fill warn" : "fill"} style={{ width: `${budgetPct}%` }} />
            </div>
          </div>
        )}
      </header>

      <div className="layout">
        <main>
          <form className="card" onSubmit={onSubmit}>
            <label htmlFor="criteria">What kind of companies are you looking for?</label>
            <textarea
              id="criteria"
              rows={3}
              value={criteria}
              onChange={(e) => setCriteria(e.target.value)}
              minLength={3}
              maxLength={500}
              required
              disabled={running}
            />
            <div className="examples">
              {EXAMPLES.map((ex) => (
                <button type="button" key={ex} className="chip" onClick={() => setCriteria(ex)} disabled={running}>
                  {ex.length > 60 ? `${ex.slice(0, 57)}…` : ex}
                </button>
              ))}
            </div>
            <div className="row">
              <div className="field grow">
                <label htmlFor="seeds">Seed tickers (optional)</label>
                <input
                  id="seeds"
                  value={seedsText}
                  onChange={(e) => setSeedsText(e.target.value)}
                  placeholder="AMZN MSFT"
                  disabled={running}
                />
                <small className="muted">The free FMP plan has no screener: the agent expands seeds with peers.</small>
              </div>
              <div className="field">
                <label htmlFor="max">Max tickers</label>
                <input
                  id="max"
                  type="number"
                  min={1}
                  max={20}
                  value={maxTickers}
                  onChange={(e) => setMaxTickers(Number(e.target.value))}
                  disabled={running}
                />
              </div>
            </div>
            <div className="actions">
              <button type="submit" className="primary" disabled={running || criteria.trim().length < 3}>
                {running ? `Researching… ${elapsed}s` : "Discover stocks"}
              </button>
              <small className="muted">Up to 25 FMP requests and a few OpenAI calls per run.</small>
            </div>
          </form>

          {error && (
            <div className="card error" role="alert">
              {error}
            </div>
          )}

          {running && !run && <div className="card muted">The agent is calling FMP tools; this usually takes 30–90 s.</div>}

          {run && <ResultCard run={run} />}
        </main>

        <aside className="card history">
          <h2>Past runs</h2>
          {history.length === 0 && <p className="muted">No runs yet.</p>}
          <ul>
            {history.map((h) => (
              <li key={h.id}>
                <button className={run?.id === h.id ? "item active" : "item"} onClick={() => void openRun(h.id)}>
                  <span className="item-date">
                    {formatDate(h.created_at)} {h.mock && <span className="tag">mock</span>}
                  </span>
                  <span className="item-criteria">{h.criteria}</span>
                  <span className="item-tickers">{h.tickers.join(" ") || "—"}</span>
                </button>
              </li>
            ))}
          </ul>
        </aside>
      </div>
    </div>
  );
}

function ResultCard({ run }: { run: ResearchRun }) {
  const tickers = run.result.candidates.map((c) => c.ticker).join(" ");
  const commands = [
    { label: "Walk-forward (Jev)", cmd: `python -m jevbt walkforward --tickers ${tickers} --start ${BACKTEST_START} --end ${yesterday()}` },
    { label: "Baseline", cmd: `python -m jevbt baseline --tickers ${tickers} --start ${BACKTEST_START} --end ${yesterday()}` },
  ];

  return (
    <section className="card">
      <div className="result-head">
        <h2>
          {run.result.candidates.length} candidates {run.mock && <span className="tag">mock</span>}
        </h2>
        <span className="muted">{formatDate(run.created_at)}</span>
      </div>
      <p className="muted criteria">“{run.result.criteria}”</p>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Rationale (from the data the agent retrieved)</th>
            </tr>
          </thead>
          <tbody>
            {run.result.candidates.map((c) => (
              <tr key={c.ticker}>
                <td className="ticker">{c.ticker}</td>
                <td>{c.rationale}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {run.result.notes && (
        <details>
          <summary>Agent notes</summary>
          <p>{run.result.notes}</p>
        </details>
      )}

      <div className="warning">
        These tickers were chosen with <strong>today's</strong> data. Backtesting them over past periods is biased in
        their favour (hindsight / survivorship); prefer recent periods or forward testing. Some symbols may not have
        prices on the free FMP plan (e.g. ORCL).
      </div>

      {tickers && (
        <div className="commands">
          {commands.map(({ label, cmd }) => (
            <CommandLine key={label} label={label} cmd={cmd} />
          ))}
        </div>
      )}
    </section>
  );
}

function CommandLine({ label, cmd }: { label: string; cmd: string }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(cmd);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard not available (e.g. insecure context): the command stays selectable
    }
  }
  return (
    <div className="command">
      <span className="command-label">{label}</span>
      <code>{cmd}</code>
      <button type="button" onClick={() => void copy()}>
        {copied ? "Copied" : "Copy"}
      </button>
    </div>
  );
}
