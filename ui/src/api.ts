// Typed client for the Python API in src/jevbt/api.py.

export interface Candidate {
  ticker: string;
  rationale: string;
}

export interface ResearchResult {
  criteria: string;
  candidates: Candidate[];
  notes: string;
}

export interface ResearchRun {
  id: string;
  created_at: string;
  mock: boolean;
  result: ResearchResult;
}

export interface ResearchSummary {
  id: string;
  created_at: string;
  mock: boolean;
  criteria: string;
  tickers: string[];
}

export interface Budget {
  date: string | null;
  count: number;
  limit: number;
}

export interface ResearchRequest {
  criteria: string;
  seeds: string[];
  max_tickers: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      // non-JSON error body: keep the status line
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  budget: () => request<Budget>("/api/budget"),
  listResearch: () => request<ResearchSummary[]>("/api/research"),
  getResearch: (id: string) => request<ResearchRun>(`/api/research/${encodeURIComponent(id)}`),
  runResearch: (body: ResearchRequest) =>
    request<ResearchRun>("/api/research", { method: "POST", body: JSON.stringify(body) }),
};
