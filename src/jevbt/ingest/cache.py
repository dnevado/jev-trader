"""Parquet cache helpers shared by the REST clients."""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def load_range(path: Path, start: date, end: date, fetch: Callable[[date, date], pd.DataFrame],
               refresh: bool = False) -> pd.DataFrame:
    """Daily rows in [start, end] indexed by date, fetching only the days the cache does not cover yet.

    The covered range lives in `<path>.meta.json` (a range with no rows, e.g. holidays, still counts as covered).
    `fetch(a, b)` must return a frame with a `date` column for the inclusive range [a, b].
    """
    meta_path = path.with_suffix(".meta.json")
    cached = pd.read_parquet(path) if path.exists() and not refresh else None
    covered = json.loads(meta_path.read_text()) if cached is not None and meta_path.exists() else None

    if covered is None:
        missing = [(start, end)]
        cov_start, cov_end = start, end
    else:
        cov_start = date.fromisoformat(covered["start"])
        cov_end = date.fromisoformat(covered["end"])
        missing = []
        if start < cov_start:
            missing.append((start, cov_start - timedelta(days=1)))
        if end > cov_end:
            missing.append((cov_end + timedelta(days=1), end))
        cov_start, cov_end = min(start, cov_start), max(end, cov_end)

    if missing:
        frames = [] if cached is None else [cached]
        for a, b in missing:
            frames.append(fetch(a, b))
        frames = [f for f in frames if not f.empty]
        merged = (
            pd.concat(frames, ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
            if frames else pd.DataFrame(columns=["date"])
        )
        write_parquet(merged.reset_index(drop=True), path)
        meta_path.write_text(json.dumps({"start": cov_start.isoformat(), "end": cov_end.isoformat()}))
        cached = merged

    df = cached.set_index("date")
    return df.loc[pd.Timestamp(start):pd.Timestamp(end)]
