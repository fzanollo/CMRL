"""Loading of OTF-DCS comparison results produced by ``run_mtsa.py``.

All heuristics (RA, the prior RL baseline, and the CMRL agents) are written
by ``run_mtsa.py`` into a single ``comparative_results_all.csv`` per
(run, problem), so a single loader is enough for every script in this
package. RA and the prior monolithic-derived baselines do not depend on
training randomness, so they are only present in ``run_1``; the merge
functions in ``gains_core.py`` account for that automatically via inner
joins on ``run_id``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

from . import config


def load_run_problem(results_root: Path, run_id: int, problem: str) -> pd.DataFrame:
    """Load one (run, problem) comparative_results_all.csv, if it exists."""
    csv_path = results_root / f"run_{run_id}" / "results" / problem / "comparative_results_all.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df["run_id"] = run_id
    df["problem"] = problem
    return df


def load_all_runs(
    results_root: Path = config.DEFAULT_RESULTS_ROOT,
    problems: Optional[List[str]] = None,
    runs: int = config.DEFAULT_RUNS,
    only_ok: bool = True,
) -> pd.DataFrame:
    """Load and concatenate results for every (run, problem) combination.

    Returns a long dataframe with one row per (run, problem, Model,
    instance_name) and numeric metric columns coerced from strings.
    """
    problems = problems or config.DEFAULT_PROBLEMS

    frames = [
        load_run_problem(results_root, run_id, problem)
        for run_id in range(1, runs + 1)
        for problem in problems
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise FileNotFoundError(
            f"No comparative_results_all.csv found under {results_root} "
            f"for problems={problems}, runs=1..{runs}"
        )

    df = pd.concat(frames, ignore_index=True)

    for metric in config.METRICS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

    if only_ok and "result" in df.columns:
        n_before = len(df)
        df = df[df["result"] == "OK"].copy()
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"[data_loading] Dropped {n_dropped} non-OK rows (timeouts/errors).")

    return df


def filter_heuristics(df: pd.DataFrame, heuristic_keys: List[str]) -> pd.DataFrame:
    """Keep only the rows whose Model matches one of the requested heuristics."""
    info = config.resolve_heuristics(heuristic_keys)
    models = {v["model"] for v in info.values()}
    return df[df["Model"].isin(models)].copy()
