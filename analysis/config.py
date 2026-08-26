"""Central configuration for the analysis / table-generation scripts.

Every script in this package reads its default paths and its default set of
heuristics from here, so adding a new heuristic or changing a folder layout
only requires touching this one file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, TypedDict

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RESULTS_ROOT = REPO_ROOT / "rl_results"
DEFAULT_RA_DIR = REPO_ROOT / "ra_results"
DEFAULT_MONOLITHIC_DIR = REPO_ROOT / "monolithic_results"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "output"

DEFAULT_PROBLEMS: List[str] = ["sokoban-4-1", "sokoban-5-1", "sokoban-6-1", "sokoban-7-1"]
DEFAULT_RUNS = 5

# Metrics available in comparative_results_all.csv that make sense to compare
# across heuristics (lower is better for all of them).
METRICS: List[str] = ["execution_time", "ExpandedStates", "ExpandedTransitions"]
DEFAULT_METRIC = "execution_time"


class HeuristicInfo(TypedDict):
    model: str      # exact value of the "Model" column in comparative_results_all.csv
    label: str      # display label used in plots/tables
    color: str
    marker: str


# Registry of every heuristic the pipelines know how to load/plot.
# Add new heuristics here (and to run_mtsa.py's output) rather than in
# individual scripts.
HEURISTICS: Dict[str, HeuristicInfo] = {
    "ra": {"model": "RA", "label": "Ready Abstraction", "color": "tab:green", "marker": "D"},
    "prev_rl": {"model": "Previous RL", "label": "Delgado et al.", "color": "tab:red", "marker": "s"},
    "cmrl_400e": {"model": "DQN Agent 400e", "label": "CMRL 400e", "color": "tab:blue", "marker": "o"},
    "cmrl_1000e": {"model": "DQN Agent 1000e", "label": "CMRL 1000e", "color": "tab:orange", "marker": "^"},
}

# Heuristics included by default whenever a script builds a table/plot.
# This is the "RA, CMRL, Delgado/previous RL" default the tables ship with;
# override per-invocation with --heuristics / --baseline / --agents.
DEFAULT_TABLE_HEURISTICS: List[str] = ["ra", "prev_rl", "cmrl_400e", "cmrl_1000e"]

# Default baseline/agents used for gain and win-rate computations (this is
# the pairing used in the paper's main comparison table).
DEFAULT_BASELINE = "prev_rl"
DEFAULT_AGENTS: List[str] = ["cmrl_400e", "cmrl_1000e"]


def resolve_heuristics(keys: List[str]) -> Dict[str, HeuristicInfo]:
    """Validate CLI heuristic keys and return {key: heuristic_info} in order."""
    unknown = [k for k in keys if k not in HEURISTICS]
    if unknown:
        raise ValueError(
            f"Unknown heuristic(s): {unknown}. Available: {list(HEURISTICS)}"
        )
    return {k: HEURISTICS[k] for k in keys}
