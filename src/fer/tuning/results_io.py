"""Saves and formats grid-search results for console/CSV output."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from fer.tuning.trial_result import TrialResult


def results_to_dataframe(results: list[TrialResult]) -> pd.DataFrame:
    """Convert a list of TrialResults into a table, in the order given.

    Args:
        results: Trial results, as returned by
            fer.tuning.grid_search.run_grid_search (already sorted best-first).

    Returns:
        A DataFrame with columns lr, weight_decay, val_f1_macro,
        val_accuracy, epochs_trained.
    """
    return pd.DataFrame([dataclasses.asdict(r) for r in results])


def save_results_csv(results: list[TrialResult], path: Path) -> None:
    """Save grid-search results to a CSV file, creating parent directories as needed.

    Args:
        results: Trial results, as returned by
            fer.tuning.grid_search.run_grid_search.
        path: Destination CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    results_to_dataframe(results).to_csv(path, index=False)


def format_results_table(results: list[TrialResult]) -> str:
    """Render grid-search results as a plain-text table for console output.

    Args:
        results: Trial results, as returned by
            fer.tuning.grid_search.run_grid_search.

    Returns:
        A formatted string table.
    """
    return results_to_dataframe(results).to_string(index=False)
