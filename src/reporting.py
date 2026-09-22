"""
Reporting, JSON serialization, and tidy CSV export utilities
for EB-NeRD Phase 1 validation outcome analysis.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR
from src.data import js


def export_json_artifact(data: Dict[str, Any], filepath: Path | str) -> None:
    """Save dictionary to JSON with indentation and clean type conversion."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(js(data), f, indent=2)


def export_tidy_csv(df: pd.DataFrame, filepath: Path | str) -> None:
    """Save tidy table to CSV."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def format_stage_effects_summary(
    avail_boot: Any,
    choice_boot: Any,
    overall_boot: Any,
) -> Dict[str, Any]:
    """Format stage effects results for stage_effects.json."""
    return {
        "available_to_exposed": {
            "point_estimate": avail_boot.point_estimate,
            "ci_95": [avail_boot.ci_lower, avail_boot.ci_upper],
            "std_error": avail_boot.std_error,
            "between_user_sd": avail_boot.between_user_sd,
            "standardized_effect": avail_boot.standardized_effect,
            "practical_significance": avail_boot.practical_significance,
            "n_users": avail_boot.n_users,
        },
        "exposed_to_consumed": {
            "point_estimate": choice_boot.point_estimate,
            "ci_95": [choice_boot.ci_lower, choice_boot.ci_upper],
            "std_error": choice_boot.std_error,
            "between_user_sd": choice_boot.between_user_sd,
            "standardized_effect": choice_boot.standardized_effect,
            "practical_significance": choice_boot.practical_significance,
            "n_users": choice_boot.n_users,
        },
        "overall_pipeline": {
            "point_estimate": overall_boot.point_estimate,
            "ci_95": [overall_boot.ci_lower, overall_boot.ci_upper],
            "std_error": overall_boot.std_error,
            "between_user_sd": overall_boot.between_user_sd,
            "standardized_effect": overall_boot.standardized_effect,
            "practical_significance": overall_boot.practical_significance,
            "n_users": overall_boot.n_users,
        },
    }
