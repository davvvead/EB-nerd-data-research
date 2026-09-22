"""
Configuration and frozen parameters for the EB-NeRD Phase 1 analysis pipeline.
Source of truth: analysis_protocol_v1.yaml, analysis_protocol_notes.md, IMPLEMENTATION_DECISIONS.md
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

# Workspace Root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Frozen Audit Identifiers
PROTOCOL_COMMIT_HASH = "cb14d50736d1b8c8202b1aa67158e15c3eabb804"
PREOUTCOME_GATE_COMMIT_HASH = "db8d1003358f50d51e66672ca3a6a8e8f02d1c0a"
IMPLEMENTATION_LOCK_COMMIT_HASH = "083e93104483658b6418900770de7d357665c8e8"
PRIMARY_RESULTS_FREEZE_COMMIT_HASH = "598b3553346b9be36195b074530827448fdc8e4b"
SEED = 20260919
PROTOCOL_VERSION = "phase1-parameter-freeze-v1"

# Frozen Protocol Values (from analysis_protocol_v1.yaml)
MINIMUM_HISTORY_LENGTH = 15
ELIGIBLE_SLATE_SIZE_MIN = 5
ELIGIBLE_SLATE_SIZE_MAX = 26

PRIMARY_SUPPLY_WINDOW_HOURS = 24
SUPPLY_SENSITIVITIES_HOURS = [12, 48]
SUPPLY_RATIO_REQUIREMENT = 3.0

PRIMARY_SEMANTIC_REPRESENTATION = "contrastive_vector.parquet"
ROBUSTNESS_SEMANTIC_REPRESENTATION = "bert_base_multilingual_cased.parquet"
EMBEDDING_DIMENSION = 768

PRIMARY_HISTORY_K = 5
SENSITIVITY_HISTORY_K = [3, 10]

# Shannon entropy tertile cutpoints from train eligible users
BREADTH_TERTILE_LOW_TO_MED = 1.4638116962471364
BREADTH_TERTILE_MED_TO_HIGH = 1.5905145971043473

# History length quartile bins from train eligible users: [min, max]
HISTORY_LENGTH_BINS = [
    (15, 49),
    (50, 111),
    (112, 244),
    (245, float("inf")),
]

# Context matching parameters
MATCH_TIME_BUCKET_MINUTES = 30
SLATE_SIZE_CALIPER = 1
COMMON_SUPPORT_N_PRIMARY = 3
COMMON_SUPPORT_N_SENSITIVITY = 5

# Null slate parameters
NULL_DRAWS_PER_IMPRESSION = 200
POPULARITY_WINDOW_HOURS = 24
RECENCY_HALF_LIFE_HOURS = 12.0
ZERO_CLICK_PSEUDOCOUNT = 1.0

# Relevance model qualification thresholds
MIN_QUALIFICATION_AUC = 0.60
MIN_QUALIFICATION_NDCG_LIFT = 0.01

# Headroom constraints
HEADROOM_MAX_SWAPS = 2
HEADROOM_PRIMARY_RELEVANCE_TOLERANCE = 0.03
HEADROOM_SENSITIVITY_TOLERANCES = [0.01, 0.05]

# Bootstrap parameters
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
PRACTICAL_EFFECT_THRESHOLD_SD = 0.10

# Paths
CACHE_DIR = ROOT_DIR / "cache"
CACHE_TRAIN_DIR = CACHE_DIR / "train"
CACHE_VAL_DIR = CACHE_DIR / "validation"
CACHE_EMB_DIR = CACHE_DIR / "embeddings"
OUTPUTS_DIR = ROOT_DIR / "outputs"

TRAIN_HISTORY_PATH = ROOT_DIR / "ebnerd_small" / "train" / "history.parquet"
VALIDATION_HISTORY_PATH = ROOT_DIR / "ebnerd_small" / "validation" / "history.parquet"
VALIDATION_BEHAVIORS_PATH = ROOT_DIR / "ebnerd_small" / "validation" / "behaviors.parquet"


def ensure_directories() -> None:
    """Ensure all required cache and output directories exist."""
    for d in [CACHE_DIR, CACHE_TRAIN_DIR, CACHE_VAL_DIR, CACHE_EMB_DIR, OUTPUTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_train_manifest() -> Dict[str, Any]:
    """Load the train manifest metadata."""
    manifest_path = CACHE_TRAIN_DIR / "train_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "protocol_commit_hash": PROTOCOL_COMMIT_HASH,
        "preoutcome_gate_commit_hash": PREOUTCOME_GATE_COMMIT_HASH,
        "implementation_lock_commit_hash": IMPLEMENTATION_LOCK_COMMIT_HASH,
        "seed": SEED,
        "protocol_version": PROTOCOL_VERSION,
    }
