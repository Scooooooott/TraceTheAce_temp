"""Load the raw competition tables from data/raw/.

Filenames carry DrivenData's random hash suffixes; they are pinned here so the
rest of the codebase never has to know about them. If the files are
re-downloaded with new hashes, update the constants below.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

TRAIN_FEATURES_PATH = RAW_DIR / "train_features_TMQTWsB.csv"
TRAIN_LABELS_PATH = RAW_DIR / "train_labels_44ujmj2.csv"
TRAIN_TRANSCRIPTS_DIR = RAW_DIR / "train_transcripts"

# Local reference copies only -- the competition container provides its own
# data/test_features.csv, data/submission_format.csv and data/test_transcripts/
# under fixed names, unrelated to these hashed filenames.
SUBMISSION_FORMAT_SMOKE_PATH = RAW_DIR / "submission_format_5muR4s3.csv"
SUBMISSION_FORMAT_FULL_PATH = RAW_DIR / "submission_format_ZQLcKx7.csv"


def load_features() -> pd.DataFrame:
    """response_id, session_id, learning_objective_id, learning_objective."""
    return pd.read_csv(TRAIN_FEATURES_PATH)


def load_labels() -> pd.DataFrame:
    """response_id, is_correct (0.0/1.0)."""
    return pd.read_csv(TRAIN_LABELS_PATH)


def load_training_data() -> pd.DataFrame:
    """Features joined with labels, one row per response_id."""
    return load_features().merge(load_labels(), on="response_id", how="inner")


def load_transcript(session_id: str) -> pd.DataFrame:
    """utterance_id, role (tutor/student/background), content, timestamp."""
    return pd.read_csv(TRAIN_TRANSCRIPTS_DIR / f"{session_id}.csv")
