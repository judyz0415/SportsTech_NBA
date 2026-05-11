"""Repository layout: data under ``data/``, generated files under ``outputs/``."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Directory containing ``src/``, ``data/``, and ``README.md``."""
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Folder with ``* - Segmented`` directories, ``Close Calls/``, and ``close_calls_labels.csv``."""
    return Path(os.environ.get("GOALTEND_DATA_DIR", project_root() / "data")).resolve()


def outputs_dir() -> Path:
    """Folder for prediction CSVs and other generated artifacts."""
    return Path(os.environ.get("GOALTEND_OUTPUT_DIR", project_root() / "outputs")).resolve()


def labels_csv_path() -> Path:
    """CSV listing close-call filenames and labels; override with ``GOALTEND_LABELS_PATH``."""
    override = os.environ.get("GOALTEND_LABELS_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return data_root() / "close_calls_labels.csv"
