#!/usr/bin/env python3
"""
One-time migration: legacy ``* - Segmented`` + ``Close Calls/`` →

  data/goaltends/{segmented,close_calls}/
  data/legal contacts/{blocks,hand_on_backboard,ball_on_rim,other_data,close_calls}/

Run from ``nba-goaltend-project`` (so ``data/`` is beside ``scripts/``):

  python scripts/reorganize_data_layout.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OLD_CLOSE = DATA / "Close Calls"
LABELS = DATA / "cleaned_ground_truth.csv"

LEGAL = DATA / "legal contacts"
GOAL = DATA / "goaltends"


def _norm_fn(s: str) -> str:
    s = str(s).strip()
    return s if s.endswith(".csv") else f"{s}.csv"


def main() -> int:
    if not DATA.is_dir():
        print("No data dir", DATA, file=sys.stderr)
        return 1

    LEGAL.mkdir(parents=True, exist_ok=True)
    GOAL.mkdir(parents=True, exist_ok=True)
    for sub in ("blocks", "hand_on_backboard", "ball_on_rim", "other_data", "close_calls"):
        (LEGAL / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("segmented", "close_calls"):
        (GOAL / sub).mkdir(parents=True, exist_ok=True)

    moves: list[tuple[Path, Path]] = []

    # Segmented → typed folders
    seg_map = [
        (DATA / "Blocks - Segmented", LEGAL / "blocks"),
        (DATA / "Hand on Backboard - Segmented", LEGAL / "hand_on_backboard"),
        (DATA / "Ball on Rim - Segmented", LEGAL / "ball_on_rim"),
        (DATA / "Other Data - Segmented", LEGAL / "other_data"),
        (DATA / "Goaltends - Segmented", GOAL / "segmented"),
    ]
    for src_dir, dst_dir in seg_map:
        if not src_dir.is_dir():
            continue
        for f in sorted(src_dir.glob("*.csv")):
            moves.append((f, dst_dir / f.name))

    # Close calls by label from cleaned_ground_truth.csv
    if LABELS.is_file() and OLD_CLOSE.is_dir():
        df = pd.read_csv(LABELS)
        cols = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
        fn_col = cols.get("file_name") or cols.get("filename")
        gt_col = cols.get("ground_truth")
        if not fn_col or not gt_col:
            print("cleaned_ground_truth.csv missing File Name / Ground Truth columns", file=sys.stderr)
            return 1
        for _, row in df.iterrows():
            fn = _norm_fn(row[fn_col])
            gt = str(row[gt_col]).strip().lower()
            if gt in ("block", "legal", "leg"):
                dst = LEGAL / "close_calls" / fn
            elif "goaltend" in gt:
                dst = GOAL / "close_calls" / fn
            else:
                continue
            src = OLD_CLOSE / fn
            if src.is_file():
                moves.append((src, dst))

    # Execute moves (skip if dest exists and same inode)
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if src.resolve() == dst.resolve():
                continue
            dst.unlink()
        shutil.move(str(src), str(dst))

    # Remove empty legacy dirs
    for d in (
        DATA / "Blocks - Segmented",
        DATA / "Hand on Backboard - Segmented",
        DATA / "Ball on Rim - Segmented",
        DATA / "Other Data - Segmented",
        DATA / "Goaltends - Segmented",
        OLD_CLOSE,
    ):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass  # not empty — leave for manual cleanup

    (LEGAL / "ball_on_rim" / ".gitkeep").touch(exist_ok=True)
    print("Done. Moved", len(moves), "files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
