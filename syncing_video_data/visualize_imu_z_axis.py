#!/usr/bin/env python3
"""
Plot Z-axis accelerometer traces vs CSV clock time.

Use this to choose an IMU time by eye (e.g. rim contact on the plotted peak), pair it with a
video time from videoFrameNavigator or the player scrubber, then run:

  overlay_imu_on_video.py --manual-knock --imu-knock <CSV_s> --video-knock <video_s> ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _short_label(name: str) -> str:
    s = str(name).strip()
    return s.split(":", 1)[-1].strip() if ":" in s else s


def _guess_z_columns(columns: list[str]) -> list[str]:
    """Any header with both acceleration and a Z-axis marker (PASCO/Vernier style)."""
    out: list[str] = []
    for c in columns:
        lc = str(c).lower().replace("²", "")
        if "acceleration" not in lc:
            continue
        if "z-axis" in lc or "z acceleration" in lc:
            out.append(str(c).strip())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize IMU CSV Z-axis vs time.")
    parser.add_argument("csv", type=Path, help="Path to accelerometer CSV")
    parser.add_argument("--time-col", required=True, help=r'CSV time column e.g. "Latest: Time (s)"')
    parser.add_argument(
        "--z-cols",
        nargs="+",
        default=None,
        metavar="COL",
        help='Z-axis columns (default: all columns matching "*Z* Acceleration*"',
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Print CSV time under each left-click (for manual knock values)",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--ylim", nargs=2, type=float)
    parser.add_argument("--xlim", nargs=2, type=float)
    parser.add_argument("--figsize", nargs=2, type=float, default=(12.0, 5.0))
    args = parser.parse_args()

    path = args.csv.expanduser().resolve()
    if not path.is_file():
        print(f"Error: CSV not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    df.columns = [str(c).strip().strip('"') for c in df.columns]

    tc = args.time_col.strip().strip('"')
    if tc not in df.columns:
        print(f"Error: missing time column {tc!r}", file=sys.stderr)
        print("Got:", df.columns.tolist(), file=sys.stderr)
        sys.exit(1)

    z_cols = args.z_cols or _guess_z_columns(df.columns.tolist())

    missing = [c for c in z_cols if c not in df.columns]
    if missing:
        print(f"Error: unknown column(s): {missing}", file=sys.stderr)
        sys.exit(1)

    t = pd.to_numeric(df[tc], errors="coerce")
    finite_t = np.asarray(t[~t.isna()], dtype=float)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    for col in z_cols:
        z = pd.to_numeric(df[col], errors="coerce")
        ax.plot(t, z, lw=1.1, alpha=0.9, label=_short_label(col))

    ax.axhline(0.0, color="k", lw=0.6, alpha=0.35)
    ax.set_xlabel(tc)
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title(args.title or path.name)
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.35)
    if args.xlim is not None:
        ax.set_xlim(args.xlim)
    if args.ylim is not None:
        ax.set_ylim(args.ylim)

    if args.interactive:
        try:
            if fig.canvas.manager is not None:
                fig.canvas.manager.set_window_title("IMU Z — click to print CSV time")
        except Exception:
            pass

        def onclick(ev):
            if ev.inaxes != ax or ev.button != 1:
                return
            if ev.xdata is None:
                return
            xc = float(ev.xdata)
            if finite_t.size:
                idx = int(np.argmin(np.abs(finite_t - xc)))
                near = float(finite_t[idx])
                dt = xc - near
                print(f"clicked t ≈ {xc:.6f} s CSV  |  nearest row {near:.6f} s (Δclick {dt*1000:+.3f} ms)")
            else:
                print(f"clicked t ≈ {xc:.6f} s CSV")

        fig.canvas.mpl_connect("button_press_event", onclick)
        print(
            "Left-click the plot → prints CSV clock time.",
            "\nPick one event here (imu_knock) and pair with the same motion in the video (video_knock).",
            file=sys.stderr,
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
