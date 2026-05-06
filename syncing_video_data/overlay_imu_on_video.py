"""
Overlay accelerometer data from a CSV on a video (same pipeline as code.py / code2.py).

Either give manual knock times (--imu-knock / --video-knock) or align the IMU spike
peak to a known video time (--sync-peak-at-video).

Omit --acc-col to auto-pick an axis. Default (--axis-pick robust-mad): per channel,
subtract median (removes gravity / DC bias on that axis), then score spikes by
max |a − median| divided by a robust spread (MAD), so axes near 0 m/s² and near
~10 m/s² are compared fairly. Use --axis-pick max-deviation for the old behaviour.

Tip: playback time defaults to successive frame reads × (1/FPS); use --timeline-fps if metadata lies.

Two close mechanical hits are common on goaltends: the **first in time** is often **backboard** contact;
the **second** may be **rim** contact — on your rigs the rim impulse can be **larger** on the plotted axis.
Peaks are ranked by **strongest |a − median|** (see --csv-align-peak-index): **index 0 = largest** excursion
(often rim if it dominates), **index 1 = second-largest** (often the weaker backboard bump). Use a
CSV time window (--csv-align-window) to bracket the pairing and silence junk before contact.

  python overlay_imu_on_video.py \\
    --video "/path/to/closecalls4.MP4" \\
    --csv "/path/to/close_calls_4.csv" \\
    --time-col "Latest: Time (s)" \\
    --sync-peak-at-video 6.506 \\
    --csv-align-window 3.2 14.75 \\
    --csv-align-peak-index 0 \\
    --timeline-fps 960 \\
    --xlim 4 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def _accel_columns(df: pd.DataFrame, time_col: str) -> list[str]:
    tnorm = str(time_col).strip()
    cols = []
    for c in df.columns:
        c_st = str(c).strip()
        if c_st == tnorm:
            continue
        if "acceleration" in c_st.lower() and "m/s" in c_st.lower():
            cols.append(c)
    return cols


def _robust_spread(vals: np.ndarray, med: float) -> float:
    """Typical scale of variation on this axis (DC already removed in dev)."""
    dev = np.abs(vals.astype(np.float64) - med)
    mad = float(np.median(dev)) if dev.size else 0.0
    sigma_mad = 1.4826 * mad
    sd = float(np.std(vals)) if vals.size else 0.0
    p10, p90 = np.percentile(vals, [10, 90]) if vals.size else (0.0, 0.0)
    spread = float(p90 - p10)
    return max(sigma_mad, sd, 0.2 * spread, 1e-6)


def _peak_times_ranked(
    df: pd.DataFrame,
    time_col: str,
    acc_col: str,
    *,
    csv_t_min: float | None,
    csv_t_max: float | None,
    min_peak_sep_s: float,
) -> list[tuple[float, float]]:
    """
    Local-maxima peaks of |a − median|, non-max-suppression in time, sorted strongest first.

    Uses the **full-record median** per axis so gravity bias stays consistent inside a narrow window.

    Each item is (csv_time_peak, deviation_magnitude_at_peak).
    """
    s_all = pd.to_numeric(df[acc_col], errors="coerce")
    t_all = pd.to_numeric(df[time_col], errors="coerce")
    mask_ok = s_all.notna() & t_all.notna()
    if int(mask_ok.sum()) < 6:
        return []
    full_med = float(s_all[mask_ok].median())

    tab = pd.DataFrame(
        {
            "t": t_all[mask_ok].astype(np.float64),
            "a": s_all[mask_ok].astype(np.float64),
        }
    ).sort_values("t")
    tab["dev"] = (tab["a"] - full_med).abs()

    dd = tab
    if csv_t_min is not None:
        dd = dd[dd["t"] >= float(csv_t_min)]
    if csv_t_max is not None:
        dd = dd[dd["t"] <= float(csv_t_max)]
    if len(dd) < 6:
        return []

    t = dd["t"].to_numpy(dtype=np.float64)
    y = dd["dev"].to_numpy(dtype=np.float64)

    lm_idx = []
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] > 0:
            lm_idx.append(i)

    peaks: list[tuple[float, float]] = []
    for i in sorted(lm_idx, key=lambda j: y[j], reverse=True):
        tt = float(t[i])
        if all(abs(tt - pk[0]) >= min_peak_sep_s for pk in peaks):
            peaks.append((tt, float(y[i])))
    peaks.sort(key=lambda p: -p[1])
    return peaks


def _axis_peak_median_relative(
    df: pd.DataFrame, time_col: str, acc_col: str
) -> tuple[float, float, float, float] | None:
    """
    Peak relative to each axis median (removes gravity / constant bias on that axis).

    Returns (csv_time_peak, median, peak_abs_deviation, robust_spread) or None.
    """
    s = pd.to_numeric(df[acc_col], errors="coerce")
    mask = s.notna()
    if mask.sum() < 3:
        return None
    vals = s[mask].to_numpy(dtype=np.float64)
    t_series = pd.to_numeric(df.loc[mask, time_col], errors="coerce")
    if t_series.isna().all():
        return None
    med = float(np.median(vals))
    abs_dev = np.abs(vals - med)
    ir = int(np.argmax(abs_dev))
    peak_abs = float(abs_dev[ir])
    peak_t = float(t_series.iloc[ir])
    spread = _robust_spread(vals, med)
    return peak_t, med, peak_abs, spread


def pick_axis_auto(
    df: pd.DataFrame,
    time_col: str,
    acc_col: str | None,
    strategy: str,
    *,
    csv_t_min: float | None,
    csv_t_max: float | None,
    align_peak_index: int,
    min_peak_sep_s: float,
) -> tuple[str, float]:
    """
    Pick axis (optional) and return CSV time used for synchronization.

    If ``acc_col`` is set, only timing is recomputed via ranked peaks inside the CSV window.

    Peaks here are merged local maxima of |a − median|, sorted **by magnitude**, not chronological order.
    So align_peak_index **0** aligns the **strongest** separated bump (often **rim** if it dominates),
    and **1** the **second-strongest** (often the **smaller** backboard precursor when both are picked up).
    """
    t_min, t_max = csv_t_min, csv_t_max
    win_txt = ""
    if t_min is not None or t_max is not None:
        win_txt = f" window CSV [{t_min}, {t_max}]"

    ranked_for = (
        lambda col: _peak_times_ranked(
            df, time_col, col, csv_t_min=t_min, csv_t_max=t_max, min_peak_sep_s=min_peak_sep_s
        )
    )

    if acc_col is not None:
        if acc_col not in df.columns:
            raise ValueError(f"Column not found: {acc_col!r}")
        ranked = ranked_for(acc_col)
        if align_peak_index < 0 or align_peak_index >= len(ranked):
            raise ValueError(
                f"Column {acc_col!r}:{win_txt}: need peak index {align_peak_index}, "
                f"found {len(ranked)} separated peaks {[f'{x[0]:.4f}s' for x in ranked[:6]]}"
            )
        t_peak = ranked[align_peak_index][0]
        print(f"Align on {acc_col!r}:{win_txt} peak[{align_peak_index}] at CSV t={t_peak:.6f} s")
        if ranked:
            print("  Separated peaks in window (CSV s): " + ", ".join(f"{x[0]:.4f}" for x in ranked[:8]))
        return acc_col, t_peak

    candidates = _accel_columns(df, time_col)
    if not candidates:
        raise ValueError(
            "No acceleration channels found (names should include 'Acceleration' and 'm/s'). "
            "Pass --acc-col explicitly."
        )

    strategy = strategy.strip().lower()
    if strategy not in ("robust-mad", "max-deviation"):
        raise ValueError(f"Unknown --axis-pick strategy: {strategy!r}")

    best_col = None
    best_score = -1.0
    best_peak_t = 0.0
    best_mag = 0.0
    best_spread = 1.0
    best_med = 0.0
    best_ranked: list[tuple[float, float]] = []

    for c in candidates:
        ranked = ranked_for(c)
        if align_peak_index < 0 or align_peak_index >= len(ranked):
            continue
        tpk = ranked[align_peak_index][0]
        mag = ranked[align_peak_index][1]
        stats_full = _axis_peak_median_relative(df, time_col, c)
        if stats_full is None:
            continue
        _pt, med, _pa, spread = stats_full
        if strategy == "robust-mad":
            score = mag / spread
        else:
            score = mag
        if score > best_score:
            best_score = score
            best_col = c
            best_peak_t = tpk
            best_mag = mag
            best_spread = spread
            best_med = med
            best_ranked = ranked

    if best_col is None:
        raise ValueError(
            "Could not auto-select an acceleration channel for this CSV window / peak index. "
            "Widen --csv-align-window or lower --min-peak-sep, "
            "or pick the axis/time yourself with --manual-knock and --imu-knock."
        )

    z_eq = best_mag / best_spread if best_spread else 0.0
    print(
        f"Axis pick ({strategy}), align peak[{align_peak_index}]{win_txt}: median≈{best_med:.4f} m/s², "
        f"|Δ|≈{best_mag:.4f} m/s², spread≈{best_spread:.4f} m/s² (~{z_eq:.2f}×) at CSV t={best_peak_t:.6f} s → "
        f"{best_col!r}"
    )
    print("  Peaks in scope (CSV s): " + ", ".join(f"{x[0]:.4f}" for x in best_ranked[:8]))
    return best_col, best_peak_t


def _peak_deviation_time(
    df: pd.DataFrame,
    time_col: str,
    acc_col: str,
    *,
    csv_t_min: float | None = None,
    csv_t_max: float | None = None,
    align_peak_index: int = 0,
    min_peak_sep_s: float = 0.07,
) -> float:
    """CSV time at the chosen merged local-maximum of |a − median(a)|."""
    ranked = _peak_times_ranked(
        df,
        time_col,
        acc_col,
        csv_t_min=csv_t_min,
        csv_t_max=csv_t_max,
        min_peak_sep_s=min_peak_sep_s,
    )
    if align_peak_index < 0 or align_peak_index >= len(ranked):
        raise ValueError(
            f"Column {acc_col!r}: need peak index {align_peak_index}, found {len(ranked)} peaks"
        )
    return ranked[align_peak_index][0]


# Backwards compatibility for imports / quick tests
def pick_max_deviation_axis(df: pd.DataFrame, time_col: str, acc_col: str | None) -> tuple[str, float]:
    return pick_axis_auto(
        df,
        time_col,
        acc_col,
        "max-deviation",
        csv_t_min=None,
        csv_t_max=None,
        align_peak_index=0,
        min_peak_sep_s=0.07,
    )


def short_label(full_name: str) -> str:
    s = full_name.strip()
    if ":" in s:
        s = s.split(":", 1)[1].strip()
    return s if len(s) < 52 else s[:49] + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay IMU acceleration on video.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--sync-peak-at-video",
        dest="peak_video_sync",
        type=float,
        default=None,
        metavar="T",
        help="Video time (s) for IMU spike peak alignment",
    )
    g.add_argument(
        "--manual-knock",
        dest="manual_knock",
        action="store_true",
        help="Use --imu-knock / --video-knock instead",
    )

    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--time-col", required=True)
    parser.add_argument(
        "--acc-col",
        default=None,
        help="Acceleration column to plot (see --axis-pick when omitted)",
    )
    parser.add_argument(
        "--axis-pick",
        choices=("robust-mad", "max-deviation"),
        default="robust-mad",
        help=(
            "How to auto-pick channel: robust-mad = max |(a−median)/spread| "
            "(fair across ~0 vs ~g gravity bias); "
            "max-deviation = largest raw |a−median|."
        ),
    )
    parser.add_argument("--imu-knock", type=float, default=None)
    parser.add_argument("--video-knock", type=float, default=None)
    parser.add_argument("--timeline-fps", type=float, default=None)
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--crop-left-pct", type=float, default=0.12)
    parser.add_argument("--crop-bottom-pct", type=float, default=0.15)
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--scale", type=float, default=0.75)
    parser.add_argument("--xlim", nargs=2, type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps-divisor", type=float, default=1.0)
    parser.add_argument("--large-labels", action="store_true")
    parser.add_argument(
        "--csv-align-window",
        nargs=2,
        type=float,
        metavar=("CSV_T0", "CSV_T1"),
        help=(
            "Only consider IMU spikes in this CSV clock range when auto-aligning/ranking peaks "
            "(helps bracket backboard+rim pair moments and drop pre-shot noise)"
        ),
    )
    parser.add_argument(
        "--csv-align-peak-index",
        type=int,
        default=0,
        help=(
            "Which separated peak by |a−median| strength in scope "
            "(0=largest excursion, often rim; 1=second-strongest, often weaker backboard if both merged)"
        ),
    )
    parser.add_argument(
        "--min-peak-sep",
        type=float,
        default=0.07,
        help="CSV seconds between merged spikes for multi-hit separation",
    )
    parser.add_argument(
        "--use-opencv-pos-frames",
        action="store_true",
        help=(
            "Use OpenCV CAP_PROP_POS_FRAMES for playback time instead of sequential frame counting "
            "(can desync overlays on some codecs)"
        ),
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open OpenCV preview windows (faster batch render; no 'q' quit)",
    )

    args = parser.parse_args()

    if args.manual_knock and (args.imu_knock is None or args.video_knock is None):
        parser.error("--manual-knock requires --imu-knock and --video-knock")
    video_path = args.video.resolve()
    data_path = args.csv.resolve()

    if not video_path.is_file():
        print(f"Error: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    if not data_path.is_file():
        print(f"Error: CSV not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    root_df = pd.read_csv(data_path)
    root_df.columns = [str(c).strip().strip('"') for c in root_df.columns]

    if args.time_col not in root_df.columns:
        print(f"Error: time column not in CSV: {args.time_col!r}", file=sys.stderr)
        print("Columns:", list(root_df.columns), file=sys.stderr)
        sys.exit(1)

    tcol = args.time_col.rstrip()

    if args.csv_align_window is not None:
        csv_t_min = float(args.csv_align_window[0])
        csv_t_max = float(args.csv_align_window[1])
        if csv_t_min >= csv_t_max:
            parser.error("--csv-align-window: T0 must be < T1")
    else:
        csv_t_min, csv_t_max = None, None

    pk_idx = args.csv_align_peak_index
    if pk_idx < 0:
        parser.error("--csv-align-peak-index must be >= 0")

    pick_kw = dict(
        csv_t_min=csv_t_min,
        csv_t_max=csv_t_max,
        align_peak_index=pk_idx,
        min_peak_sep_s=args.min_peak_sep,
    )

    try:
        if args.manual_knock:
            if args.acc_col:
                acc_col = args.acc_col.strip().strip('"')
                if acc_col not in root_df.columns:
                    print(f"Error: acc column not in CSV: {acc_col!r}", file=sys.stderr)
                    sys.exit(1)
            else:
                acc_col, _ = pick_axis_auto(
                    root_df, tcol, None, args.axis_pick, **pick_kw
                )
            peak_t_csv = float(args.imu_knock)
        elif args.acc_col:
            acc_col = args.acc_col.strip().strip('"')
            if acc_col not in root_df.columns:
                print(f"Error: acc column not in CSV: {acc_col!r}", file=sys.stderr)
                sys.exit(1)
            peak_t_csv = _peak_deviation_time(root_df, tcol, acc_col, **pick_kw)
            ranked = _peak_times_ranked(
                root_df, tcol, acc_col,
                csv_t_min=csv_t_min, csv_t_max=csv_t_max, min_peak_sep_s=args.min_peak_sep,
            )
            if ranked:
                print("  Peaks in scope (CSV s): " + ", ".join(f"{x[0]:.4f}" for x in ranked[:8]))
        else:
            acc_col, peak_t_csv = pick_axis_auto(
                root_df, tcol, None, args.axis_pick, **pick_kw
            )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.manual_knock:
        imu_knock = float(args.imu_knock)
        video_knock = float(args.video_knock)
    else:
        imu_knock = peak_t_csv
        video_knock = float(args.peak_video_sync)

    offset = video_knock - imu_knock
    if args.acc_col:
        print(f"Using axis (--acc-col): {acc_col}")
        if not args.manual_knock:
            print(f"Alignment peak[{pk_idx}] on that axis CSV time {peak_t_csv:.6f} s")
    if args.manual_knock:
        print(f"Manual sync: CSV {imu_knock:.6f} s ↔ video {video_knock:.6f} s (offset {offset:+.6f} s)")
    else:
        print(f"Chosen peak aligned to video {video_knock:.6f} s (offset {offset:+.6f} s)")

    plot_df = root_df[[tcol, acc_col]].copy()
    plot_df[acc_col] = pd.to_numeric(plot_df[acc_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[acc_col])

    plot_df["video_time"] = plot_df[tcol] + offset

    vmax = plot_df[acc_col].max()
    vmin = plot_df[acc_col].min()

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: could not open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    meta_fps = cap.get(cv.CAP_PROP_FPS)
    if meta_fps <= 0:
        meta_fps = 30.0
    timeline_fps = float(args.timeline_fps) if args.timeline_fps and args.timeline_fps > 0 else meta_fps

    if args.timeline_fps is None and meta_fps < 240:
        print(
            f"Note: container reports FPS≈{meta_fps:.3f}. If clips are truly high-speed, "
            f"overlay time may be wrong — pass --timeline-fps matching capture rate.",
            file=sys.stderr,
        )

    orig_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    if should_rotate := args.rotate:
        rotated_width = orig_height
        rotated_height = orig_width
    else:
        rotated_width = orig_width
        rotated_height = orig_height

    scale = args.scale
    scale_width = int(rotated_width * scale)
    scale_height = int(rotated_height * scale)

    if args.no_crop:
        crop_left = 0
        crop_bottom = 0
    else:
        crop_left = int(rotated_width * args.crop_left_pct)
        crop_bottom = int(rotated_height * args.crop_bottom_pct)

    out_fps = max(1.0, round(timeline_fps / args.fps_divisor))
    out_path = args.output
    if out_path is None:
        out_path = video_path.parent / f"{video_path.stem}_overlay.mp4"
    else:
        out_path = out_path.resolve()

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(str(out_path), fourcc, out_fps, (scale_width, scale_height))

    xlim = args.xlim
    if xlim is None:
        vt = plot_df["video_time"]
        span = max(2.0, float(vt.max() - vt.min()))
        mid = float(vt.mean())
        xlim = (mid - span / 2, mid + span / 2)

    if args.timeline_fps:
        print(f"Timeline FPS override: {timeline_fps} (OpenCV reported {meta_fps})")

    playback_frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if args.use_opencv_pos_frames:
            current_time = cap.get(cv.CAP_PROP_POS_FRAMES) / timeline_fps
        else:
            current_time = playback_frame_idx / timeline_fps
            playback_frame_idx += 1

        if should_rotate:
            frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
            if args.no_crop:
                frame = cv.resize(frame, (scale_width, scale_height))
            else:
                frame = frame[0 : rotated_height - crop_bottom, crop_left:rotated_width]
                frame = cv.resize(frame, (scale_width, scale_height))
        else:
            if args.no_crop:
                frame = cv.resize(frame, (scale_width, scale_height))
            else:
                frame = frame[0 : rotated_height - crop_bottom, crop_left:rotated_width]
                frame = cv.resize(frame, (scale_width, scale_height))

        graph_df = plot_df[plot_df["video_time"] <= current_time]

        if not graph_df.empty:
            if args.large_labels:
                fig = Figure(figsize=(24, 5), dpi=100)
                title_fs, axis_fs, tick_fs = 16, 16, 15
            else:
                fig = Figure(figsize=(11, 2), dpi=100)
                title_fs, axis_fs, tick_fs = 10, 8, 8

            canvas = FigureCanvas(fig)
            fig.patch.set_alpha(0.0)

            ax = fig.add_subplot(111)
            ax.plot(graph_df["video_time"], graph_df[acc_col], color="blue")
            ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.4)
            ax.set_xlim(xlim)
            pad = max(5.0, 0.05 * (abs(vmax) + abs(vmin)))
            ax.set_ylim([vmin - pad, vmax + pad])

            plain_title = f"{short_label(acc_col)} (m/s²)"
            ax.set_title(plain_title, fontsize=title_fs)
            ax.set_xlabel("Video time (s)", fontsize=axis_fs)
            ax.set_ylabel(plain_title, fontsize=axis_fs)

            if args.large_labels:
                ax.tick_params(axis="both", labelsize=tick_fs, width=2)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontweight("bold")
                tick_step = max(0.5, (xlim[1] - xlim[0]) / 5)
                ax.set_xticks(np.arange(xlim[0], xlim[1] + tick_step * 0.5, tick_step))
                ax.set_facecolor((0.85, 0.85, 0.85, 0.6))
            else:
                ax.tick_params(axis="both", labelsize=tick_fs)
                ax.set_xticks(np.linspace(xlim[0], xlim[1], min(5, max(3, int(xlim[1] - xlim[0]) + 1))))
                ax.set_facecolor((1, 1, 1, 0.2))

            canvas.draw()

            graph_img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            graph_img = graph_img.reshape(canvas.get_width_height()[::-1] + (4,))
            graph_rgb = graph_img[:, :, :3]
            alpha_mask = graph_img[:, :, 3] / 255.0

            gh, gw, _ = graph_rgb.shape
            insert_x, insert_y = 10, scale_height - gh - 10

            roi = frame[insert_y : insert_y + gh, insert_x : insert_x + gw].astype(np.float32)
            blended = roi * (1 - alpha_mask[..., None]) + graph_rgb.astype(np.float32) * alpha_mask[..., None]
            frame[insert_y : insert_y + gh, insert_x : insert_x + gw] = blended.astype(np.uint8)

        if not args.no_display:
            cv.imshow("Video", frame)
        out.write(frame)
        if not args.no_display and (cv.waitKey(1) & 0xFF == ord("q")):
            break

    cap.release()
    out.release()
    if not args.no_display:
        cv.destroyAllWindows()
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
