# ===========================================================================
# Sync Mike Video with IMU Data — Progressive Graph Overlay
# ===========================================================================
# The graph draws progressively as the video plays (data revealed up to the
# current frame). The x-axis is fixed to the full time range so it never
# scrolls. Light semi-transparent background.
# ===========================================================================

import cv2 as cv
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURE THESE FOUR VALUES
# ---------------------------------------------------------------------------
VIDEO_NUMBER = 8

IMU_KNOCK_TIME_S   = 2.867186
VIDEO_KNOCK_TIME_S = 4.653

SENSOR = 1

SHOULD_ROTATE = False
SCALE = 0.75
GRAPH_HEIGHT_PX = 260
# ---------------------------------------------------------------------------

MIKE_VIDEOS_DIR = Path(__file__).resolve().parents[4] / "MikeVideos"

video_path = MIKE_VIDEOS_DIR / f"{VIDEO_NUMBER}.MOV"
csv_path   = MIKE_VIDEOS_DIR / f"videos_final_{VIDEO_NUMBER}.csv"

time_col = "Latest: Time (s)"
z_col    = f"Latest: Z Acceleration {SENSOR} (m/s²)"

df = pd.read_csv(csv_path)
df = df[[time_col, z_col]].copy()

offset = VIDEO_KNOCK_TIME_S - IMU_KNOCK_TIME_S
df["video_time"] = df[time_col] + offset

z_min = df[z_col].min()
z_max = df[z_col].max()
t_min = df["video_time"].min()
t_max = df["video_time"].max()

# --- Video setup ---
cap = cv.VideoCapture(str(video_path))
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

fps    = cap.get(cv.CAP_PROP_FPS)
orig_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

frame_w, frame_h = (orig_h, orig_w) if SHOULD_ROTATE else (orig_w, orig_h)
out_w = int(frame_w * SCALE)
out_h = int(frame_h * SCALE)

DPI   = 100
fig_w = (out_w * 0.75) / DPI
fig_h = GRAPH_HEIGHT_PX / DPI

out_path = video_path.parent / f"{video_path.stem}_progressive_overlay.mp4"
fourcc = cv.VideoWriter_fourcc(*"mp4v")
writer = cv.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

print(f"Video : {video_path.name}  ({fps:.2f} fps)")
print(f"CSV   : {csv_path.name}")
print(f"Output: {out_path.name}")
print("Processing… (press Q in the preview window to stop early)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num    = cap.get(cv.CAP_PROP_POS_FRAMES)
    current_time = frame_num / fps

    if SHOULD_ROTATE:
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    frame = cv.resize(frame, (out_w, out_h))

    # Data revealed up to current video time; x-axis stays fixed
    graph_df = df[df["video_time"] <= current_time]

    if not graph_df.empty:
        fig = Figure(figsize=(fig_w, fig_h), dpi=DPI)
        canvas = FigureCanvas(fig)
        fig.patch.set_alpha(0.0)

        ax = fig.add_subplot(111)
        ax.plot(graph_df["video_time"], graph_df[z_col], color="#0a3a6e", linewidth=1.8)
        ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_xlim([8, 10])
        ax.set_ylim([z_min - 5, z_max + 5])
        ax.set_xlabel("Time (s)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Z Accel (m/s²)", fontsize=10, fontweight="bold")
        ax.set_title(f"Sensor {SENSOR} — Z-axis Acceleration", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_facecolor((0.97, 0.97, 0.97, 0.92))
        fig.tight_layout(pad=0.4)

        canvas.draw()
        img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(canvas.get_width_height()[::-1] + (4,))
        graph_rgb  = img[:, :, :3]
        alpha_mask = img[:, :, 3] / 255.0

        gh, gw = graph_rgb.shape[:2]
        ix, iy = 0, out_h - gh

        if iy >= 0 and gw <= out_w:
            roi = frame[iy:iy+gh, ix:ix+gw].astype(np.float32)
            blended = roi * (1 - alpha_mask[..., None]) + graph_rgb.astype(np.float32) * alpha_mask[..., None]
            frame[iy:iy+gh, ix:ix+gw] = blended.astype(np.uint8)

    cv.imshow("Overlay Preview", frame)
    writer.write(frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        print("Stopped early by user.")
        break

cap.release()
writer.release()
cv.destroyAllWindows()
print(f"Done. Saved to: {out_path}")
