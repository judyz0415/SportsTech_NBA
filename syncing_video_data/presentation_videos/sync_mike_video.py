# ===========================================================================
# Sync Mike Video with IMU Data — Presentation Overlay
# ===========================================================================
# PURPOSE
#   Overlays a live-drawing Z-axis acceleration graph onto one of the
#   MikeVideos (.MOV) files, aligned to a known sync point.
#
# QUICK START
#   1. Use videoFrameNavigator.py (in the parent folder) to find the exact
#      video time of the sync event (e.g. the ball hitting the rim).
#      Run:  python ../videoFrameNavigator.py <path_to_video>
#      Press T at the sync frame → it prints/saves the frame number and time.
#
#   2. In your CSV data, find the time value (Latest: Time (s)) that
#      corresponds to that same event.
#
#   3. Fill in VIDEO_NUMBER, IMU_KNOCK_TIME_S, VIDEO_KNOCK_TIME_S,
#      and SENSOR below, then run:
#         python sync_mike_video.py
#
# OUTPUT
#   <video_stem>_overlay.mp4  saved next to the video file.
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
VIDEO_NUMBER = 1          # integer 1-8 (matches 1.MOV and videos_final_1.csv)

IMU_KNOCK_TIME_S  = 0.911302   # TODO: time in the CSV (Latest: Time (s)) at the sync event
VIDEO_KNOCK_TIME_S = 04.234  # TODO: seconds into the video at the same sync event
                           #       (use videoFrameNavigator.py to find this)

SENSOR = 1                 # 1 or 2 — which accelerometer column set to plot

SHOULD_ROTATE = False      # True if the phone video needs 90° clockwise rotation
SCALE = 0.75               # resize factor for the output video
GRAPH_WINDOW_S = 4.0       # how many seconds of history to show in the graph
# ---------------------------------------------------------------------------

MIKE_VIDEOS_DIR = Path(__file__).resolve().parents[4] / "MikeVideos"

video_path = MIKE_VIDEOS_DIR / f"{VIDEO_NUMBER}.MOV"
csv_path   = MIKE_VIDEOS_DIR / f"videos_final_{VIDEO_NUMBER}.csv"

time_col  = "Latest: Time (s)"
z_col     = f"Latest: Z Acceleration {SENSOR} (m/s²)"

# --- Load and align IMU data ---
df = pd.read_csv(csv_path)
df = df[[time_col, z_col]].copy()

offset = VIDEO_KNOCK_TIME_S - IMU_KNOCK_TIME_S
df["video_time"] = df[time_col] + offset

z_min = df[z_col].min()
z_max = df[z_col].max()

# --- Video setup ---
cap = cv.VideoCapture(str(video_path))
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

fps        = cap.get(cv.CAP_PROP_FPS)
orig_w     = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
orig_h     = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

if SHOULD_ROTATE:
    frame_w, frame_h = orig_h, orig_w
else:
    frame_w, frame_h = orig_w, orig_h

out_w = int(frame_w * SCALE)
out_h = int(frame_h * SCALE)

out_path = video_path.parent / f"{video_path.stem}_overlay.mp4"
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

    # IMU data visible up to this video time
    window_start = current_time - GRAPH_WINDOW_S
    graph_df = df[(df["video_time"] >= window_start) & (df["video_time"] <= current_time)]

    if not graph_df.empty:
        fig = Figure(figsize=(12, 2.5), dpi=100)
        canvas = FigureCanvas(fig)
        fig.patch.set_alpha(0.0)

        ax = fig.add_subplot(111)
        ax.plot(graph_df["video_time"], graph_df[z_col], color="dodgerblue", linewidth=1.8)
        ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_xlim([window_start, current_time])
        ax.set_ylim([z_min - 5, z_max + 5])
        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Z Accel (m/s²)", fontsize=12, fontweight="bold")
        ax.set_title(f"Sensor {SENSOR} — Z-axis Acceleration", fontsize=11)
        ax.tick_params(axis="both", labelsize=11)
        ax.set_facecolor((0.9, 0.9, 0.9, 0.6))

        canvas.draw()
        img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(canvas.get_width_height()[::-1] + (4,))
        graph_rgb  = img[:, :, :3]
        alpha_mask = img[:, :, 3] / 255.0

        gh, gw = graph_rgb.shape[:2]
        ix, iy = 10, out_h - gh - 10

        if iy >= 0 and ix + gw <= out_w:
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
