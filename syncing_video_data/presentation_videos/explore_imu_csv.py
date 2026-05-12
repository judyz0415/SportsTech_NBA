# ===========================================================================
# IMU CSV Explorer — find the knock time for syncing
# ===========================================================================
# USAGE
#   python explore_imu_csv.py <path_to_csv>
#
# HOW TO USE
#   - Hover: crosshair + tooltip shows exact time under cursor
#   - Click: drops a red marker, prints the time AND the closest CSV row
#   - R: clear all markers
#   - Close the window: prints a summary of all clicked times
# ===========================================================================

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

if len(sys.argv) != 2:
    print("Usage: python explore_imu_csv.py <path_to_csv>")
    sys.exit(1)

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

time_col = "Latest: Time (s)"
z1_col   = "Latest: Z Acceleration 1 (m/s²)"
z2_col   = "Latest: Z Acceleration 2 (m/s²)"

for col in [time_col, z1_col, z2_col]:
    if col not in df.columns:
        print(f"ERROR: column not found: '{col}'")
        print("Available columns:", list(df.columns))
        sys.exit(1)

t  = df[time_col].values
z1 = df[z1_col].values
z2 = df[z2_col].values

# --- Build figure ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.suptitle(
    f"{csv_path.split('/')[-1]}\n"
    "Hover to inspect  |  Click to mark knock time  |  R = clear markers",
    fontsize=11
)

ax1.plot(t, z1, color="dodgerblue", linewidth=1.2, label="Sensor 1 — Z")
ax1.axhline(0, color="black", linewidth=0.6, alpha=0.4)
ax1.set_ylabel("Z Accel 1 (m/s²)", fontsize=11)
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.plot(t, z2, color="darkorange", linewidth=1.2, label="Sensor 2 — Z")
ax2.axhline(0, color="black", linewidth=0.6, alpha=0.4)
ax2.set_ylabel("Z Accel 2 (m/s²)", fontsize=11)
ax2.set_xlabel("Time (s)", fontsize=11)
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# --- State ---
clicked_times = []
marker_artists = []   # list of artists to remove on reset

# --- Crosshair + hover tooltip ---
vline1 = ax1.axvline(color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
vline2 = ax2.axvline(color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
tooltip = ax1.text(
    0, 0, "", fontsize=9, color="black", va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.9),
    zorder=10
)
tooltip.set_visible(False)


def closest_row(click_time):
    idx = (np.abs(t - click_time)).argmin()
    row = df.iloc[idx]
    return idx, row


def on_move(event):
    if event.inaxes not in (ax1, ax2):
        tooltip.set_visible(False)
        fig.canvas.draw_idle()
        return
    x = event.xdata
    if x is None:
        return

    vline1.set_xdata([x])
    vline2.set_xdata([x])

    idx, row = closest_row(x)
    tip = (
        f"t = {x:.4f} s\n"
        f"Closest row {idx}:  t={row[time_col]:.4f} s\n"
        f"  Z1 = {row[z1_col]:.4f}    Z2 = {row[z2_col]:.4f}"
    )
    tooltip.set_text(tip)
    tooltip.set_position((x, ax1.get_ylim()[1]))
    tooltip.set_visible(True)
    fig.canvas.draw_idle()


def on_click(event):
    if event.inaxes not in (ax1, ax2) or event.button != 1:
        return
    x = event.xdata
    if x is None:
        return

    idx, row = closest_row(x)
    exact_t  = row[time_col]
    z1_val   = row[z1_col]
    z2_val   = row[z2_col]

    clicked_times.append(exact_t)

    print(f"\n  Clicked ~{x:.4f} s")
    print(f"  Closest CSV row : {idx}")
    print(f"    Latest: Time (s)                   = {exact_t:.6f}")
    print(f"    Latest: Z Acceleration 1 (m/s²)    = {z1_val:.4f}")
    print(f"    Latest: Z Acceleration 2 (m/s²)    = {z2_val:.4f}")
    print(f"  → Set  IMU_KNOCK_TIME_S = {exact_t:.6f}")

    # Red marker on both axes
    l1 = ax1.axvline(x=exact_t, color="red", linewidth=1.6, linestyle="--", alpha=0.85)
    l2 = ax2.axvline(x=exact_t, color="red", linewidth=1.6, linestyle="--", alpha=0.85)

    # Label on top axis
    label_kw = dict(
        rotation=90, va="bottom", ha="right", fontsize=8, color="red",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")]
    )
    lbl = ax1.text(exact_t, ax1.get_ylim()[1], f" t={exact_t:.4f}s\n row {idx}", **label_kw)

    marker_artists.extend([l1, l2, lbl])
    fig.canvas.draw_idle()


def on_key(event):
    if event.key == "r":
        for a in marker_artists:
            a.remove()
        marker_artists.clear()
        clicked_times.clear()
        print("\n  Markers cleared.")
        fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()

if clicked_times:
    print("\n--- Summary of marked times ---")
    for i, tv in enumerate(clicked_times, 1):
        print(f"  [{i}]  IMU_KNOCK_TIME_S = {tv:.6f}")
