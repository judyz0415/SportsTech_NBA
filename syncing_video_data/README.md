# SportsTech_NBA — video / IMU sync

Scripts to overlay accelerometer traces on basketball video (`code.py`, `code2.py` — one pair per dataset), **`overlay_imu_on_video.py`** (CLI — use with arbitrary folders such as HighFrameVideos), and to scrub frame-by-frame for release labeling (`videoFrameNavigator.py`).

Place matching `.mov` / `.csv` files beside the script you run, or edit the paths at the top of each overlay script.

Video outputs and `.mov` inputs are ignored by git (see repo root `.gitignore`).

## HighFrameVideos (example data folder)

Your high–frame-rate clips can live outside this repo, for example:

`/Users/mariaangellobon/Desktop/2.98_Goaltending Project/HighFrameVideos`

### Frame navigator (release timing)

Run from this directory with the **absolute path** to your `.mp4` / `.mov`:

```bash
cd syncing_video_data
python videoFrameNavigator.py "/Users/mariaangellobon/Desktop/2.98_Goaltending Project/HighFrameVideos/C0186.MP4"
```

Release markers are written next to the video: `<name>_release_frames.txt`.

### IMU overlay (`overlay_imu_on_video.py`)

Auto mode **subtracts each channel’s own median** first, so a channel sitting near **0 m/s²** and one near **≈10 m/s²** (gravity on that axis) are compared **after removing that DC tilt**.  

Then it scores each channel (**default `--axis-pick robust-mad`**): spike strength is **peak |a − median| divided by a robust spread** (MAD / std / percentile range), so a quiet axis isn’t drowned out by a noisier one. Legacy behaviour: **`--axis-pick max-deviation`** = largest raw |a − median| only.

It **aligns a chosen IMU peak** (CSV seconds) to the video time **`--sync-peak-at-video`**.

**Two hits (backboard then rim):** chronologically the **first** impulse is typically **backboard** and the **second** often **rim** — on typical rig orientations the **rim** excursion can **still be largest** on the plotted axis (`|a − median|`). This script merges nearby local maxima and ranks separated peaks **by magnitude** (not arrival time): **`--csv-align-peak-index 0`** aligns the **strongest** bump (often rim), **`1`** the **second-strongest** (often the smaller backboard hit). **`--csv-align-window`** trims the CSV range to bracket that pair / cut pre-shot clutter. Prefer **`--manual-knock`** if you pick **rim times by eye** in both streams.

**Timeline:** playback time defaults to **frame 0, 1, 2, … × (1/FPS)** (many MP4s mis-report `CAP_PROP_POS_FRAMES`; that was skewing overlays). Use **`--use-opencv-pos-frames`** only if you intentionally want the old behaviour.

**FPS:** If the file is truly high-speed but metadata FPS is low, pass **`--timeline-fps 960`** (or measured rate) and derive **`--sync-peak-at-video`** with the same FPS convention you used when labeling the video.

```bash
cd syncing_video_data
python overlay_imu_on_video.py \
  --video "/Users/mariaangellobon/Desktop/2.98_Goaltending Project/HighFramevideos/closecalls4.MP4" \
  --csv "/Users/mariaangellobon/Desktop/2.98_Goaltending Project/HighFramevideos/close_calls_4.csv" \
  --time-col "Latest: Time (s)" \
  --sync-peak-at-video 6.506 \
  --csv-align-window 3.6 5.2 \
  --csv-align-peak-index 0 \
  --timeline-fps 960 \
  --xlim 4 10
```

Tune `csv-align-window` from a quick plot of the CSV; if two peaks stay merged, lower **`--min-peak-sep`** (default 0.07 s).

Also: `--acc-col "…"`, `--axis-pick max-deviation`, `--no-crop`, `--large-labels`, `--output /path/out.mp4`.

Manual rim sync: `--manual-knock --imu-knock <rim_CSV_s> --video-knock <rim_video_s>` (see help).

Defaults write `<video_stem>_overlay.mp4` next to the input video.
