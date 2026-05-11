# Backboard accelerometer: goaltend vs. legal (close-call support)

This package turns **tri-axial backboard accelerometer** recordings into a **binary assist** for reviewers: *legal block / contact* versus *goaltend*, with emphasis on **marginal (“close”) plays** where video alone is ambiguous.

**Production classifier:** **AdaBoost** on shallow, class-balanced decision trees, trained on **engineered motion features** (spectral shape of direction-change signals plus time-domain shape cues). It was selected after a broad model sweep and a focused hyperparameter search; see **Methodology** and **Results**.

**Audience note:** This document is written for **basketball operations and replay-center stakeholders** (e.g. NBA League Office). It states what the system does, how it is evaluated, and how to interpret the numbers—without assuming a machine-learning background.

---

## Executive summary

| Question | Answer |
|----------|--------|
| **What does the tool output?** | For each clip: predicted class (**legal** vs **goaltend**) and class probabilities from cross-validated models. |
| **What data does it use?** | (1) **Clean reference clips** from segmented folders (clear legal vs goaltend examples). (2) **Labeled close-call trials** from `Close Calls/` with league-reviewed ground truth (e.g. `cleaned_ground_truth.csv`). |
| **Primary accuracy metric (close calls)** | **~74%** out-of-fold accuracy on labeled **close-call** clips when each fold is trained on **all** reference segmented clips plus other close calls—mirroring a deployment where the model can use the full reference library. |
| **Blended “all clips” metric** | **~84%** when every clip (reference + close call) is evaluated with strict pooled cross-validation; reference clips are easier and pull this number up; close-call-only accuracy in *that* stricter protocol is lower (~65%). **These two numbers are not interchangeable** (see **Interpreting the metrics**). |
| **Court use** | Supportive analytics only—not a substitute for rules expertise, crew chief judgment, or official replay protocols. |

---

## Repository layout

| Path | Purpose |
|------|---------|
| `src/goaltend_close_call/` | Package: ingestion (`sensor_io`), features (`fusion_features`, `shape_time_features`), **main model** (`close_call_model`), CV utilities (`close_call_cv`), labels (`close_call_labels`) |
| `data/` | Segmented class folders, `Close Calls/`, label CSVs (`close_calls_labels.csv`, `cleaned_ground_truth.csv`, …) |
| `outputs/` | Generated OOF prediction CSVs (typically gitignored except `.gitkeep`) |
| `notebooks/` | Exploratory analysis |
| `syncing_video_data/` | Video / IMU alignment utilities (separate from classification) |

---

## Methodology

### 1. Sensor inputs and fair comparison across clips

Each CSV provides **Latest** tri-axial acceleration for one or two mounted sensors. The pipeline:

- Loads **sensor 1 + sensor 2** for standard (e.g. blocks) folders, and **sensor 1 + sensor 3** for **Goaltends** exports so both channels represent comparable backboard physics (`sensor_io.load_recording_csv`).
- **Crops ~1 s** centered on the **peak** combined acceleration magnitude so each clip highlights the contact window (`crop_peak_window`).

### 2. Features (hand-crafted, interpretable family)

Rather than feeding raw waveforms to a black box only, the **default production path** uses **fusion features** (`extract_fusion_features`):

- **Direction-change spectrograms:** Acceleration vectors are **unit-normalized** so overall “hard vs soft” hit size is not the dominant cue; the model uses **how the direction of the acceleration vector evolves** and summaries of its **frequency content** (centroid, band energy, rolloff, etc.).
- **Shape_* time-domain cues:** Peak structure on **normalized** magnitude envelopes, symmetry, cross-sensor lag / correlation—capturing **timing and pulse shape** of the event.

Features are **standardized per cross-validation fold** so scale does not favor one sensor or one arena export.

**Why this matters for the League:** The design intent is to approximate **kinematic patterns** of rim/backboard response, not to memorize a single “peak g” threshold that would confuse legal hard contacts with goaltends.

### 3. Classifier: AdaBoost (champion)

**AdaBoost** combines many **weak** decision trees (here: shallow trees with **class-balanced** weighting). It performed best on **close-call** evaluation among a wide sklearn benchmark (logistic regression, random forests, gradient boosting, k-NN, neural nets, etc.) and was **further tuned** with `adaboost_opt.py` (random search over tree depth, leaf sizes, number of estimators, learning rate).

Hyperparameters are centralized in `close_call_model.py` (env overrides prefixed with `GOALTEND_ADA_*` and `GOALTEND_ADABOOST_RANDOM_STATE`).

### 4. Labels

- **Reference (segmented) clips:** Folder names encode **legal** vs **goaltend** supervision.
- **Close calls:** Rows need a **filename** and a binary **ground truth** (`legal` / `block` vs `goaltend`). Ambiguous rows are skipped (`close_call_labels.py`).

Use `GOALTEND_LABELS_PATH` to point at the active label file (e.g. `data/cleaned_ground_truth.csv`).

---

## Evaluation design (no cheating on labels)

All reported accuracies use **cross-validation**: the model is **never** tested on a clip it was trained on. Two protocols are implemented:

### A. Close-call protocol (headline for “did we learn borderline plays?”)

- Split only the **close-call** set into stratified folds.
- Each training fold uses **all** segmented reference clips **plus** the close calls not in the test fold.
- **Report accuracy only on the held-out close-call fold.**

This matches the practical question: *If we keep a large labeled reference library, how often do we get the close call right?*

**Observed (cleaned ground truth, ~89 usable close calls, 5 folds):** **~74%** OOF accuracy for the champion AdaBoost (exact value depends slightly on seed; run locally to reproduce).

### B. Pooled protocol (strict “every row treated the same”)

- Concatenate **reference + close-call** rows.
- Stratified K-fold on **all** 190 clips; each fold trains on ~80% of **both** pools.

**Observed:** **~84%** overall; **reference-only slice ~100%** in that run; **close-call slice ~65%**—because the model no longer always has the **full** reference library in training for every fold.

**Important:** **Do not** compare 84% to 74% as “improvement.” They answer different operational questions.

---

## Interpreting results for decision-makers

1. **Close-call ~74%** means roughly **three in four** reviewed borderline clips are classified correctly **under CV**, when the model may use the **entire** reference library during training. Errors are still material: this is a **decision-support** statistic, not certification-ready automation.

2. **Probabilities** (`P_goaltend`, `P_legal` in output CSVs) express **model confidence**, not officiating probability. Calibration can be improved (e.g. isotonic regression) if the League wants probability bands for UI.

3. **Sample size:** ~89 close calls is enough to **rank models** and catch gross failure, but confidence intervals on accuracy are **non-trivial**. Expanding labeled close calls will stabilize metrics.

4. **Generalization:** Performance on **new arenas, mounts, or export formats** should be revalidated before reliance in new contexts.

---

## Setup

```bash
python -m venv .venv
source .venv/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt   # installs package + notebook extras
```

---

## Run the production model

From `nba-goaltend-project/` with `src` on `PYTHONPATH`:

```bash
export GOALTEND_LABELS_PATH="$PWD/data/cleaned_ground_truth.csv"   # or your label file
PYTHONPATH=src python -m goaltend_close_call.close_call_model
```

**Outputs:**

- `outputs/close_calls_oof_predictions.csv` — close-call protocol OOF predictions.
- `outputs/pooled_oof_predictions.csv` — pooled protocol OOF predictions (all clips).

**Default training mode:** segmented reference **plus** close-call training folds per fold (`GOALTEND_TRAIN_CLOSE_ONLY=0`). Set `GOALTEND_TRAIN_CLOSE_ONLY=1` to train **only** on close calls (usually weaker).

**Default model:** `adaboost` (champion). Override with e.g. `GOALTEND_MODEL=logistic` for interpretable linear coefficients (`outputs/close_calls_logistic_coefficients.csv`).

### Main environment variables

| Variable | Default | Role |
|----------|---------|------|
| `GOALTEND_DATA_DIR` | `<repo>/data` | Data root (segmented folders, `Close Calls/`) |
| `GOALTEND_LABELS_PATH` | `<data>/close_calls_labels.csv` | Label CSV (`GOALTEND_LABELS_PATH` overrides filename) |
| `GOALTEND_OUTPUT_DIR` | `<repo>/outputs` | Prediction CSV output directory |
| `GOALTEND_MODEL` | `adaboost` | `adaboost` (default), `logistic`, `hgb`, `rf` |
| `GOALTEND_TRAIN_CLOSE_ONLY` | `0` | `0` = include all segmented reference clips in each training fold; `1` = close calls only |
| `GOALTEND_CC_CV_SPLITS` | `5` | Stratified fold count (capped by class counts) |
| `GOALTEND_ADABOOST_RANDOM_STATE` | `133742` | RNG for champion AdaBoost (reproducibility) |
| `GOALTEND_ADA_*` | see `close_call_model.py` | Optional overrides for tree depth, `n_estimators`, `learning_rate`, etc. |
| `GOALTEND_WIN_SEC`, `GOALTEND_NPERSEG` | `1.0`, `256` | Crop window length; STFT length for direction-change features |

---

## Hyperparameter search (optional)

- **`python -m goaltend_close_call.adaboost_opt`** — random search over AdaBoost / tree settings; writes `outputs/adaboost_random_search_trials.csv` and `outputs/adaboost_best_params.json`.
- **`python -m goaltend_close_call.close_call_benchmark`** — compares many sklearn models on the **close-call** protocol; writes `outputs/close_call_model_benchmark.csv`.

---

## Experimental: deep sequence model

`deep_sequence_model.py` (requires `pip install -e ".[deep]"`) is a **CNN on raw windows** for research comparison. It is **not** the production path documented above.

---

## Sensor conventions

- **Non-goaltend** segmented CSVs: physical sensors **1 and 2** as `(a1, a2)`.
- **Goaltends - Segmented** files: physical sensors **1 and 3** as `(a1, a2)` (`sensor_io.load_recording_csv`).

---

## License

MIT — see [LICENSE](LICENSE).
