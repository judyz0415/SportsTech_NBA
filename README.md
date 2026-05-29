# NBA Goaltend Detection — Backboard IMU + Machine Learning

> **Built a sensor-driven ML system that classifies goaltend calls from backboard accelerometer data — with 74% accuracy on the contested, borderline plays that directly affect game outcomes, and sub-20ms real-time inference for live game use.**

---

## Why This Matters to a Front Office

A missed or incorrectly awarded goaltend call is worth **2–3 points and a possession** — among the highest-leverage single-call errors in basketball officiating. Close-call goaltends are routinely the subject of coach's challenges and league reviews, yet no sensor-based analytical layer exists to support those decisions.

This project builds that layer: a machine learning pipeline that reads raw vibration data off a backboard-mounted accelerometer and outputs a probability that a play is a goaltend — fast enough to inform an in-game challenge and rigorous enough to hold up under post-game review.

**Potential front office applications:**
- **Challenge strategy:** Give coaches a second data stream before burning a challenge on a disputed block/goaltend play.
- **Opponent scouting:** Identify which opposing bigs play near the goaltend boundary — and which defensive schemes create reviewable situations.
- **Officiating analytics:** Track disputed-call patterns across arenas, officials, and game situations to inform front-office advocacy with the league office.
- **Defensive scheme design:** Understand the physical signature of legal vs. illegal backboard contacts to train rim protectors on timing and hand placement.

---

## Results at a Glance

| Metric | Value | Context |
|--------|-------|---------|
| **Close-call accuracy** | **~74%** | On labeled borderline plays — the calls that actually get challenged |
| **Overall accuracy** | **~82.6%** | Across 190 total clips (reference + close calls, pooled CV) |
| **Real-time inference** | **< 20 ms** | AdaBoost track; compatible with live broadcast and shot-clock workflows |
| **Review-mode inference** | **~2.4 s** | ROCKET + TabPFN track; suited for challenge / official review |
| **Dataset** | **190 labeled clips** | 101 reference + 89 close calls with league-reviewed ground truth |

All accuracy figures use **cross-validated hold-out evaluation** — the model is never tested on clips it trained on.

---

## Two-Track System: Real-Time Detection vs. Deep Review

The project ships two complementary models designed for different moments in the game:

| | **AdaBoost (production)** | **ROCKET + TabPFN (review)** |
|--|---------------------------|-------------------------------|
| **Best for** | Live game — flag plays for challenge consideration | Post-play review, official challenge support |
| **Latency** | < 20 ms | ~2.4 s |
| **Accuracy on close calls** | ~74% (close-call CV protocol) | Evaluated via leave-one-out CV on reference set |
| **Strengths** | Fast; interpretable features; trained for borderline plays | Confidence scores; no retraining needed; flags uncertain calls |
| **Limitation** | Weaker on the most extreme edge cases | Too slow for strict real-time |

---

## What Makes the Close-Call Problem Hard

Standard video review struggles with plays where the ball is near the cylinder boundary and contact is brief. The IMU signal captures what video cannot: the **direction and frequency content of the backboard's vibration response** in the 100–200ms window around contact.

Two key engineering choices make this work on borderline plays:

1. **Direction-change spectrograms, not raw g-force.** The model uses *how the acceleration vector rotates*, not how hard the hit was. This separates a legal hard block from a soft goaltend by kinematic signature, not impact magnitude — the same physical insight a physicist would apply to the problem.

2. **Full-reference training for close-call evaluation.** Each close-call fold trains on all 101 reference clips plus the non-held-out close calls. This mirrors real operational use: the model always has access to the full labeled library when evaluating a new disputed play.

---

## Technical Architecture

### Sensor Pipeline
- Tri-axial accelerometers mounted on the backboard, recording at high frequency
- Signal is **cropped to ~1 second** centered on peak combined acceleration magnitude to isolate the contact window
- Dual-sensor channels (sensors 1 + 2 for standard clips; 1 + 3 for legacy goaltend reference) provide independent backboard physics measurements

### Feature Engineering (AdaBoost Track)
- **Direction-change spectrograms:** Unit-normalized acceleration vectors → STFT → frequency centroid, band energy, spectral rolloff
- **Shape features (`shape_*`):** Peak structure, symmetry, cross-sensor lag, correlation on normalized magnitude envelopes
- All features standardized **per CV fold** to prevent scale leakage across different sensor mounts or arena exports

### Classifier Selection
AdaBoost on class-balanced decision trees was selected after a broad sklearn benchmark and hyperparameter search (random search over tree depth, leaf size, `n_estimators`, learning rate). Hyperparameters are centralized and reproducible via environment variable overrides.

### ROCKET + TabPFN Track
- **ROCKET:** 500 random convolutional kernels applied to all 6 acceleration channels; max + mean pooled to a fixed feature vector
- **TabPFN:** Meta-learned transformer pretrained on synthetic tabular datasets; requires no hyperparameter tuning; outputs calibrated class probabilities
- Evaluation: leave-one-out CV (most rigorous for small N) and stratified 80/20 holdout

### Video–IMU Sync Pipeline (`syncing_video_data/`)
A separate pipeline aligns high-speed camera footage with IMU time series for visual validation and analyst review:
- Median-removal and robust MAD-based spike detection identify the contact frame
- Overlay video generated as MP4 for use in presentation and review workflows
- Supports 960fps+ high-speed cameras with manual sync correction

---

## Evaluation Design (No Label Leakage)

Two protocols are reported for the AdaBoost track:

**Close-call protocol** *(headline: how often do we get borderline plays right?)*
- Stratified K-fold CV splits only the close-call set
- Each fold trains on **all** reference clips plus the non-held-out close calls
- Accuracy reported only on the held-out close-call fold
- **Result: ~74%** (74.2% in presentation materials, ~89 usable close calls, 5 folds)

**Pooled protocol** *(strict: every clip treated equally)*
- All 190 clips in a single stratified CV pool
- **Result: ~82.6%** overall; close-call slice ~65% because the full reference library isn't always available in training
- These two numbers answer different operational questions and should not be compared directly

**Permuted-label null test (ROCKET + TabPFN track):** shuffling labels collapses ROC AUC to ~0.5, confirming the model is learning signal from the IMU traces — not exploiting label artifacts or feature leakage.

---

## Repository Layout

| Path | Purpose |
|------|---------|
| `src/goaltend_close_call/` | **`sensors`** (CSV load, crop), **`spectrogram`** / **`shape`** (features), **`fusion`**, **`model`** (AdaBoost + CV), **`cv`**, **`labels`**, **`paths`** |
| `src/goaltend_tabpfn/` | ROCKET + TabPFN pipeline: classify, holdout, wrong-call analysis, label-null test |
| `data/` | `goaltends/` and `legal contacts/` trees (reference + close-call CSVs); label CSVs (`cleaned_ground_truth.csv`, etc.) |
| `outputs/` | OOF prediction CSVs, TabPFN analysis plots, fitted pipeline artifacts |
| `notebooks/` | Exploratory analysis: spectrogram visualization, sensor inspection |
| `syncing_video_data/` | Video / IMU alignment and overlay tools (see `syncing_video_data/README.md`) |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

**TabPFN track** (requires HuggingFace gated weights):
```bash
pip install -e ".[tabpfn]"
export HF_TOKEN=<your_token>     # accept licence at Prior-Labs/TabPFN-v2-clf on HuggingFace
```

---

## Run the Production Model (AdaBoost)

```bash
export GOALTEND_LABELS_PATH="$PWD/data/cleaned_ground_truth.csv"
PYTHONPATH=src python -m goaltend_close_call.model
```

Outputs:
- `outputs/close_calls_oof_predictions.csv` — close-call protocol OOF predictions
- `outputs/pooled_oof_predictions.csv` — pooled protocol OOF predictions

### Key Environment Variables

| Variable | Default | Role |
|----------|---------|------|
| `GOALTEND_DATA_DIR` | `<repo>/data` | Data root |
| `GOALTEND_LABELS_PATH` | `<data>/close_calls_labels.csv` | Label CSV override |
| `GOALTEND_OUTPUT_DIR` | `<repo>/outputs` | Output directory |
| `GOALTEND_MODEL` | `adaboost` | `adaboost`, `logistic`, `hgb`, `rf` |
| `GOALTEND_TRAIN_CLOSE_ONLY` | `0` | `1` = train on close calls only (weaker baseline) |
| `GOALTEND_CC_CV_SPLITS` | `5` | Stratified fold count |
| `GOALTEND_ADA_*` | see `model.py` | Hyperparameter overrides (depth, `n_estimators`, `learning_rate`) |

---

## Run the ROCKET + TabPFN Track

```bash
# Leave-one-out CV (most rigorous)
python -m goaltend_tabpfn.goaltend_classify

# Stratified 80/20 holdout
python -m goaltend_tabpfn.goaltend_classify --holdout

# Wrong-call analysis — misclassified clips with IMU trace plots
python -m goaltend_tabpfn.tabpfn_analysis --split loo --view wrong
python -m goaltend_tabpfn.tabpfn_analysis --split loo --view confident_wrong --threshold 0.75

# Permuted-label sanity check (expect ROC AUC ~0.5)
python -m goaltend_tabpfn.tabpfn_label_null --split holdout --seed 0
```

---

## Sensor Conventions

- **Non-goaltend segmented CSVs:** physical sensors 1 and 2 as `(a1, a2)`
- **Goaltend reference CSVs** (`goaltends/segmented/`): physical sensors 1 and 3 as `(a1, a2)` (legacy mount layout)
- **Close-call CSVs** (both classes): sensors 1 and 2

---

## License

MIT. See [LICENSE](LICENSE).
