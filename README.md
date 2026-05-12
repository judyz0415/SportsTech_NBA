# NBA goaltend - backboard IMU ML

Portfolio repository for supervised learning on **backboard-mounted accelerometers** applied to NBA-style goaltend detection. Tri-axial acceleration recordings are used to classify plays as **legal block / contact** vs **goaltend**, with particular emphasis on **marginal ("close") plays** where video alone is ambiguous.

Two modeling tracks share the same data layout (`GOALTEND_DATA_DIR`) and sensor ingestion layer:

| Track | Package | Approach |
|-------|---------|----------|
| **Close-call AdaBoost** | `src/goaltend_close_call/` | Direction-change spectrograms + time-domain shape features fed into **AdaBoost** on class-balanced decision trees. Selected after a broad sklearn benchmark and hyperparameter search. Stratified K-fold CV on labeled close calls. |
| **ROCKET + TabPFN** | `src/goaltend_tabpfn/` | Full six-channel variable-length segmented traces fed into **aeon ROCKET** embeddings then **TabPFN** (HuggingFace gated weights). Leave-one-out and 80/20 holdout CV. Includes wrong-call analysis script. |

---

## Executive summary (close-call AdaBoost track)

| Question | Answer |
|----------|--------|
| **What does the tool output?** | For each clip: predicted class (**legal** vs **goaltend**) and class probabilities from cross-validated models. |
| **What data does it use?** | (1) **Clean reference clips** under `data/goaltends/segmented/` and `data/legal contacts/{blocks,hand_on_backboard,...}/`. (2) **Labeled close-call trials** split into `data/goaltends/close_calls/` vs `data/legal contacts/close_calls/` using league-reviewed ground truth (e.g. `cleaned_ground_truth.csv`). |
| **Primary accuracy metric (close calls)** | **~74%** out-of-fold on labeled **close-call** clips when each fold trains on **all** on-disk reference clips (101) plus other close calls—e.g. **74.2%** in presentation materials (~73% in some fresh repo runs; dependency-sensitive). |
| **Blended "all clips" metric** | **~82.6%** pooled OOF on **190** clips (strict stratified CV); the close-call **slice** in that pooled setup is often **~65%**. Deck materials pair that with **100%** on obvious-call clips and **74.2%** on close calls under the **full-reference** close-call protocol—same dataset, different question than the **~65%** pooled slice. See **Evaluation design**. |
| **Court use** | Supportive analytics only, not a substitute for rules expertise, crew chief judgment, or official replay protocols. |

---

## Repository layout

| Path | Purpose |
|------|---------|
| `src/goaltend_close_call/` | Ingestion (`sensor_io`), features (`fusion_features`, `shape_time_features`), **AdaBoost model** (`close_call_model`), CV utilities (`close_call_cv`), labels (`close_call_labels`), hyperparameter search (`adaboost_opt`), model benchmark (`close_call_benchmark`) |
| `src/goaltend_tabpfn/` | ROCKET + TabPFN: see **TabPFN track scripts** below (`goaltend_classify`, `goaltend_holdout`, `tabpfn_analysis`, `tabpfn_label_null`) |
| `data/` | `goaltends/` and `legal contacts/` trees (reference + close-call CSVs), label CSVs (`close_calls_labels.csv`, `cleaned_ground_truth.csv`, ...) |
| `outputs/` | Generated OOF prediction CSVs, AdaBoost params, benchmark results (typically gitignored except `.gitkeep`) |
| `notebooks/` | Exploratory analysis (spectrogram, sensor visualization) |
| `scripts/` | Data layout utilities (`reorganize_data_layout.py`) |
| `syncing_video_data/` | Video / IMU alignment and overlays (see `syncing_video_data/README.md`; separate from ML classification) |

---

## Methodology: Close-call AdaBoost track

### 1. Sensor inputs and fair comparison across clips

Each CSV provides **Latest** tri-axial acceleration for one or two mounted sensors. The pipeline:

- Loads **sensor 1 + sensor 2** for standard and close-call clips, and **sensor 1 + sensor 3** only for **goaltend reference** exports under `goaltends/segmented/` (legacy: `Goaltends - Segmented`) so both channels represent comparable backboard physics where that mount layout exists (`sensor_io.load_recording_csv`).
- **Crops ~1 s** centered on the **peak** combined acceleration magnitude so each clip highlights the contact window (`crop_peak_window`).

### 2. Features (hand-crafted, interpretable family)

Rather than feeding raw waveforms into a black box, the **default production path** uses **fusion features** (`extract_fusion_features`):

- **Direction-change spectrograms:** Acceleration vectors are **unit-normalized** so overall "hard vs soft" hit size is not the dominant cue; the model uses **how the direction of the acceleration vector evolves** and summaries of its **frequency content** (centroid, band energy, rolloff, etc.).
- **Shape_* time-domain cues:** Peak structure on **normalized** magnitude envelopes, symmetry, cross-sensor lag / correlation, capturing **timing and pulse shape** of the event.

Features are **standardized per cross-validation fold** so scale does not favor one sensor or one arena export.

The design intent is to approximate **kinematic patterns** of rim/backboard response, not to memorize a single "peak g" threshold that would confuse legal hard contacts with goaltends.

### 3. Classifier: AdaBoost (champion)

**AdaBoost** combines many **weak** decision trees (here: shallow trees with **class-balanced** weighting). It performed best on **close-call** evaluation among a wide sklearn benchmark (logistic regression, random forests, gradient boosting, k-NN, neural nets, etc.) and was further tuned with `adaboost_opt.py` (random search over tree depth, leaf sizes, number of estimators, learning rate).

Hyperparameters are centralized in `close_call_model.py` (env overrides prefixed with `GOALTEND_ADA_*` and `GOALTEND_ADABOOST_RANDOM_STATE`).

### 4. Labels

- **Reference (segmented) clips:** Folder names encode **legal** vs **goaltend** supervision.
- **Close calls:** Rows need a **filename** and a binary **ground truth** (`legal` / `block` vs `goaltend`). Ambiguous rows are skipped (`close_call_labels.py`).

Use `GOALTEND_LABELS_PATH` to point at the active label file (e.g. `data/cleaned_ground_truth.csv`).

---

## Methodology: ROCKET + TabPFN track

### 1. Input representation

Each CSV provides all 6 acceleration channels (X1, Y1, Z1, Z2, Y2, X2) for a full segmented trace. Samples are zero-padded to the training set's maximum length per fold so that the test sample's length does not leak into the training representation. Per-sample, per-channel z-score normalization makes the model invariant to absolute acceleration magnitude across different sensors and mounting positions.

### 2. ROCKET feature extraction

**aeon ROCKET** applies N random convolutional kernels to each time series and extracts mean + max-pooled summary statistics per kernel. Default: 500 kernels, 1 ensemble group. ROCKET is fit on training samples only; the same fitted transform is applied to the held-out test sample.

### 3. Classifier: TabPFN

**TabPFN** is a meta-learned transformer pretrained on a large collection of synthetic tabular datasets. It requires no hyperparameter tuning and performs well in the ~100-sample regime typical of leave-one-out CV on segmented clips. Weights are gated on HuggingFace (requires `HF_TOKEN`; see setup below).

### 4. Evaluation protocols

- **Leave-one-out (LOO):** train on all samples except one, test on that one, repeat for every sample. Most rigorous for small datasets.
- **Stratified 80/20 holdout:** single split preserving class proportions; faster for exploratory runs.
- **Wrong-call analysis:** `tabpfn_analysis.py` re-runs the chosen protocol, filters by error type (all wrong, confidently wrong, low-confidence), and saves per-clip IMU trace plots and a summary CSV under `outputs/tabpfn_analysis/`.

### 5. Labels

Folder names encode **legal** vs **goaltend** supervision (same label map as the AdaBoost track). The ROCKET + TabPFN track uses the segmented reference folders; no separate close-call label CSV is required.

---

## Evaluation design (no cheating on labels)

All reported accuracies use **cross-validation**: the model is **never** tested on a clip it was trained on. Two protocols are implemented for the AdaBoost close-call track:

### A. Close-call protocol (headline: "did we learn borderline plays?")

- Split only the **close-call** set into stratified folds.
- Each training fold uses **all** segmented reference clips **plus** the close calls not in the test fold.
- **Report accuracy only on the held-out close-call fold.**

This matches the practical question: if we keep a large labeled reference library, how often do we get the close call right?

**Observed** (cleaned ground truth, ~89 usable close calls, 5 folds, 101-reference library including ball-on-rim): close-call OOF for the champion AdaBoost is typically **~73–74%** (e.g. **74.2%** in presentation deck materials; exact value varies slightly with dependencies—run locally to reproduce).

### B. Pooled protocol (strict: "every row treated the same")

- Concatenate **reference + close-call** rows (190 with the default library: 101 reference + 89 usable close calls).
- Stratified K-fold on **all** 190 rows; each fold trains on ~80% of both pools.

**Observed:** pooled OOF **~82.6%** (matches presentation “overall on 190 clips”); reference-only slice often **~98%** in repo runs; close-call **slice** in this strict protocol **~65%**, because the model no longer always has the **full** reference library in training for every fold. Slightly higher pooled figures (~83.7%) have been seen when the reference mix or dependency versions differed slightly.

**Do not** compare pooled headline accuracy to close-call headline accuracy as "improvement." They answer different operational questions.

### TabPFN track: permuted-label null (ROC AUC ≈ chance)

For the **ROCKET + TabPFN** reference-only track, labels can be **shuffled** so they no longer match each CSV’s IMU trace. Under the same protocol (LOO or stratified 80/20 holdout), **ROC AUC should collapse to ~0.5**; accuracy should sit near a random / majority baseline. That is a quick guard against accidental **label** misalignment or a broken metric path. It is a **complementary** check alongside proper CV—not a full leakage audit (e.g. feature-level or group leakage still needs fold design and domain review).

**Script:** `python -m goaltend_tabpfn.tabpfn_label_null` (default `--split holdout` for speed; use `--split loo` to mirror full leave-one-out). Optional `--n-perm N` repeats with seeds `seed`, `seed+1`, … and prints mean ± std of ROC AUC. The evaluator accepts injected labels via `y_override` in `goaltend_classify.evaluate_goaltend` / `evaluate_goaltend_holdout`.

---

## Interpreting results for decision-makers

1. **AdaBoost (production path).** The same materials report **82.6%** overall accuracy on all **190** clips under **strict pooled** holdout (“every row treated the same” per fold) and **100%** OOF on **obvious-call** clips, meaning mistakes concentrate on the hard subset, not on clear catalogue plays. **Do not** read the **~65%** close-call **slice** under strict pooled CV as the same number as **74.2%**; the protocols differ (see **Evaluation design**). This remains **decision-support**, not certification-ready automation.

2. **ROCKET + TabPFN (presentation track).** Using the **full variable-length trace**, **ROCKET** features, and **TabPFN**, deck results used a stratified **80/20** holdout on the in-repo library and reported **~84.2%** accuracy evaluated on **190** clips, **ROC AUC ~0.94**, and **~85%** on **20** additional fully unseen recordings—useful for **generalization** discussion but **not** the same fold scheme as the AdaBoost close-call headline. Confidence matters: when max class probability was **≥ 0.75**, accuracy was **~96%** (**26/27**); below **0.75** it fell to **~55%** (**6/11**), i.e. near coin-flip—so **high-confidence** predictions are much more trustworthy than low-confidence ones in that analysis.

3. **Probabilities** (`P_goaltend`, `P_legal` in AdaBoost CSVs; analogous scores from TabPFN) express **model confidence**, not officiating probability. The high/low confidence split above is why calibration (e.g. isotonic regression) matters if the League wants UI bands.

4. **Sample size:** ~89 close calls is enough to **rank models** and catch gross failure, but confidence intervals on accuracy are non-trivial. Expanding labeled close calls will stabilize metrics.

5. **Generalization:** TabPFN materials on **20 unseen** clips are encouraging but narrow; performance on new arenas, mounts, or export formats should still be **revalidated** before reliance in new contexts.

---

## Efficiency and accuracy tradeoffs

High-level comparison (from project presentation materials). Latency figures are **order-of-magnitude** and depend on hardware and batching; treat as planning guidance, not SLAs.

| Model | Pros | Cons | Use case |
|-------|------|------|----------|
| **Physics-based** | Easy to understand; simple | Noisy data → false positives | Reference / baseline for goaltending detection |
| **AdaBoost** (this repo’s production path) | Fast — **under ~20 ms** to process and evaluate an unseen trial | Weakest on **very** close / borderline calls | Fast in-game detection paired with high-speed camera workflows |
| **ROCKET + TabPFN** | Built-in confidence scores; **no retraining** needed on new tabular rows at inference | **~2.4 s** inference per trial in deck materials — too slow for strict real-time; less interpretable (“black box” vs hand-crafted features) | **Challenge / review:** officials or analysts trigger after a disputed call |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt   # installs package + notebook extras
```

---

## Run the production model (AdaBoost)

From `nba-goaltend-project/` with `src` on `PYTHONPATH`:

```bash
export GOALTEND_LABELS_PATH="$PWD/data/cleaned_ground_truth.csv"   # or your label file
PYTHONPATH=src python -m goaltend_close_call.close_call_model
```

**Outputs:**

- `outputs/close_calls_oof_predictions.csv` - close-call protocol OOF predictions.
- `outputs/pooled_oof_predictions.csv` - pooled protocol OOF predictions (all clips).

**Default training mode:** segmented reference **plus** close-call training folds per fold (`GOALTEND_TRAIN_CLOSE_ONLY=0`). Set `GOALTEND_TRAIN_CLOSE_ONLY=1` to train **only** on close calls (usually weaker).

**Default model:** `adaboost` (champion). Override with e.g. `GOALTEND_MODEL=logistic` for interpretable linear coefficients (`outputs/close_calls_logistic_coefficients.csv`).

### Main environment variables

| Variable | Default | Role |
|----------|---------|------|
| `GOALTEND_DATA_DIR` | `<repo>/data` | Data root (`goaltends/`, `legal contacts/`; legacy flat `Close Calls/` is still resolved if present) |
| `GOALTEND_LABELS_PATH` | `<data>/close_calls_labels.csv` | Label CSV path override |
| `GOALTEND_OUTPUT_DIR` | `<repo>/outputs` | Prediction CSV output directory |
| `GOALTEND_MODEL` | `adaboost` | `adaboost` (default), `logistic`, `hgb`, `rf` |
| `GOALTEND_TRAIN_CLOSE_ONLY` | `0` | `0` = include all segmented reference clips in each training fold; `1` = close calls only |
| `GOALTEND_CC_CV_SPLITS` | `5` | Stratified fold count (capped by class counts) |
| `GOALTEND_ADABOOST_RANDOM_STATE` | `133742` | RNG for champion AdaBoost (reproducibility) |
| `GOALTEND_ADA_*` | see `close_call_model.py` | Optional overrides for tree depth, `n_estimators`, `learning_rate`, etc. |
| `GOALTEND_WIN_SEC`, `GOALTEND_NPERSEG` | `1.0`, `256` | Crop window length; STFT length for direction-change features |

---

## Hyperparameter search (optional)

- **`python -m goaltend_close_call.adaboost_opt`** - random search over AdaBoost / tree settings; writes `outputs/adaboost_random_search_trials.csv` and `outputs/adaboost_best_params.json`.
- **`python -m goaltend_close_call.close_call_benchmark`** - compares many sklearn models on the **close-call** protocol; writes `outputs/close_call_model_benchmark.csv`.

---

## Experimental: deep sequence model

`deep_sequence_model.py` (requires `pip install -e ".[deep]"`) is a **CNN on raw windows** for research comparison. It is **not** the production path.

---

## ROCKET + TabPFN track (`goaltend_tabpfn`)

**TabPFN track scripts** (all under `src/goaltend_tabpfn/`, run as `python -m goaltend_tabpfn.<module>`):

| Module | Role |
|--------|------|
| `goaltend_classify` | Main LOO run (default) or `--holdout` 80/20; optional timestamped report under `outputs/tabpfn_runs/` |
| `goaltend_holdout` | Fit full train split, save `outputs/fitted_tabpfn_pipeline.joblib`; `--test` for `data/test/` |
| `tabpfn_analysis` | Wrong-call / confidence views + IMU plots → `outputs/tabpfn_analysis/` |
| `tabpfn_label_null` | Permuted labels vs traces → ROC AUC ~0.5 sanity check |

**Setup:** TabPFN weights are gated on HuggingFace. Accept the licence at [Prior-Labs/TabPFN-v2-clf](https://huggingface.co/Prior-Labs/TabPFN-v2-clf), create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then:

```bash
pip install -e ".[tabpfn]"
export HF_TOKEN=<your_token>
```

**Run:**

```bash
# Leave-one-out CV (default)
python -m goaltend_tabpfn.goaltend_classify

# Stratified 80/20 holdout
python -m goaltend_tabpfn.goaltend_classify --holdout

# Permuted labels (expect ROC AUC ~0.5); default holdout is fast
python -m goaltend_tabpfn.tabpfn_label_null --split holdout --seed 0

# Dedicated holdout + save fitted pipeline to outputs/fitted_tabpfn_pipeline.joblib
python -m goaltend_tabpfn.goaltend_holdout

# Predict unseen CSVs in data/test/ using the saved pipeline
python -m goaltend_tabpfn.goaltend_holdout --test
```

**Inspect wrong or confidence-bucketed calls** (writes `filtered.csv`, `summary.txt`, `figures/*.png` under `outputs/tabpfn_analysis/`):

```bash
# All misclassified samples
python -m goaltend_tabpfn.tabpfn_analysis --split loo --view wrong

# Errors where model was >= 75% confident in its wrong prediction
python -m goaltend_tabpfn.tabpfn_analysis --split loo --view confident_wrong --threshold 0.75

# Samples where max class probability was below threshold (ambiguous)
python -m goaltend_tabpfn.tabpfn_analysis --split loo --view low_confidence --threshold 0.75
```

Use `--split holdout` to run the same views on the stratified holdout test split instead of LOO.

---

## Sensor conventions

- **Non-goaltend** segmented CSVs: physical sensors **1 and 2** as `(a1, a2)`.
- **Paths under `goaltends/segmented/`** (and legacy goaltend reference folders): physical sensors **1 and 3** as `(a1, a2)`. **Close-call** CSVs under `goaltends/close_calls/` or `legal contacts/close_calls/` use **sensors 1 and 2** (`sensor_io.load_recording_csv`).

---

## License

MIT. See [LICENSE](LICENSE).
