# Data layout

Reference and close-call accelerometer CSVs live under two top-level folders:

## `goaltends/`

- **`segmented/`** — obvious goaltend reference clips (formerly `Goaltends - Segmented/`).
- **`close_calls/`** — marginal trials labeled **goaltend** in the active label CSV (e.g. `cleaned_ground_truth.csv`).

## `legal contacts/`

Legal / non-goaltend supervision and marginal **block** (or legal) close calls:

- **`blocks/`**, **`hand_on_backboard/`** — class-specific reference clips.
- **`ball_on_rim/`** — ball-on-rim **reference** clips (migrated from `Ball on Rim - Segmented/` when present).
- **`other_data/`** — other legal-contact reference clips.
- **`close_calls/`** — marginal trials labeled **block** / **legal** in the label table.

**Label table** — by default `close_calls_labels.csv` with `filename` and **`ground_truth`** (`legal` / `block` / `goaltend`, etc.) and/or **`eyeballed_contact`**. Rows without a usable binary label are skipped.

**Alternative labels:** set `GOALTEND_LABELS_PATH` to an absolute path (e.g. `.../data/cleaned_ground_truth.csv` with columns `File Name` / `Ground Truth`).

Override the data root with **`GOALTEND_DATA_DIR`**.

**Migrating from the old layout** (`* - Segmented/`, flat `Close Calls/`): run `python scripts/reorganize_data_layout.py` from the `nba-goaltend-project` directory (requires `pandas`).

For methodology, metrics, and how to run the production **AdaBoost** model, see the **[project README](../README.md)**.
