# Data layout

Place **labeled segmented** folders here (each directory name must contain `Segmented`), plus:

- **`Close Calls/`** — one CSV per marginal trial.
- **Label table** — by default `close_calls_labels.csv` with `filename` and **`ground_truth`** (`legal` / `block` / `goaltend`, etc.) and/or **`eyeballed_contact`**. Rows without a usable binary label are skipped.

**Alternative labels:** set `GOALTEND_LABELS_PATH` to an absolute path (e.g. `.../data/cleaned_ground_truth.csv` with columns `File Name` / `Ground Truth`).

Override the data root with **`GOALTEND_DATA_DIR`**.

For methodology, metrics, and how to run the production **AdaBoost** model, see the **[project README](../README.md)**.
