"""
Goaltend vs legal model for close-call trials.

**Architecture (fixed feature pipeline + interchangeable classifier)**

1. **Input:** one CSV per clip → crop ~1 s window around peak acceleration → tri-axial
   sensor streams per ``sensor_io`` (Blocks: sensors 1+2; Goaltends folders: 1+3).
2. **Features:** ``fusion_features.extract_fusion_features`` — spectrogram summaries on
   *direction-change* scalars (scale-free) **plus** prefixed ``shape_*`` time-domain cues
   (envelopes, peaks, cross-sensor lag/corr). Typically tens of scalar features.
3. **Transform:** ``StandardScaler`` fit on each CV training fold (zero mean / unit var per feature).
4. **Classifier:** default **AdaBoost** on shallow decision trees (league-office champion;
   see README). Alternatives: ``logistic``, ``hgb``, ``rf`` via ``GOALTEND_MODEL``.

Training pool (default): **segmented + close-call folds per CV** (set ``GOALTEND_TRAIN_CLOSE_ONLY=1``
to use **only** labeled close calls in each fold; no segmented union).

Evaluation: **StratifiedK-fold OOF** on close calls; **pooled OOF** on segmented ∪ close
calls (every row gets a hold-out prediction); optional **coefficient CSV** for logistic
(refit on all labeled close calls for interpretation — see ``run()``).

Sensor conventions match ``sensor_io`` (Blocks etc.: sensors 1+2; Goaltends folder: 1+3).

Writes ``outputs/close_calls_oof_predictions.csv``, ``outputs/pooled_oof_predictions.csv``.
Logistic mode also writes ``outputs/close_calls_logistic_coefficients.csv``.
"""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .close_call_cv import stratified_kfold_eval_close_calls, stratified_kfold_eval_pooled
from .close_call_labels import load_usable_close_call_binary_labels
from .fusion_features import extract_fusion_features
from .paths import data_root, labels_csv_path, outputs_dir
from .sensor_io import discover_segmented_folders, estimate_fs, load_recording_csv, crop_peak_window

DATA_ROOT = data_root()
OUTPUT_DIR = outputs_dir()
WIN_SEC = float(os.environ.get("GOALTEND_WIN_SEC", "1.0"))
NPERSEG = int(os.environ.get("GOALTEND_NPERSEG", "256"))
SENSOR_1_ONLY = False
LABELS_PATH = labels_csv_path()
CLOSE_DIR = DATA_ROOT / "Close Calls"

USE_SYNTHETIC_TRAINING = False

CV_N_SPLITS = int(os.environ.get("GOALTEND_CC_CV_SPLITS", "5"))
CV_RANDOM_STATE = 42

# Default ``0``: train each CV fold with segmented data + close-call training folds.
TRAIN_CLOSE_ONLY = os.environ.get("GOALTEND_TRAIN_CLOSE_ONLY", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)

# Hist gradient boosting: stronger regularization + class balancing for segmented→close-call shift.
HGB_PARAMS = dict(
    max_iter=int(os.environ.get("GOALTEND_HGB_MAX_ITER", "600")),
    learning_rate=float(os.environ.get("GOALTEND_HGB_LEARNING_RATE", "0.06")),
    max_depth=int(os.environ.get("GOALTEND_HGB_MAX_DEPTH", "10")),
    min_samples_leaf=int(os.environ.get("GOALTEND_HGB_MIN_SAMPLES_LEAF", "14")),
    l2_regularization=float(os.environ.get("GOALTEND_HGB_L2", "0.08")),
    class_weight="balanced",
    random_state=42,
)

RF_PARAMS = dict(
    n_estimators=600,
    max_depth=14,
    min_samples_leaf=4,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

LOGISTIC_PARAMS = dict(
    C=float(os.environ.get("GOALTEND_LOGISTIC_C", "1.0")),
    max_iter=int(os.environ.get("GOALTEND_LOGISTIC_MAX_ITER", "5000")),
    class_weight="balanced",
    random_state=42,
    solver="lbfgs",
)

# League-office production classifier: tuned via ``adaboost_opt`` on close-call OOF (see README).
GOALTEND_ADABOOST_RANDOM_STATE = int(os.environ.get("GOALTEND_ADABOOST_RANDOM_STATE", "133742"))
ADA_CHAMPION_N_EST = int(os.environ.get("GOALTEND_ADA_N_ESTIMATORS", "275"))
ADA_CHAMPION_LR = float(os.environ.get("GOALTEND_ADA_LEARNING_RATE", "1.155555391181257"))
ADA_CHAMPION_TREE = dict(
    max_depth=int(os.environ.get("GOALTEND_ADA_MAX_DEPTH", "3")),
    min_samples_leaf=int(os.environ.get("GOALTEND_ADA_MIN_SAMPLES_LEAF", "4")),
    min_samples_split=int(os.environ.get("GOALTEND_ADA_MIN_SAMPLES_SPLIT", "24")),
    class_weight="balanced",
    max_features=None,
)


def make_adaboost_champion_pipeline(random_state: int | None = None) -> Pipeline:
    """StandardScaler + AdaBoost on shallow balanced trees (validated hyperparameters)."""
    rs = GOALTEND_ADABOOST_RANDOM_STATE if random_state is None else int(random_state)
    est = DecisionTreeClassifier(**{**ADA_CHAMPION_TREE, "random_state": rs + 101})
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                AdaBoostClassifier(
                    estimator=est,
                    n_estimators=ADA_CHAMPION_N_EST,
                    learning_rate=ADA_CHAMPION_LR,
                    random_state=rs,
                ),
            ),
        ]
    )


def resolve_model_kind() -> str:
    """One of ``adaboost`` (default), ``logistic``, ``rf``, ``hgb``."""
    explicit = os.environ.get("GOALTEND_MODEL", "").strip().lower()
    if explicit in ("logistic", "lr", "linear"):
        return "logistic"
    if explicit in ("rf", "random_forest"):
        return "rf"
    if explicit in ("hgb", "hist_gradient_boosting", "gradient_boosting"):
        return "hgb"
    if explicit in ("adaboost", "ada", "adaboost_champion"):
        return "adaboost"
    if os.environ.get("GOALTEND_USE_RF", "").strip() in ("1", "true", "yes"):
        return "rf"
    if not explicit:
        return "adaboost"
    return "adaboost"


def make_classifier_pipeline() -> Pipeline:
    kind = resolve_model_kind()
    if kind == "adaboost":
        return make_adaboost_champion_pipeline()
    if kind == "logistic":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(**LOGISTIC_PARAMS)),
            ]
        )
    if kind == "rf":
        p = {**RF_PARAMS}
        p["n_jobs"] = 1
        return Pipeline(
            [("scaler", StandardScaler()), ("clf", RandomForestClassifier(**p))]
        )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(**HGB_PARAMS)),
        ]
    )


def export_logistic_coefficients_csv(
    df_cc: pd.DataFrame,
    feat_cols: list[str],
    out_path: Path,
) -> Path:
    """
    Fit the same logistic pipeline on **all** labeled close calls and write ranked coefficients.

    This refit is for **interpretation** only. OOF accuracy remains in
    ``close_calls_oof_predictions.csv``.
    """
    pipe = make_classifier_pipeline()
    if not isinstance(pipe.named_steps["clf"], LogisticRegression):
        raise TypeError("export_logistic_coefficients_csv requires a logistic classifier")
    X = df_cc[feat_cols].values.astype(np.float64)
    y = df_cc["y"].values
    pipe.fit(X, y)
    lr: LogisticRegression = pipe.named_steps["clf"]
    coef = np.asarray(lr.coef_).ravel()
    if len(coef) != len(feat_cols):
        raise ValueError("coef_ length does not match feature count")
    c0, c1 = lr.classes_[0], lr.classes_[1]
    meta = (
        f"# StandardScaler + LogisticRegression; rows sorted by |coefficient|.\n"
        f"# Positive coefficient → higher log-odds of class {c1!r} vs {c0!r} (after scaling).\n"
    )
    order = np.argsort(np.abs(coef))[::-1]
    rows = [
        {
            "feature": feat_cols[i],
            "coefficient": float(coef[i]),
            "abs_coefficient": float(abs(coef[i])),
        }
        for i in order
    ]
    buf = StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(meta + buf.getvalue())
    return out_path


def _features_for_file(csv_path: Path) -> dict:
    t, a1, a2 = load_recording_csv(csv_path, sensor_1_only=SENSOR_1_ONLY)
    fs = estimate_fs(t)
    t, a1, a2 = crop_peak_window(
        t, a1, a2, win_sec=WIN_SEC, fs=fs, sensor_1_only=SENSOR_1_ONLY
    )
    return extract_fusion_features(
        t, a1, a2, fs=fs, nperseg=NPERSEG, sensor_1_only=SENSOR_1_ONLY
    )


def build_base_binary() -> tuple[pd.DataFrame, list[str]]:
    rows = []
    for folder, label in discover_segmented_folders(DATA_ROOT):
        if folder.name == "Other Data - Segmented":
            continue
        for csv_path in sorted(folder.glob("*.csv")):
            feat = _features_for_file(csv_path)
            feat["y"] = "goaltend" if label == "goaltends" else "legal"
            feat["clip_id"] = csv_path.name
            rows.append(feat)
    df = pd.DataFrame(rows)
    meta = {"y", "clip_id"}
    feat_cols = sorted([c for c in df.columns if c not in meta])
    return df, feat_cols


def build_usable_close_calls_df(feat_cols: list[str]) -> pd.DataFrame:
    """Feature rows for close-call CSVs that have usable binary labels."""
    manifest = load_usable_close_call_binary_labels(LABELS_PATH, CLOSE_DIR)
    rows = []
    for _, r in manifest.iterrows():
        feat = _features_for_file(Path(r["path"]))
        feat["y"] = r["y"]
        feat["filename"] = r["filename"]
        rows.append(feat)
    df = pd.DataFrame(rows)
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df


def segmented_stratified_kfold_accuracy(
    df_base: pd.DataFrame,
    feat_cols: list[str],
    *,
    rf_params: dict | None = None,
    estimator: Pipeline | None = None,
    n_splits: int = CV_N_SPLITS,
    random_state: int = CV_RANDOM_STATE,
) -> dict[str, float | int | list[float]]:
    """
    Stratified k-fold accuracy on the segmented table only (no close calls).
    Uses the same pipeline template as close-call CV unless ``estimator`` is passed.
    ``n_splits`` is lowered if the minority class count is smaller than ``n_splits``.
    """
    params = {**RF_PARAMS, **(rf_params or {})}
    cv_clf_params = {**params, "n_jobs": 1}
    X = df_base[feat_cols].values
    y = df_base["y"].values
    min_class = int(pd.Series(y).value_counts().min())
    k = int(min(n_splits, min_class))
    if k < 2:
        raise ValueError(
            f"Need at least 2 samples per class for CV; got min class count {min_class}"
        )
    if estimator is None:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(**cv_clf_params)),
            ]
        )
    else:
        pipe = estimator
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    return {
        "cv_n_splits_requested": n_splits,
        "cv_n_splits_effective": k,
        "cv_fold_accuracies": [float(s) for s in scores],
        "cv_mean_accuracy": float(np.mean(scores)),
        "cv_std_accuracy": float(np.std(scores)),
    }


def _sanitize_feature_block(df: pd.DataFrame, feat_cols: list[str]) -> None:
    raw = df[feat_cols].values.astype(np.float64)
    df[feat_cols] = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)


def run(rf_params: dict | None = None) -> dict:
    params = {**RF_PARAMS, **(rf_params or {})}
    df_base, feat_cols = build_base_binary()
    df_cc = build_usable_close_calls_df(feat_cols)
    _sanitize_feature_block(df_base, feat_cols)
    _sanitize_feature_block(df_cc, feat_cols)

    clf_template = make_classifier_pipeline()
    if TRAIN_CLOSE_ONLY:
        cv_summary_seg = {
            "cv_n_splits_requested": CV_N_SPLITS,
            "cv_n_splits_effective": 0,
            "cv_fold_accuracies": [],
            "cv_mean_accuracy": float("nan"),
            "cv_std_accuracy": float("nan"),
            "segmented_cv_skipped": True,
        }
    else:
        cv_summary_seg = segmented_stratified_kfold_accuracy(
            df_base, feat_cols, estimator=clf_template
        )

    df_train = df_base
    n_synth = 0
    if USE_SYNTHETIC_TRAINING and not TRAIN_CLOSE_ONLY:
        from .synthetic_close_training import (
            SYNTHETIC_AUGMENTS_PER_SOURCE_FILE,
            build_synthetic_feature_rows,
        )

        def _extract_synth(t, a1, a2, fs: float) -> dict:
            return extract_fusion_features(
                t, a1, a2, fs=fs, nperseg=NPERSEG, sensor_1_only=SENSOR_1_ONLY
            )

        df_syn = build_synthetic_feature_rows(
            DATA_ROOT,
            _extract_synth,
            augments_per_file=SYNTHETIC_AUGMENTS_PER_SOURCE_FILE,
            win_sec=WIN_SEC,
            sensor_1_only=SENSOR_1_ONLY,
            rng=np.random.default_rng(42),
        )
        for c in feat_cols:
            if c not in df_syn.columns:
                df_syn[c] = 0.0
        df_syn = df_syn[feat_cols + ["y"]]
        df_train = pd.concat([df_train, df_syn], ignore_index=True)
        n_synth = len(df_syn)

    seg_side = df_base.iloc[0:0] if TRAIN_CLOSE_ONLY else df_train

    cc_cv = stratified_kfold_eval_close_calls(
        seg_side,
        df_cc,
        feat_cols,
        make_pipeline=make_classifier_pipeline,
        include_segmented=not TRAIN_CLOSE_ONLY,
        n_splits=CV_N_SPLITS,
        random_state=CV_RANDOM_STATE,
    )

    pooled = stratified_kfold_eval_pooled(
        df_base,
        df_cc,
        feat_cols,
        make_pipeline=make_classifier_pipeline,
        n_splits=CV_N_SPLITS,
        random_state=CV_RANDOM_STATE,
    )

    kind = resolve_model_kind()
    if kind == "logistic":
        params_report = {**LOGISTIC_PARAMS, "model": "logistic"}
    elif kind == "rf":
        params_report = {**params, "model": "rf"}
    elif kind == "adaboost":
        params_report = {
            "model": "adaboost",
            **ADA_CHAMPION_TREE,
            "n_estimators": ADA_CHAMPION_N_EST,
            "learning_rate": ADA_CHAMPION_LR,
            "random_state": GOALTEND_ADABOOST_RANDOM_STATE,
        }
    else:
        params_report = {**HGB_PARAMS, "model": "hgb"}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "close_calls_oof_predictions.csv"
    cc_cv["oof_predictions_df"].to_csv(out_path, index=False)
    pooled_path = OUTPUT_DIR / "pooled_oof_predictions.csv"
    pooled["oof_predictions_df"].to_csv(pooled_path, index=False)

    coef_csv_path: str | None = None
    if kind == "logistic":
        coef_csv_path = str(
            export_logistic_coefficients_csv(
                df_cc,
                feat_cols,
                OUTPUT_DIR / "close_calls_logistic_coefficients.csv",
            )
        )

    out = {
        "output_csv": str(out_path),
        "pooled_output_csv": str(pooled_path),
        "oof_accuracy_pooled": pooled["oof_accuracy_pooled"],
        "oof_accuracy_pooled_segmented": pooled["oof_accuracy_segmented"],
        "oof_accuracy_pooled_close_calls": pooled["oof_accuracy_close_calls"],
        "pooled_fold_accuracies": pooled["fold_accuracies"],
        "pooled_cv_n_splits_effective": pooled["cv_n_splits_effective"],
        "pooled_oof_predictions_df": pooled["oof_predictions_df"],
        "params": params_report,
        "model_kind": kind,
        "coefficients_csv": coef_csv_path,
        "n_segmented_train": len(df_base),
        "n_synthetic_train": n_synth,
        "n_close_calls_usable": len(df_cc),
        "oof_accuracy_close_calls": cc_cv["oof_accuracy_close_calls"],
        "close_call_fold_accuracies": cc_cv["fold_accuracies_close_calls_only"],
        "close_call_cv_splits_effective": cc_cv["cv_n_splits_effective"],
        "oof_predictions_df": cc_cv["oof_predictions_df"],
        "train_close_calls_only": TRAIN_CLOSE_ONLY,
        "include_segmented_in_train": cc_cv["include_segmented_in_train"],
    }
    out.update(cv_summary_seg)
    out.update(
        {
            "close_call_cv_n_splits_requested": cc_cv["cv_n_splits_requested"],
            "close_call_cv_n_splits_effective": cc_cv["cv_n_splits_effective"],
            "pooled_cv_n_splits_requested": pooled["cv_n_splits_requested"],
        }
    )
    return out


if __name__ == "__main__":
    r = run()
    print("Classifier:", r["model_kind"])
    print("Params:", r["params"])
    if r.get("coefficients_csv"):
        print("Interpretability (coef refit on all labeled close calls):", r["coefficients_csv"])
    print(
        "Training mode:",
        "labeled close calls only"
        if r["train_close_calls_only"]
        else "segmented (+ optional synth) + close-call folds",
    )
    print(
        "Training rows — segmented:",
        r["n_segmented_train"],
        "synthetic:",
        r["n_synthetic_train"],
    )
    print("Usable labeled close calls:", r["n_close_calls_usable"])
    print("Close-call CV: StratifiedKFold (legal/goaltend mix per fold)")
    print(
        "Close-call CV splits (effective):",
        r["close_call_cv_n_splits_effective"],
        "(requested",
        r["close_call_cv_n_splits_requested"],
        ")",
    )
    print(
        "Per-fold accuracy (close-call test sets):",
        [round(x, 4) for x in r["close_call_fold_accuracies"]],
    )
    print(
        "OOF accuracy on close calls (no leakage):",
        round(r["oof_accuracy_close_calls"], 4),
    )
    if r.get("segmented_cv_skipped"):
        print("Stratified CV on segmented: skipped (training uses close calls only)")
    else:
        print(
            "Stratified CV on segmented only —",
            f"k={r['cv_n_splits_effective']} (requested {r['cv_n_splits_requested']}),",
            "fold accs:",
            [round(x, 4) for x in r["cv_fold_accuracies"]],
        )
        print(
            "CV mean ± std (segmented):",
            round(r["cv_mean_accuracy"], 4),
            "±",
            round(r["cv_std_accuracy"], 4),
        )
    print("Wrote:", r["output_csv"])
    print("\n--- Pooled eval (segmented ∪ close calls, stratified OOF, no leakage) ---")
    print("Wrote:", r["pooled_output_csv"])
    print(
        "Pooled CV splits (effective):",
        r["pooled_cv_n_splits_effective"],
        "(requested",
        r["pooled_cv_n_splits_requested"],
        ")",
    )
    print(
        "Per-fold accuracy (pooled test folds):",
        [round(x, 4) for x in r["pooled_fold_accuracies"]],
    )
    print("OOF accuracy — all pooled rows:", round(r["oof_accuracy_pooled"], 4))
    print(
        "OOF accuracy — segmented only:",
        round(r["oof_accuracy_pooled_segmented"], 4),
    )
    print(
        "OOF accuracy — close calls only (same labels as close-call CV):",
        round(r["oof_accuracy_pooled_close_calls"], 4),
    )
    print("\n" + Path(r["output_csv"]).read_text())
