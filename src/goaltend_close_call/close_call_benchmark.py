"""
Benchmark diverse classifiers for **close-call-only OOF accuracy**.

Training protocol matches ``close_call_cv.stratified_kfold_eval_close_calls`` with
``include_segmented=True``: every fold trains on **all segmented clips** plus **close-call
training folds**; scores are computed **only on the held-out close-call fold**
(the regime that maximizes transferable signal for borderline clips).

Fusion scalar features (`extract_fusion_features`) are sanitized (NaNs/inf).

Also reports **threshold-tuned** OOF accuracy: best single threshold on ``P(goaltend``
over OOF pooled predictions (helps calibrate linear/noisy probabilistic outputs).

Examples::

    PYTHONPATH=src GOALTEND_LABELS_PATH=data/cleaned_ground_truth.csv \\
        python -m goaltend_close_call.close_call_benchmark
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

N_PARALLEL = max(1, int(os.environ.get("GOALTEND_BENCHMARK_N_JOBS", "1")))

from .close_call_cv import stratified_kfold_eval_close_calls
from .close_call_model import (
    CV_N_SPLITS,
    CV_RANDOM_STATE,
    HGB_PARAMS,
    build_base_binary,
    build_usable_close_calls_df,
    make_adaboost_champion_pipeline,
)
from .paths import outputs_dir

OUTPUT_DIR = outputs_dir()


class MLPEncoderWrapper(BaseEstimator, ClassifierMixin):
    """
    Encode string labels before MLPClassifier (avoids brittle ``np.isnan`` dtype checks).

    ``classes_`` and ``predict_proba`` column order match ``LabelEncoder.classes_``.
    """

    def __init__(self, **mlp_kw: Any) -> None:
        self.mlp_kw = mlp_kw
        self.mlp_: MLPClassifier | None = None
        self.label_encoder_: LabelEncoder | None = None

    def fit(self, X, y):  # noqa: ANN001
        enc = LabelEncoder()
        yi = enc.fit_transform(np.asarray(y, dtype=str))
        self.label_encoder_ = enc
        self.classes_ = enc.classes_
        self.mlp_ = MLPClassifier(**self.mlp_kw)
        self.mlp_.fit(X, yi.astype(np.int64))
        return self

    def predict_proba(self, X):  # noqa: ANN001
        assert self.mlp_ is not None
        return np.asarray(self.mlp_.predict_proba(X))

    def predict(self, X):  # noqa: ANN001
        xi = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[xi.astype(int)]


def _sanitize_dfs(df_seg: pd.DataFrame, df_cc: pd.DataFrame, feat_cols: list[str]) -> None:
    for df in (df_seg, df_cc):
        raw = df[feat_cols].values.astype(np.float64)
        df[feat_cols] = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)


def _thresh_tuned_acc(y_str: np.ndarray, p_goal: np.ndarray, grid: int = 201) -> float:
    y = np.array([1 if s == "goaltend" else 0 for s in y_str], dtype=np.int32)
    if len(np.unique(y)) < 2:
        return float("nan")
    best = 0.0
    ts = np.linspace(0.0, 1.0, grid)
    for t in ts:
        pred = np.where(p_goal >= t, 1, 0).astype(np.int32)
        acc = float(np.mean(pred == y))
        if acc > best:
            best = acc
    return best


def _best_threshold(y_str: np.ndarray, p_goal: np.ndarray, grid: int = 201) -> float:
    y = np.array([1 if s == "goaltend" else 0 for s in y_str], dtype=np.int32)
    best_t, best_acc = 0.5, -1.0
    for t in np.linspace(0.0, 1.0, grid):
        pred = np.where(p_goal >= t, 1, 0).astype(np.int32)
        acc = float(np.mean(pred == y))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t


def _scal_clf(est) -> Pipeline:
    """StandardScaler → estimator (trees ignore scale; linear methods need it)."""
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])


def model_factories(seed: int) -> list[tuple[str, Callable[[], Pipeline]]]:
    return [
        (
            "log_c0.03",
            lambda: _scal_clf(
                LogisticRegression(
                    C=0.03,
                    max_iter=8000,
                    class_weight="balanced",
                    random_state=seed,
                    solver="lbfgs",
                )
            ),
        ),
        (
            "log_c0.25",
            lambda: _scal_clf(
                LogisticRegression(
                    C=0.25,
                    max_iter=8000,
                    class_weight="balanced",
                    random_state=seed,
                    solver="lbfgs",
                )
            ),
        ),
        (
            "log_c1",
            lambda: _scal_clf(
                LogisticRegression(
                    C=1.0,
                    max_iter=8000,
                    class_weight="balanced",
                    random_state=seed,
                    solver="lbfgs",
                )
            ),
        ),
        (
            "log_c4",
            lambda: _scal_clf(
                LogisticRegression(
                    C=4.0,
                    max_iter=8000,
                    class_weight="balanced",
                    random_state=seed,
                    solver="lbfgs",
                )
            ),
        ),
        (
            "sgd_huber_elastic",
            lambda: _scal_clf(
                SGDClassifier(
                    loss="modified_huber",
                    penalty="elasticnet",
                    l1_ratio=0.25,
                    alpha=1e-3,
                    class_weight="balanced",
                    random_state=seed,
                    max_iter=3000,
                    tol=1e-4,
                    n_jobs=N_PARALLEL,
                )
            ),
        ),
        (
            "lin_svc_calibrated",
            lambda: _scal_clf(
                CalibratedClassifierCV(
                    LinearSVC(
                        class_weight="balanced",
                        dual=False,
                        max_iter=6000,
                        random_state=seed,
                        C=0.5,
                    ),
                    method="sigmoid",
                    cv=3,
                )
            ),
        ),
        (
            "gaussian_nb",
            lambda: _scal_clf(GaussianNB()),
        ),
        (
            "qda_reg",
            lambda: _scal_clf(QuadraticDiscriminantAnalysis(reg_param=0.15)),
        ),
        (
            "knn_k11_uniform",
            lambda: _scal_clf(
                KNeighborsClassifier(n_neighbors=11, weights="uniform", n_jobs=-1)
            ),
        ),
        (
            "knn_k9_distance",
            lambda: _scal_clf(
                KNeighborsClassifier(n_neighbors=9, weights="distance", n_jobs=-1)
            ),
        ),
        (
            "mlp_earlystop",
            lambda: _scal_clf(
                MLPEncoderWrapper(
                    hidden_layer_sizes=(96, 48),
                    alpha=8e-3,
                    max_iter=800,
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=25,
                    random_state=seed,
                )
            ),
        ),
        (
            "mlp_wide",
            lambda: _scal_clf(
                MLPEncoderWrapper(
                    hidden_layer_sizes=(160, 80, 40),
                    alpha=0.015,
                    max_iter=900,
                    early_stopping=True,
                    validation_fraction=0.18,
                    n_iter_no_change=25,
                    random_state=seed,
                )
            ),
        ),
        (
            "extra_trees_deep",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        ExtraTreesClassifier(
                            n_estimators=500,
                            max_depth=16,
                            min_samples_leaf=2,
                            max_features="sqrt",
                            class_weight="balanced_subsample",
                            random_state=seed,
                            n_jobs=N_PARALLEL,
                        ),
                    ),
                ]
            ),
        ),
        (
            "extra_trees_shallow",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        ExtraTreesClassifier(
                            n_estimators=800,
                            max_depth=9,
                            min_samples_leaf=6,
                            max_features=None,
                            class_weight="balanced_subsample",
                            random_state=seed,
                            n_jobs=N_PARALLEL,
                        ),
                    ),
                ]
            ),
        ),
        (
            "rf_deep",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=700,
                            max_depth=None,
                            min_samples_leaf=2,
                            class_weight="balanced_subsample",
                            random_state=seed,
                            n_jobs=N_PARALLEL,
                        ),
                    ),
                ]
            ),
        ),
        (
            "rf_med",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=600,
                            max_depth=14,
                            min_samples_leaf=4,
                            class_weight="balanced_subsample",
                            random_state=seed,
                            n_jobs=N_PARALLEL,
                        ),
                    ),
                ]
            ),
        ),
        (
            "rf_shallow_wide",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=900,
                            max_depth=7,
                            min_samples_leaf=8,
                            class_weight="balanced_subsample",
                            random_state=seed,
                            n_jobs=N_PARALLEL,
                        ),
                    ),
                ]
            ),
        ),
        (
            "hgb_default",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", HistGradientBoostingClassifier(**{**HGB_PARAMS, "random_state": seed})),
                ]
            ),
        ),
        (
            "hgb_cap",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_iter=800,
                            learning_rate=0.05,
                            max_depth=14,
                            min_samples_leaf=10,
                            l2_regularization=0.12,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
        (
            "hgb_conservative",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_iter=500,
                            learning_rate=0.08,
                            max_depth=6,
                            min_samples_leaf=20,
                            l2_regularization=0.35,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
        (
            "gb_legacy",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        GradientBoostingClassifier(
                            n_estimators=220,
                            learning_rate=0.06,
                            max_depth=4,
                            min_samples_leaf=5,
                            subsample=0.85,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
        (
            "vote_soft_lr_rf_et",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        VotingClassifier(
                            estimators=[
                                (
                                    "lr",
                                    LogisticRegression(
                                        C=0.5,
                                        max_iter=8000,
                                        class_weight="balanced",
                                        solver="lbfgs",
                                        random_state=seed,
                                    ),
                                ),
                                (
                                    "rf",
                                    RandomForestClassifier(
                                        n_estimators=380,
                                        max_depth=14,
                                        min_samples_leaf=4,
                                        class_weight="balanced_subsample",
                                        random_state=seed + 17,
                                        n_jobs=N_PARALLEL,
                                    ),
                                ),
                                (
                                    "et",
                                    ExtraTreesClassifier(
                                        n_estimators=380,
                                        max_depth=14,
                                        min_samples_leaf=3,
                                        max_features="sqrt",
                                        class_weight="balanced_subsample",
                                        random_state=seed + 19,
                                        n_jobs=N_PARALLEL,
                                    ),
                                ),
                            ],
                            voting="soft",
                            n_jobs=N_PARALLEL,
                        ),
                    ),
                ]
            ),
        ),
        (
            "adaboost_champion",
            lambda: make_adaboost_champion_pipeline(),
        ),
        (
            "adaboost_stump_prior730",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        AdaBoostClassifier(
                            estimator=DecisionTreeClassifier(
                                max_depth=2,
                                min_samples_leaf=8,
                                class_weight="balanced",
                                random_state=seed + 11,
                            ),
                            n_estimators=180,
                            learning_rate=0.7,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
        (
            "bagging_deep_tree",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        BaggingClassifier(
                            estimator=DecisionTreeClassifier(
                                max_depth=10,
                                min_samples_leaf=4,
                                class_weight="balanced",
                                random_state=seed + 3,
                            ),
                            n_estimators=60,
                            max_samples=0.85,
                            max_features=0.75,
                            random_state=seed,
                            n_jobs=N_PARALLEL,
                        ),
                    ),
                ]
            ),
        ),
    ]


def run_benchmark(
    *,
    n_splits: int | None = None,
    random_state: int = CV_RANDOM_STATE,
    save_best_oof_csv: bool = True,
) -> tuple[pd.DataFrame, str | None]:
    """
    Returns (results table sorted by ``oof_thr_tuned`` desc), best model name.
    """
    k = n_splits if n_splits is not None else CV_N_SPLITS
    df_base, feat_cols = build_base_binary()
    df_cc = build_usable_close_calls_df(feat_cols)
    _sanitize_dfs(df_base, df_cc, feat_cols)

    rows: list[dict] = []
    best_name: str | None = None
    best_thr_score = -1.0
    best_secondary = -1.0
    best_oof: pd.DataFrame | None = None
    best_thresh = 0.5

    for name, mk in model_factories(random_state):
        df_snap: pd.DataFrame | None = None
        try:
            r = stratified_kfold_eval_close_calls(
                df_base,
                df_cc,
                feat_cols,
                make_pipeline=mk,
                include_segmented=True,
                n_splits=k,
                random_state=random_state,
            )
            df_snap = r["oof_predictions_df"]
            y_cc = df_snap["y"].values
            pgoal = df_snap["P_goaltend"].values.astype(np.float64)
            pgoal = np.nan_to_num(
                np.asarray(pgoal, dtype=np.float64), nan=0.5, posinf=1.0, neginf=0.0
            )
            thr_tuned = _thresh_tuned_acc(y_cc, pgoal)
            best_t = _best_threshold(y_cc, pgoal)
            row = {
                "model": name,
                "oof_acc_0.5": float(r["oof_accuracy_close_calls"]),
                "oof_acc_thr_tuned": float(thr_tuned),
                "best_threshold": float(best_t),
                "fold_mean": float(np.mean(r["fold_accuracies_close_calls_only"])),
                "fold_std": float(np.std(r["fold_accuracies_close_calls_only"])),
                "error": "",
            }
            oof_acc_plain = row["oof_acc_0.5"]
        except Exception as exc:  # pragma: no cover
            row = {
                "model": name,
                "oof_acc_0.5": float("nan"),
                "oof_acc_thr_tuned": float("nan"),
                "best_threshold": float("nan"),
                "fold_mean": float("nan"),
                "fold_std": float("nan"),
                "error": str(exc)[:200],
            }
            thr_tuned = float("nan")
            oof_acc_plain = float("nan")

        rows.append(row)
        if row["error"] or df_snap is None:
            continue
        st = thr_tuned if np.isfinite(thr_tuned) else -1.0
        sp = float(oof_acc_plain) if np.isfinite(oof_acc_plain) else -1.0
        tie_better = (abs(st - best_thr_score) < 1e-9 and sp > best_secondary + 1e-9) or (
            st > best_thr_score + 1e-9
        )
        if tie_better:
            best_thr_score = st
            best_secondary = sp
            best_name = name
            best_thresh = float(row["best_threshold"]) if np.isfinite(row["best_threshold"]) else 0.5
            yo = df_snap["y"].values
            pg = df_snap["P_goaltend"].values.astype(np.float64)
            pred_idx = np.where(pg >= best_thresh, "goaltend", "legal")
            best_oof = df_snap.copy()
            best_oof["predicted_thr_tuned"] = pred_idx
            best_oof["correct_thr_tuned"] = pred_idx == yo

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["oof_acc_thr_tuned", "oof_acc_0.5"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "close_call_model_benchmark.csv"
    out.to_csv(csv_path, index=False)

    if save_best_oof_csv and best_oof is not None and best_name:
        outp = OUTPUT_DIR / "close_calls_oof_predictions_benchmark_best.csv"
        meta = (
            f"# best_model={best_name}; threshold_on_P_goaltend={best_thresh:.6f}; "
            f"protocol=segmented_union_cc_train_folds_close_call_test\n"
        )
        outp.write_text(meta + best_oof.to_csv(index=False))
    return out, best_name


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark classifiers on close-call OOF accuracy.")
    p.add_argument(
        "--splits",
        type=int,
        default=None,
        help=f"Stratified folds (default: env GOALTEND_CC_CV_SPLITS or {CV_N_SPLITS})",
    )
    args = p.parse_args()
    table, best = run_benchmark(n_splits=args.splits)
    print("Close-call OOF benchmark (train = all segmented + CC train folds per fold)\n")
    pd.set_option("display.max_rows", 40)
    pd.set_option("display.width", 120)
    print(table.to_string(index=False))
    print("\nBest by threshold-tuned OOF:", best)
    print("Wrote:", OUTPUT_DIR / "close_call_model_benchmark.csv")
    if best:
        bp = OUTPUT_DIR / "close_calls_oof_predictions_benchmark_best.csv"
        if bp.exists():
            print("Best model OOF (threshold-tuned):", bp)


if __name__ == "__main__":
    main()
