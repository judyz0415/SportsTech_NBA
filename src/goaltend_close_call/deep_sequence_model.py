"""
**Experimental / research** — optional deep baseline, not the league-office production model.

End-to-end **1D convolutional network** on cropped, resampled two-sensor accelerometer windows.

Uses the same ingestion as scalar fusion (`load_recording_csv`, `crop_peak_window`) then
linear interpolation to fixed length × 6 channels (sensor1 XYZ + sensor2 XYZ). Stratified K-fold on
**segmented ∪ labeled close-call** clips only (every row evaluated out-of-fold).

Install: ``pip install -e ".[deep]"`` or ``pip install torch``.

Writes ``outputs/deep_sequence_oof_predictions.csv``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from .close_call_labels import load_usable_close_call_binary_labels
from .paths import data_root, labels_csv_path, outputs_dir
from .sensor_io import (
    crop_peak_window,
    discover_segmented_folders,
    estimate_fs,
    load_recording_csv,
)

try:
    import torch
    from torch import nn
except ImportError as err:  # pragma: no cover
    torch = None
    nn = None
    _TORCH_IMPORT_ERR = err


DATA_ROOT = data_root()
OUTPUT_DIR = outputs_dir()
CLOSE_DIR = DATA_ROOT / "Close Calls"
LABELS_PATH = labels_csv_path()

WIN_SEC = float(os.environ.get("GOALTEND_WIN_SEC", "1.0"))
SENSOR_1_ONLY = False
TARGET_LEN = int(os.environ.get("GOALTEND_SEQ_LEN", "512"))
CV_SPLITS = int(os.environ.get("GOALTEND_CC_CV_SPLITS", "5"))
CV_RANDOM_STATE = int(os.environ.get("GOALTEND_CV_RANDOM_STATE", "42"))

if torch is not None:
    torch.manual_seed(CV_RANDOM_STATE)
np.random.seed(CV_RANDOM_STATE)

EPOCHS_MAX = int(os.environ.get("GOALTEND_DEEP_EPOCHS_MAX", "140"))
LR = float(os.environ.get("GOALTEND_DEEP_LR", "4e-4"))
WEIGHT_DECAY = float(os.environ.get("GOALTEND_DEEP_WEIGHT_DECAY", "8e-5"))
DROP = float(os.environ.get("GOALTEND_DEEP_DROPOUT", "0.22"))
VAL_FRAC = float(os.environ.get("GOALTEND_DEEP_VAL_FRAC", "0.18"))
EARLY_STOP = int(os.environ.get("GOALTEND_DEEP_EARLY_STOP", "18"))
BAT = int(os.environ.get("GOALTEND_DEEP_BATCH", "28"))
LABELS = ["legal", "goaltend"]
LEG, GOALT = 0, 1


@dataclass(frozen=True)
class ClipRow:
    path: Path
    y: str
    source: str
    clip_id: str


def _iter_manifest() -> list[ClipRow]:
    rows: list[ClipRow] = []
    for folder, label in discover_segmented_folders(DATA_ROOT):
        if folder.name == "Other Data - Segmented":
            continue
        y = "goaltend" if label == "goaltends" else "legal"
        for p in sorted(folder.glob("*.csv")):
            rows.append(ClipRow(p, y, "segmented", p.name))

    usable = load_usable_close_call_binary_labels(LABELS_PATH, CLOSE_DIR)
    for _, r in usable.iterrows():
        rows.append(
            ClipRow(Path(r["path"]), str(r["y"]), "close_call", str(r["filename"]))
        )
    return rows


def _load_crop_resample(p: Path) -> np.ndarray:
    """Fixed-length × 6 acceleration (two tri-axes, sensor rules from ``sensor_io``)."""
    t, a1, a2 = load_recording_csv(p, sensor_1_only=SENSOR_1_ONLY)
    fs = estimate_fs(t)
    t, a1, a2 = crop_peak_window(t, a1, a2, win_sec=WIN_SEC, fs=fs, sensor_1_only=SENSOR_1_ONLY)
    if len(t) < 2:
        return np.zeros((TARGET_LEN, 6), dtype=np.float32)
    a1 = np.nan_to_num(a1.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    a2 = np.nan_to_num(a2.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    x = np.concatenate([a1, a2], axis=1)
    old_axis = np.linspace(0.0, 1.0, num=x.shape[0])
    new_axis = np.linspace(0.0, 1.0, num=TARGET_LEN)
    out = np.column_stack([np.interp(new_axis, old_axis, x[:, ch]) for ch in range(6)])
    return np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _encode_y(labels: np.ndarray) -> np.ndarray:
    return np.asarray([LABELS.index(s) if s in LABELS else 0 for s in labels])


class ConvAccelNet(nn.Module):
    """Temporal CNN → one logit (positive ⇒ goaltend)."""

    def __init__(self, seq_len: int, *, drop: float = DROP, emb: int = 160) -> None:
        super().__init__()
        assert nn is not None
        del seq_len
        # GroupNorm: stable with tiny batches (stratified CV slices).
        self.net = nn.Sequential(
            nn.Conv1d(6, 48, kernel_size=21, padding=10),
            nn.GroupNorm(8, 48),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 96, kernel_size=15, padding=7),
            nn.GroupNorm(8, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.MaxPool1d(2),
            nn.Conv1d(96, emb, kernel_size=11, padding=5),
            nn.GroupNorm(8, emb),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(drop * 0.5), nn.Linear(emb, 1))

    def forward(self, x):  # (B,C,T) → logits (B,1) goaltend vs legal
        return self.head(self.net(x))


def run() -> dict:
    if nn is None:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for deep_sequence_model. Install with: pip install 'torch>=2.0'"
        ) from _TORCH_IMPORT_ERR

    manifest = _iter_manifest()
    if len(manifest) < 10:
        raise ValueError(f"Too few labeled clips ({len(manifest)}) for pooled deep CV.")

    X_list = [_load_crop_resample(r.path) for r in manifest]
    X = np.stack(X_list, axis=0)  # (N, T, 6)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_str = np.array([r.y for r in manifest])
    y = _encode_y(y_str)
    src = np.array([r.source for r in manifest])
    clip_ids = np.array([r.clip_id for r in manifest])

    vc = pd.Series(y_str).value_counts()
    min_class = int(vc.min())
    if min_class < 2:
        raise ValueError(f"Need ≥2 samples per class; counts: {vc.to_dict()}.")

    k = int(min(CV_SPLITS, min_class))
    if k < 2:
        raise ValueError("Stratified k-fold requires k >= 2.")

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=CV_RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    oof_pred_idx = np.empty(len(X), dtype=np.int64)
    oof_p_goal = np.zeros(len(X), dtype=np.float64)
    oof_fold = np.full(len(X), -1, dtype=np.int32)
    fold_scores: list[float] = []

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(X)), y)):
        Xt_raw, Xtest_raw = X[train_idx], X[test_idx]
        yt, y_test = y[train_idx], y[test_idx]

        # Per-fold channel normalization (train statistics only).
        mu = np.nan_to_num(Xt_raw.mean(axis=(0, 1), dtype=np.float64))
        sig = np.nan_to_num(Xt_raw.std(axis=(0, 1), dtype=np.float64)) + 1e-5
        Xt = np.nan_to_num((Xt_raw - mu) / sig)
        Xte = np.nan_to_num((Xtest_raw - mu) / sig)

        Xt_tr, Xt_val, yt_tr, yt_val = train_test_split(
            Xt,
            yt,
            test_size=VAL_FRAC,
            stratify=yt,
            random_state=CV_RANDOM_STATE + fold_id,
        )

        pos = float((yt_tr == GOALT).sum())
        neg = float(len(yt_tr) - pos)
        pos_weight_val = neg / max(pos, 1.0)

        # (N,C,T)
        Xt_tr_bc = torch.from_numpy(np.swapaxes(Xt_tr, 1, 2).copy()).float().to(device)
        Xt_val_bc = torch.from_numpy(np.swapaxes(Xt_val, 1, 2).copy()).float().to(device)
        yt_tr_t = torch.from_numpy(yt_tr.astype(np.float64)).float().to(device).view(-1, 1)
        yt_val_t = torch.from_numpy(yt_val.astype(np.float64)).float().to(device).view(-1, 1)

        model = ConvAccelNet(TARGET_LEN, drop=DROP).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        crit = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_val], device=device),
        )

        best_bacc = -1.0
        bad = 0
        best_state = None

        for epoch in range(EPOCHS_MAX):
            model.train()
            perm = torch.randperm(len(Xt_tr_bc), device=device)
            for i in range(0, len(perm), BAT):
                batch_idx = perm[i : i + BAT]
                xb = Xt_tr_bc[batch_idx]
                yb = yt_tr_t[batch_idx]
                opt.zero_grad(set_to_none=True)
                logit = model(xb)
                loss = crit(logit, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

            model.eval()
            with torch.no_grad():
                val_logits = []
                for i in range(0, len(Xt_val_bc), BAT):
                    val_logits.append(model(Xt_val_bc[i : i + BAT]))
                val_l = torch.cat(val_logits, dim=0)
                val_p = torch.sigmoid(val_l).cpu().numpy().ravel()
                val_y = yt_val_t.cpu().numpy().ravel()
                val_pred = (val_p >= 0.5).astype(np.int64)
                bacc = float(balanced_accuracy_score(val_y, val_pred))

            if bacc > best_bacc + 1e-5:
                best_bacc = bacc
                bad = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= EARLY_STOP:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        Xte_bc = torch.from_numpy(np.swapaxes(Xte, 1, 2).copy()).float().to(device)
        prob_list: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(Xte_bc), BAT):
                lg = model(Xte_bc[i : i + BAT])
                sg = torch.sigmoid(lg).cpu().numpy()
                prob_list.append(sg)
        p_goal_flat = np.vstack(prob_list).ravel()
        pred = (p_goal_flat >= 0.5).astype(np.int64)
        acc = float(np.mean(pred == y_test))
        fold_scores.append(acc)

        for j_local, j_global in enumerate(test_idx):
            oof_pred_idx[j_global] = pred[j_local]
            oof_p_goal[j_global] = p_goal_flat[j_local]
            oof_fold[j_global] = fold_id

    y_pred_str = np.array([LABELS[GOALT] if i == 1 else LABELS[LEG] for i in oof_pred_idx])
    correct = y_pred_str == y_str
    overall = float(np.mean(correct))

    out_df = pd.DataFrame(
        {
            "source": src,
            "clip_id": clip_ids,
            "fold_id": oof_fold,
            "y": y_str,
            "predicted": y_pred_str,
            "P_legal": 1.0 - oof_p_goal,
            "P_goaltend": oof_p_goal,
            "correct": correct,
        }
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "deep_sequence_oof_predictions.csv"
    out_df.to_csv(out_csv, index=False)

    by_src = {
        sn: float(np.mean(correct[src == sn])) if np.any(src == sn) else float("nan")
        for sn in np.unique(src)
    }

    report = classification_report(y_str, y_pred_str, labels=LABELS, digits=4, zero_division=0)

    return {
        "output_csv": str(out_csv),
        "oof_accuracy_overall": overall,
        "oof_accuracy_by_source": by_src,
        "fold_scores": fold_scores,
        "cv_k": k,
        "n_samples": len(manifest),
        "classification_report": report,
        "oof_df": out_df,
    }


def main_print() -> None:
    r = run()
    print("Conv1d sequence model — pooled segmented ∪ close-call (stratified OOF)")
    assert torch is not None
    print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("Hyperparams:", f"SEQ_LEN={TARGET_LEN} WIN_SEC={WIN_SEC} lr={LR} drop={DROP} early_stop={EARLY_STOP}")
    print(f"Samples: {r['n_samples']}  folds: {r['cv_k']}")
    print(f"Fold OOF accuracy: {[round(a, 4) for a in r['fold_scores']]}")
    print(f"OOF accuracy overall: {r['oof_accuracy_overall']:.4f}")
    print("By source:", {k: round(v, 4) for k, v in r['oof_accuracy_by_source'].items()})
    print("Classification report:\n", r["classification_report"])
    print("Wrote:", r["output_csv"])


if __name__ == "__main__":
    main_print()
