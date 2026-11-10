"""Joint-level erosion metrics, computed on native SvdH scores.

Definitions follow the RAM-H1200 benchmark (evaluations/metrics.py): BACC is
macro recall over classes, QWK uses the ordered label positions (so grade 5 is
one step above grade 3), MAE and within-1 use the raw scores.
"""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from .data import LEVEL4, NATIVE_SCORES


def compute_metrics(gt_scores, pred_scores, score_values=NATIVE_SCORES):
    gt = np.asarray(gt_scores, dtype=int)
    pred = np.asarray(pred_scores, dtype=int)
    labels = list(score_values)
    cm = confusion_matrix(gt, pred, labels=labels)
    support = cm.sum(axis=1)
    recall = np.divide(np.diag(cm), support, out=np.full(len(labels), np.nan), where=support > 0)
    gt4 = np.vectorize(LEVEL4.get)(gt)
    pred4 = np.vectorize(LEVEL4.get)(pred)
    pos_gt, pos_pred = gt > 0, pred > 0
    return {
        'n_joints': int(len(gt)),
        'bacc5': float(balanced_accuracy_score(gt, pred)),
        'bacc4': float(balanced_accuracy_score(gt4, pred4)),
        'qwk': float(cohen_kappa_score(gt, pred, labels=labels, weights='quadratic')),
        'macro_f1': float(f1_score(gt, pred, labels=labels, average='macro', zero_division=0)),
        'accuracy': float(np.mean(gt == pred)),
        'mae': float(np.mean(np.abs(gt - pred))),
        'within1': float(np.mean(np.abs(gt - pred) <= 1)),
        'recall_ge2': float(np.mean(pred[gt >= 2] >= 2)) if np.any(gt >= 2) else float('nan'),
        'binary_sensitivity': float(np.mean(pos_pred[pos_gt])) if pos_gt.any() else float('nan'),
        'binary_specificity': float(np.mean(~pos_pred[~pos_gt])) if (~pos_gt).any() else float('nan'),
        'recall_per_grade': {str(s): (None if np.isnan(r) else float(r)) for s, r in zip(labels, recall)},
        'support_per_grade': {str(s): int(n) for s, n in zip(labels, support)},
        'confusion_matrix': cm.tolist(),
    }


def bootstrap_ci(df, keys=('bacc4', 'bacc5', 'qwk', 'macro_f1', 'mae'), n_boot=2000, seed=0, alpha=0.05):
    """Percentile CIs from resampling patients (joints within a patient are not independent).

    df needs columns 'patient', 'gt' and 'pred' (native scores).
    """
    rng = np.random.default_rng(seed)
    groups = {p: g for p, g in df.groupby('patient')}
    patients = np.array(sorted(groups))
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        sample = rng.choice(patients, size=len(patients), replace=True)
        gt = np.concatenate([groups[p]['gt'].to_numpy() for p in sample])
        pred = np.concatenate([groups[p]['pred'].to_numpy() for p in sample])
        m = compute_metrics(gt, pred)
        for k in keys:
            draws[k].append(m[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))] for k, v in draws.items()}
