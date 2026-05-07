"""Landmark-matching og metrics — refaktoreret fra 04_predict_and_evaluate_landmarks.py.

To rene funktioner:
- match_landmarks(gt, pred, tolerance) → MatchResult
- compute_metrics(result, mm_per_pixel) → Metrics
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MatchResult:
    """Resultat af greedy nearest-neighbor matching."""
    tp: int                          # antal predictions inden for tolerance
    fp: int                          # antal predictions uden for tolerance
    fn: int                          # antal gt der ikke blev matched
    distances_px: list[float] = field(default_factory=list)  # afstande for TP'er


@dataclass
class Metrics:
    """Sammenfatning af præstation til visning."""
    precision: float
    recall: float
    f1: float
    median_error_px: float
    median_error_mm: float
    mean_error_px: float
    mean_error_mm: float
    max_error_px: float
    max_error_mm: float


def match_landmarks(gt_centers, pred_centers, tolerance_px: float) -> MatchResult:
    """Greedy matching: hver pred får sit nærmeste gt hvis inden for tolerance.

    Args:
        gt_centers: liste af (x, y) ground truth-centre
        pred_centers: liste af (x, y) predicted centre
        tolerance_px: maks afstand for at noget regnes som korrekt match

    Returns:
        MatchResult med TP/FP/FN og distancer for matchede par.
    """
    gt = np.array(gt_centers) if len(gt_centers) > 0 else np.empty((0, 2))
    pred = np.array(pred_centers) if len(pred_centers) > 0 else np.empty((0, 2))

    matched_gt = np.zeros(len(gt), dtype=bool)
    tp, fp = 0, 0
    distances = []

    for p in pred:
        if len(gt) == 0:
            fp += 1
            continue
        # Beregn afstande fra denne pred til alle gt
        dists = np.linalg.norm(gt - p, axis=1)
        # Maskér allerede-matchede gt
        dists_masked = np.where(matched_gt, np.inf, dists)
        nearest_idx = int(np.argmin(dists_masked))
        nearest_dist = float(dists_masked[nearest_idx])

        if nearest_dist <= tolerance_px:
            tp += 1
            matched_gt[nearest_idx] = True
            distances.append(nearest_dist)
        else:
            fp += 1

    fn = int(np.sum(~matched_gt))
    return MatchResult(tp=tp, fp=fp, fn=fn, distances_px=distances)


def compute_metrics(result: MatchResult, mm_per_pixel: float) -> Metrics:
    """Beregn P/R/F1 og distance-statistik fra et match-resultat."""
    # Precision og recall — pas på division ved nul
    p_denom = result.tp + result.fp
    precision = result.tp / p_denom if p_denom > 0 else 0.0

    r_denom = result.tp + result.fn
    recall = result.tp / r_denom if r_denom > 0 else 0.0

    f1_denom = precision + recall
    f1 = 2 * precision * recall / f1_denom if f1_denom > 0 else 0.0

    # Distance-stats — håndtér tomt list
    if len(result.distances_px) > 0:
        d = np.array(result.distances_px)
        median_px = float(np.median(d))
        mean_px = float(np.mean(d))
        max_px = float(np.max(d))
    else:
        median_px = mean_px = max_px = 0.0

    return Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        median_error_px=median_px,
        median_error_mm=median_px * mm_per_pixel,
        mean_error_px=mean_px,
        mean_error_mm=mean_px * mm_per_pixel,
        max_error_px=max_px,
        max_error_mm=max_px * mm_per_pixel,
    )
