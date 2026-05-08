"""Landmark-matching og metrics — refaktoreret fra 04_predict_and_evaluate_landmarks.py.

To rene funktioner:
- match_landmarks(ground_truth, predicted, tolerance) → MatchResult
- compute_metrics(result, mm_per_pixel) → Metrics
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MatchResult:
    """Resultat af greedy nearest-neighbor matching."""
    true_positives: int                       # antal predictions inden for tolerance
    false_positives: int                      # antal predictions uden for tolerance
    false_negatives: int                      # antal ground truth der ikke blev matched
    distances_px: list[float] = field(default_factory=list)  # afstande for de matchede par


@dataclass
class Metrics:
    """Sammenfatning af præstation til visning."""
    precision: float
    recall: float
    f1_score: float
    median_error_px: float
    median_error_mm: float
    mean_error_px: float
    mean_error_mm: float
    max_error_px: float
    max_error_mm: float


def match_landmarks(ground_truth_centers, predicted_centers, tolerance_px: float) -> MatchResult:
    """Greedy matching: hver prediction får sit nærmeste ground truth hvis inden for tolerance.

    Args:
        ground_truth_centers: liste af (x, y) ground truth-centre
        predicted_centers: liste af (x, y) predicted centre
        tolerance_px: maks afstand for at noget regnes som korrekt match

    Returns:
        MatchResult med true/false positives, false negatives og distancer for matchede par.
    """
    truth_array = np.array(ground_truth_centers) if len(ground_truth_centers) > 0 else np.empty((0, 2))
    predictions_array = np.array(predicted_centers) if len(predicted_centers) > 0 else np.empty((0, 2))

    matched_truth = np.zeros(len(truth_array), dtype=bool)
    true_positives = 0
    false_positives = 0
    matched_distances = []

    for prediction in predictions_array:
        if len(truth_array) == 0:
            false_positives += 1
            continue
        # Beregn afstande fra denne prediction til alle ground truth-centre
        distances_to_truth = np.linalg.norm(truth_array - prediction, axis=1)
        # Maskér allerede-matchede ground truth, så vi ikke matcher dem igen
        distances_with_mask = np.where(matched_truth, np.inf, distances_to_truth)
        nearest_index = int(np.argmin(distances_with_mask))
        nearest_distance = float(distances_with_mask[nearest_index])

        if nearest_distance <= tolerance_px:
            true_positives += 1
            matched_truth[nearest_index] = True
            matched_distances.append(nearest_distance)
        else:
            false_positives += 1

    false_negatives = int(np.sum(~matched_truth))
    return MatchResult(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        distances_px=matched_distances,
    )


def compute_metrics(result: MatchResult, mm_per_pixel: float) -> Metrics:
    """Beregn precision, recall, F1-score og distance-statistik fra et match-resultat."""
    # Precision og recall — pas på division ved nul
    precision_denominator = result.true_positives + result.false_positives
    precision = result.true_positives / precision_denominator if precision_denominator > 0 else 0.0

    recall_denominator = result.true_positives + result.false_negatives
    recall = result.true_positives / recall_denominator if recall_denominator > 0 else 0.0

    f1_denominator = precision + recall
    f1_score = 2 * precision * recall / f1_denominator if f1_denominator > 0 else 0.0

    # Distance-statistik — håndtér tomt list
    if len(result.distances_px) > 0:
        distances_array = np.array(result.distances_px)
        median_px = float(np.median(distances_array))
        mean_px = float(np.mean(distances_array))
        max_px = float(np.max(distances_array))
    else:
        median_px = mean_px = max_px = 0.0

    return Metrics(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        median_error_px=median_px,
        median_error_mm=median_px * mm_per_pixel,
        mean_error_px=mean_px,
        mean_error_mm=mean_px * mm_per_pixel,
        max_error_px=max_px,
        max_error_mm=max_px * mm_per_pixel,
    )
