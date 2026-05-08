"""Tests for landmark matching og metrics."""
import sys
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO_DIR))

import landmarks  # noqa: E402


def test_perfekt_match_giver_fulde_metrics():
    """Identiske ground_truth og predictions → P=R=F1=1.0."""
    ground_truth = [(10, 10), (50, 50), (100, 100)]
    predictions = [(10, 10), (50, 50), (100, 100)]
    result = landmarks.match_landmarks(ground_truth, predictions, tolerance_px=5)
    assert result.true_positives == 3
    assert result.false_positives == 0
    assert result.false_negatives == 0

    metrics = landmarks.compute_metrics(result, mm_per_pixel=0.05)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.median_error_mm == 0.0


def test_ingen_predictions_giver_recall_nul():
    """Ground truth findes, ingen predictions → R=0, P udefineret (vi sætter 0)."""
    ground_truth = [(10, 10), (50, 50)]
    predictions = []
    result = landmarks.match_landmarks(ground_truth, predictions, tolerance_px=5)
    assert result.true_positives == 0
    assert result.false_positives == 0
    assert result.false_negatives == 2

    metrics = landmarks.compute_metrics(result, mm_per_pixel=0.05)
    assert metrics.precision == 0.0  # 0 predictions, definér som 0
    assert metrics.recall == 0.0
    assert metrics.f1_score == 0.0


def test_kun_falske_predictions():
    """Predictions uden ground truth → alle FP."""
    ground_truth = []
    predictions = [(10, 10), (50, 50)]
    result = landmarks.match_landmarks(ground_truth, predictions, tolerance_px=5)
    assert result.true_positives == 0
    assert result.false_positives == 2
    assert result.false_negatives == 0


def test_lille_afvigelse_inden_for_tolerance_matcher():
    """Prediction 3 pixels fra ground truth med tolerance=5 → match."""
    ground_truth = [(50, 50)]
    predictions = [(53, 50)]
    result = landmarks.match_landmarks(ground_truth, predictions, tolerance_px=5)
    assert result.true_positives == 1
    assert result.false_positives == 0
    assert len(result.distances_px) == 1
    assert abs(result.distances_px[0] - 3.0) < 0.001


def test_stor_afvigelse_uden_for_tolerance_matcher_ikke():
    """Prediction 10 pixels fra ground truth med tolerance=5 → ingen match."""
    ground_truth = [(50, 50)]
    predictions = [(60, 50)]
    result = landmarks.match_landmarks(ground_truth, predictions, tolerance_px=5)
    assert result.true_positives == 0
    assert result.false_positives == 1
    assert result.false_negatives == 1


def test_metrics_omregner_til_mm():
    """Pixel-distancer skal omregnes til mm via mm_per_pixel."""
    ground_truth = [(50, 50)]
    predictions = [(54, 50)]  # 4 pixels væk
    result = landmarks.match_landmarks(ground_truth, predictions, tolerance_px=5)
    metrics = landmarks.compute_metrics(result, mm_per_pixel=0.05)
    # 4 pixels * 0.05 mm/pixel = 0.20 mm
    assert abs(metrics.median_error_mm - 0.20) < 0.001
