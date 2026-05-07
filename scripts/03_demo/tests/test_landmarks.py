"""Tests for landmark matching og metrics."""
import sys
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO_DIR))

import landmarks  # noqa: E402


def test_perfekt_match_giver_fulde_metrics():
    """Identiske gt og pred → P=R=F1=1.0."""
    gt = [(10, 10), (50, 50), (100, 100)]
    pred = [(10, 10), (50, 50), (100, 100)]
    result = landmarks.match_landmarks(gt, pred, tolerance_px=5)
    assert result.tp == 3
    assert result.fp == 0
    assert result.fn == 0

    metrics = landmarks.compute_metrics(result, mm_per_pixel=0.05)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.median_error_mm == 0.0


def test_ingen_predictions_giver_recall_nul():
    """Ground truth findes, ingen predictions → R=0, P udefineret (vi sætter 0)."""
    gt = [(10, 10), (50, 50)]
    pred = []
    result = landmarks.match_landmarks(gt, pred, tolerance_px=5)
    assert result.tp == 0
    assert result.fp == 0
    assert result.fn == 2

    metrics = landmarks.compute_metrics(result, mm_per_pixel=0.05)
    assert metrics.precision == 0.0  # 0 predictions, definér som 0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_kun_falske_predictions():
    """Pred uden ground truth → alle FP."""
    gt = []
    pred = [(10, 10), (50, 50)]
    result = landmarks.match_landmarks(gt, pred, tolerance_px=5)
    assert result.tp == 0
    assert result.fp == 2
    assert result.fn == 0


def test_lille_afvigelse_inden_for_tolerance_matcher():
    """Pred 3 pixels fra gt med tolerance=5 → match."""
    gt = [(50, 50)]
    pred = [(53, 50)]
    result = landmarks.match_landmarks(gt, pred, tolerance_px=5)
    assert result.tp == 1
    assert result.fp == 0
    assert len(result.distances_px) == 1
    assert abs(result.distances_px[0] - 3.0) < 0.001


def test_stor_afvigelse_uden_for_tolerance_matcher_ikke():
    """Pred 10 pixels fra gt med tolerance=5 → ingen match."""
    gt = [(50, 50)]
    pred = [(60, 50)]
    result = landmarks.match_landmarks(gt, pred, tolerance_px=5)
    assert result.tp == 0
    assert result.fp == 1
    assert result.fn == 1


def test_metrics_omregner_til_mm():
    """Pixel-distancer skal omregnes til mm via mm_per_pixel."""
    gt = [(50, 50)]
    pred = [(54, 50)]  # 4 pixels væk
    result = landmarks.match_landmarks(gt, pred, tolerance_px=5)
    metrics = landmarks.compute_metrics(result, mm_per_pixel=0.05)
    # 4 pixels * 0.05 mm/pixel = 0.20 mm
    assert abs(metrics.median_error_mm - 0.20) < 0.001
