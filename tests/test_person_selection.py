"""Tests for automated person selection heuristics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pose_pipeline.utils.person_selection import (
    SelectionResult,
    _min_max_normalize,
    auto_annotate,
    evaluate_selector,
    extract_track_features,
    heuristic_select,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic track data
# ---------------------------------------------------------------------------


def _make_det(track_id: int, tlhw: list[float], confidence: float = 0.9) -> dict:
    return {"track_id": track_id, "tlhw": tlhw, "confidence": confidence}


def _make_tracks_single_person(n_frames: int = 100) -> list[list[dict]]:
    """Single person walking across frame center."""
    tracks = []
    for i in range(n_frames):
        x = 400.0 + i * 2.0
        y = 300.0
        tracks.append([_make_det(0, [x, y, 100.0, 200.0], 0.95)])
    return tracks


def _make_tracks_two_persons(n_frames: int = 100) -> list[list[dict]]:
    """Person 0 = subject (center, large, present throughout, moving).
    Person 1 = bystander (edge, small, present half the time, stationary)."""
    tracks = []
    for i in range(n_frames):
        subject = _make_det(0, [400.0 + i * 3.0, 300.0, 120.0, 240.0], 0.95)
        frame = [subject]
        if i < n_frames // 2:
            bystander = _make_det(1, [50.0, 50.0, 40.0, 80.0], 0.7)
            frame.append(bystander)
        tracks.append(frame)
    return tracks


def _make_tracks_three_persons(n_frames: int = 100) -> list[list[dict]]:
    """Person 0 = subject, Person 1 = clinician, Person 2 = brief passerby."""
    tracks = []
    for i in range(n_frames):
        frame = [_make_det(0, [400.0 + i * 2.0, 300.0, 110.0, 220.0], 0.93)]
        frame.append(_make_det(1, [100.0, 100.0, 60.0, 120.0], 0.8))
        if 20 <= i < 40:
            frame.append(_make_det(2, [700.0, 400.0, 50.0, 100.0], 0.6))
        tracks.append(frame)
    return tracks


def _make_tracks_equal_persons(n_frames: int = 100) -> list[list[dict]]:
    """Two persons with nearly identical statistics (hard case)."""
    tracks = []
    for i in range(n_frames):
        p0 = _make_det(0, [400.0 + i * 1.5, 300.0, 100.0, 200.0], 0.9)
        p1 = _make_det(1, [420.0 + i * 1.5, 310.0, 100.0, 200.0], 0.9)
        tracks.append([p0, p1])
    return tracks


# ---------------------------------------------------------------------------
# Tests: extract_track_features
# ---------------------------------------------------------------------------


class TestExtractTrackFeatures:
    def test_single_track(self) -> None:
        tracks = _make_tracks_single_person(50)
        tf = extract_track_features(tracks, 1920, 1080)

        assert len(tf.track_ids) == 1
        assert tf.track_ids[0] == 0
        assert tf.features["duration_frac"][0] == pytest.approx(1.0)
        assert tf.features["mean_confidence"][0] == pytest.approx(0.95)
        assert tf.features["mean_area"][0] > 0
        assert tf.features["total_displacement"][0] > 0

    def test_two_tracks_features(self) -> None:
        tracks = _make_tracks_two_persons(100)
        tf = extract_track_features(tracks, 1920, 1080)

        assert len(tf.track_ids) == 2
        assert set(tf.track_ids) == {0, 1}

        idx0 = np.where(tf.track_ids == 0)[0][0]
        idx1 = np.where(tf.track_ids == 1)[0][0]

        assert tf.features["duration_frac"][idx0] > tf.features["duration_frac"][idx1]
        assert tf.features["mean_area"][idx0] > tf.features["mean_area"][idx1]
        assert (
            tf.features["total_displacement"][idx0]
            > tf.features["total_displacement"][idx1]
        )
        assert (
            tf.features["mean_confidence"][idx0] > tf.features["mean_confidence"][idx1]
        )

    def test_empty_frames_handled(self) -> None:
        tracks = [[_make_det(0, [100.0, 100.0, 50.0, 100.0])], [], []]
        tf = extract_track_features(tracks, 640, 480)

        assert len(tf.track_ids) == 1
        assert tf.features["duration_frac"][0] == pytest.approx(1.0 / 3.0)

    def test_single_frame_track(self) -> None:
        tracks = [[_make_det(0, [100.0, 100.0, 50.0, 100.0])]]
        tf = extract_track_features(tracks, 640, 480)

        assert tf.features["total_displacement"][0] == 0.0
        assert tf.features["duration_frac"][0] == pytest.approx(1.0)

    def test_zero_frames(self) -> None:
        tf = extract_track_features([], 640, 480)
        assert len(tf.track_ids) == 0

    def test_missing_confidence_defaults_to_one(self) -> None:
        det = {"track_id": 5, "tlhw": [100.0, 100.0, 50.0, 100.0]}
        tracks = [[det]]
        tf = extract_track_features(tracks, 640, 480)

        assert tf.features["mean_confidence"][0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests: _min_max_normalize
# ---------------------------------------------------------------------------


class TestMinMaxNormalize:
    def test_basic(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        normed = _min_max_normalize(arr)
        np.testing.assert_allclose(normed, [0.0, 0.5, 1.0])

    def test_constant_array(self) -> None:
        arr = np.array([5.0, 5.0, 5.0])
        normed = _min_max_normalize(arr)
        np.testing.assert_allclose(normed, [0.0, 0.0, 0.0])

    def test_single_element(self) -> None:
        arr = np.array([3.0])
        normed = _min_max_normalize(arr)
        np.testing.assert_allclose(normed, [0.0])


# ---------------------------------------------------------------------------
# Tests: heuristic_select
# ---------------------------------------------------------------------------


class TestHeuristicSelect:
    def test_single_track_returns_it(self) -> None:
        tracks = _make_tracks_single_person(50)
        result = heuristic_select(tracks, 1920, 1080)

        assert result.predicted_track_id == 0
        assert result.confidence == 1.0
        assert 0 in result.scores

    def test_subject_wins_two_person(self) -> None:
        tracks = _make_tracks_two_persons(100)
        result = heuristic_select(tracks, 1920, 1080)

        assert result.predicted_track_id == 0
        assert result.confidence > 0.0
        assert result.scores[0] > result.scores[1]

    def test_subject_wins_three_person(self) -> None:
        tracks = _make_tracks_three_persons(100)
        result = heuristic_select(tracks, 1920, 1080)

        assert result.predicted_track_id == 0
        assert result.scores[0] > result.scores[1]
        assert result.scores[0] > result.scores[2]

    def test_low_confidence_for_similar_tracks(self) -> None:
        tracks = _make_tracks_equal_persons(100)
        result = heuristic_select(tracks, 1920, 1080)

        assert result.confidence < 0.3

    def test_empty_tracks_raises(self) -> None:
        with pytest.raises(ValueError, match="No tracks"):
            heuristic_select([], 640, 480)

    def test_custom_weights(self) -> None:
        tracks = _make_tracks_two_persons(100)
        weights = {
            "duration_frac": 1.0,
            "total_displacement": 0.0,
            "mean_centrality": 0.0,
            "mean_area": 0.0,
            "mean_confidence": 0.0,
        }
        result = heuristic_select(tracks, 1920, 1080, weights=weights)

        assert result.predicted_track_id == 0


# ---------------------------------------------------------------------------
# Tests: evaluate_selector
# ---------------------------------------------------------------------------


def _dummy_selector(tracks: list, width: int, height: int) -> SelectionResult:
    return heuristic_select(tracks, width, height)


class TestEvaluateSelector:
    def _make_ground_truth(self) -> list[dict]:
        return [
            {
                "key": {
                    "video_project": "test",
                    "filename": "v1.mp4",
                    "tracking_method": 8,
                },
                "keep_tracks": [0],
                "video_subject_id": 0,
                "num_tracks": 2,
                "tracks": _make_tracks_two_persons(100),
                "image_width": 1920,
                "image_height": 1080,
            },
            {
                "key": {
                    "video_project": "test",
                    "filename": "v2.mp4",
                    "tracking_method": 8,
                },
                "keep_tracks": [0],
                "video_subject_id": 0,
                "num_tracks": 3,
                "tracks": _make_tracks_three_persons(100),
                "image_width": 1920,
                "image_height": 1080,
            },
        ]

    def test_perfect_accuracy_on_easy_cases(self) -> None:
        gt = self._make_ground_truth()
        metrics = evaluate_selector(_dummy_selector, gt, thresholds=[0.0, 0.5])

        assert metrics.total == 2
        assert metrics.correct == 2
        assert metrics.top1_accuracy == pytest.approx(1.0)

    def test_accuracy_by_num_tracks(self) -> None:
        gt = self._make_ground_truth()
        metrics = evaluate_selector(_dummy_selector, gt)

        assert 2 in metrics.accuracy_by_num_tracks
        assert 3 in metrics.accuracy_by_num_tracks

    def test_coverage_at_zero_is_full(self) -> None:
        gt = self._make_ground_truth()
        metrics = evaluate_selector(_dummy_selector, gt, thresholds=[0.0])

        assert metrics.coverage_at_thresholds[0.0] == pytest.approx(1.0)

    def test_skips_invalid_entries(self) -> None:
        gt = [
            {
                "key": {
                    "video_project": "test",
                    "filename": "bad.mp4",
                    "tracking_method": 8,
                },
                "keep_tracks": [],
                "video_subject_id": -1,
                "num_tracks": 2,
                "tracks": _make_tracks_two_persons(50),
                "image_width": 640,
                "image_height": 480,
            }
        ]
        metrics = evaluate_selector(_dummy_selector, gt)

        assert metrics.total == 0

    def test_skips_empty_keep_tracks(self) -> None:
        gt = [
            {
                "key": {
                    "video_project": "test",
                    "filename": "absent.mp4",
                    "tracking_method": 8,
                },
                "keep_tracks": [],
                "video_subject_id": 0,
                "num_tracks": 2,
                "tracks": _make_tracks_two_persons(50),
                "image_width": 640,
                "image_height": 480,
            }
        ]
        metrics = evaluate_selector(_dummy_selector, gt)

        assert metrics.total == 0

    def test_empty_ground_truth(self) -> None:
        metrics = evaluate_selector(_dummy_selector, [])
        assert metrics.total == 0
        assert metrics.top1_accuracy == 0.0


# ---------------------------------------------------------------------------
# Tests: auto_annotate
# ---------------------------------------------------------------------------


def _setup_auto_annotate_mocks(
    rows: list[dict],
) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """Wire up DJ operator mocks so auto_annotate's query chain returns rows."""
    mock_pbv = MagicMock()
    mock_vi = MagicMock()
    mock_tb = MagicMock()
    mock_ml = MagicMock()

    # TrackingBbox & filt & method_filt & "num_tracks > 1" chains through __and__
    tb_chain = MagicMock()
    mock_tb.__and__ = MagicMock(return_value=tb_chain)
    tb_chain.__and__ = MagicMock(return_value=tb_chain)

    # result - PersonBboxValid
    unannotated = MagicMock()
    tb_chain.__sub__ = MagicMock(return_value=unannotated)

    # unannotated * VideoInfo -> .fetch(as_dict=True)
    joined = MagicMock()
    unannotated.__mul__ = MagicMock(return_value=joined)
    joined.fetch = MagicMock(return_value=rows)

    return mock_ml, mock_tb, mock_vi, mock_pbv


class TestAutoAnnotate:
    def test_dry_run_does_not_insert(self) -> None:
        tracks = _make_tracks_two_persons(100)
        row = {
            "video_project": "test",
            "filename": "v1.mp4",
            "tracking_method": 8,
            "tracks": tracks,
            "num_tracks": 2,
            "width": 1920,
            "height": 1080,
        }

        mock_ml, mock_tb, mock_vi, mock_pbv = _setup_auto_annotate_mocks([row])

        with (
            patch("pose_pipeline.PersonBboxValid", mock_pbv),
            patch("pose_pipeline.VideoInfo", mock_vi),
            patch("pose_pipeline.TrackingBbox", mock_tb),
            patch("pose_pipeline.TrackingBboxMethodLookup", mock_ml),
        ):
            auto_annotate("test_filt", dry_run=True)
            mock_pbv.insert1.assert_not_called()

    def test_inserts_high_confidence(self) -> None:
        tracks = _make_tracks_two_persons(100)
        row = {
            "video_project": "test",
            "filename": "v1.mp4",
            "tracking_method": 8,
            "tracks": tracks,
            "num_tracks": 2,
            "width": 1920,
            "height": 1080,
        }

        mock_ml, mock_tb, mock_vi, mock_pbv = _setup_auto_annotate_mocks([row])

        with (
            patch("pose_pipeline.PersonBboxValid", mock_pbv),
            patch("pose_pipeline.VideoInfo", mock_vi),
            patch("pose_pipeline.TrackingBbox", mock_tb),
            patch("pose_pipeline.TrackingBboxMethodLookup", mock_ml),
        ):
            summary = auto_annotate(
                "test_filt", confidence_threshold=0.0, dry_run=False
            )
            mock_pbv.insert1.assert_called_once()
            insert_arg = mock_pbv.insert1.call_args[0][0]
            assert insert_arg["video_subject_id"] == 0
            assert insert_arg["keep_tracks"] == [0]
            assert summary.auto_annotated == 1

    def test_defers_low_confidence(self) -> None:
        tracks = _make_tracks_equal_persons(100)
        row = {
            "video_project": "test",
            "filename": "v1.mp4",
            "tracking_method": 8,
            "tracks": tracks,
            "num_tracks": 2,
            "width": 1920,
            "height": 1080,
        }

        mock_ml, mock_tb, mock_vi, mock_pbv = _setup_auto_annotate_mocks([row])

        with (
            patch("pose_pipeline.PersonBboxValid", mock_pbv),
            patch("pose_pipeline.VideoInfo", mock_vi),
            patch("pose_pipeline.TrackingBbox", mock_tb),
            patch("pose_pipeline.TrackingBboxMethodLookup", mock_ml),
        ):
            summary = auto_annotate(
                "test_filt", confidence_threshold=0.9, dry_run=False
            )
            mock_pbv.insert1.assert_not_called()
            assert summary.deferred == 1
            assert summary.auto_annotated == 0
