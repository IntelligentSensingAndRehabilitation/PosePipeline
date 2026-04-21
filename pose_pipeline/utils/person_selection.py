"""Automated person-identity selection for multi-person tracking videos.

Scores tracked persons using bbox statistics to identify the study subject,
replacing manual annotation via PersonBboxValid for high-confidence cases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: dict[str, float] = {
    "duration_frac": 0.25,
    "total_displacement": 0.15,
    "mean_centrality": 0.20,
    "mean_area": 0.25,
    "mean_confidence": 0.15,
}

FEATURE_NAMES: list[str] = list(DEFAULT_WEIGHTS.keys())


@dataclass
class TrackFeatures:
    """Per-track feature vectors extracted from bbox sequences."""

    track_ids: NDArray[np.intp]
    features: dict[str, NDArray[np.floating]]


@dataclass
class SelectionResult:
    """Result of heuristic track selection."""

    predicted_track_id: int
    confidence: float
    scores: dict[int, float]


@dataclass
class EvaluationMetrics:
    """Metrics from running a selector against ground truth."""

    top1_accuracy: float
    total: int
    correct: int
    accuracy_by_num_tracks: dict[int, float]
    precision_at_thresholds: dict[float, float]
    coverage_at_thresholds: dict[float, float]


@dataclass
class AutoAnnotationSummary:
    """Summary of an auto-annotation run."""

    auto_annotated: int
    deferred: int
    details: list[dict[str, Any]] = field(default_factory=list)


def extract_track_features(
    tracks: list[list[dict]],
    image_width: int,
    image_height: int,
) -> TrackFeatures:
    """Extract scoring features from per-frame tracking bbox data.

    Args:
        tracks: Per-frame list of detections. Each detection has keys:
            track_id (int), tlhw (4,), confidence (float).
        image_width: Video frame width in pixels.
        image_height: Video frame height in pixels.

    Returns:
        TrackFeatures with arrays indexed by track_id.
    """
    num_frames = len(tracks)
    image_area = image_width * image_height
    image_cx = image_width / 2.0
    image_cy = image_height / 2.0
    image_diag = np.sqrt(image_width**2 + image_height**2)

    per_track: dict[int, list[dict]] = {}
    for frame_idx, frame_dets in enumerate(tracks):
        for det in frame_dets:
            tid = det["track_id"]
            per_track.setdefault(tid, []).append({"frame": frame_idx, **det})

    track_ids = np.array(sorted(per_track.keys()), dtype=np.intp)
    n_tracks = len(track_ids)

    duration_frac = np.zeros(n_tracks)
    total_displacement = np.zeros(n_tracks)
    mean_centrality = np.zeros(n_tracks)
    mean_area = np.zeros(n_tracks)
    mean_confidence = np.zeros(n_tracks)

    for i, tid in enumerate(track_ids):
        dets = per_track[tid]
        n_det = len(dets)

        duration_frac[i] = n_det / num_frames if num_frames > 0 else 0.0

        bboxes = np.array([d["tlhw"] for d in dets])
        centers_x = bboxes[:, 0] + bboxes[:, 2] / 2.0
        centers_y = bboxes[:, 1] + bboxes[:, 3] / 2.0

        if n_det > 1:
            dx = np.diff(centers_x)
            dy = np.diff(centers_y)
            total_displacement[i] = np.sum(np.sqrt(dx**2 + dy**2)) / image_diag
        else:
            total_displacement[i] = 0.0

        dist_from_center = np.sqrt(
            (centers_x - image_cx) ** 2 + (centers_y - image_cy) ** 2
        )
        mean_centrality[i] = 1.0 - np.mean(dist_from_center) / image_diag

        areas = bboxes[:, 2] * bboxes[:, 3]
        mean_area[i] = np.mean(areas) / image_area

        confidences = np.array([d.get("confidence", 1.0) for d in dets])
        mean_confidence[i] = np.mean(confidences)

    return TrackFeatures(
        track_ids=track_ids,
        features={
            "duration_frac": duration_frac,
            "total_displacement": total_displacement,
            "mean_centrality": mean_centrality,
            "mean_area": mean_area,
            "mean_confidence": mean_confidence,
        },
    )


def _min_max_normalize(arr: NDArray[np.floating]) -> NDArray[np.floating]:
    """Min-max normalize array to [0, 1]. Returns zeros if range is zero."""
    arr_min = arr.min()
    arr_max = arr.max()
    rng = arr_max - arr_min
    if rng < 1e-12:
        return np.zeros_like(arr)
    return (arr - arr_min) / rng


def heuristic_select(
    tracks: list[list[dict]],
    image_width: int,
    image_height: int,
    weights: dict[str, float] | None = None,
) -> SelectionResult:
    """Score tracks and select the most likely study subject.

    Args:
        tracks: Per-frame detection lists from TrackingBbox.
        image_width: Frame width.
        image_height: Frame height.
        weights: Feature weights. Uses DEFAULT_WEIGHTS if None.

    Returns:
        SelectionResult with predicted track_id, confidence, and per-track scores.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    tf = extract_track_features(tracks, image_width, image_height)

    if len(tf.track_ids) == 0:
        raise ValueError("No tracks found in video")

    if len(tf.track_ids) == 1:
        tid = int(tf.track_ids[0])
        return SelectionResult(
            predicted_track_id=tid, confidence=1.0, scores={tid: 1.0}
        )

    normalized = {name: _min_max_normalize(arr) for name, arr in tf.features.items()}

    composite = np.zeros(len(tf.track_ids))
    for name, w in weights.items():
        composite += w * normalized[name]

    scores = {int(tid): float(s) for tid, s in zip(tf.track_ids, composite)}

    sorted_idx = np.argsort(composite)[::-1]
    best_tid = int(tf.track_ids[sorted_idx[0]])

    margin = composite[sorted_idx[0]] - composite[sorted_idx[1]]
    max_possible = sum(weights.values())
    confidence = float(margin / max_possible) if max_possible > 0 else 0.0

    return SelectionResult(
        predicted_track_id=best_tid, confidence=confidence, scores=scores
    )


def load_ground_truth(
    filt: Any = None,
    tracking_method_name: str = "MMDet_deepsort",
) -> list[dict]:
    """Load human-annotated PersonBboxValid entries as ground truth.

    Args:
        filt: DataJoint restriction to apply (e.g., project filter).
        tracking_method_name: Which tracking method to load annotations for.

    Returns:
        List of dicts with keys: key, keep_tracks, video_subject_id,
        num_tracks, tracks, image_width, image_height.
    """
    from pose_pipeline import (
        PersonBboxValid,
        TrackingBbox,
        TrackingBboxMethodLookup,
        VideoInfo,
    )

    method_filt = (
        TrackingBboxMethodLookup & f'tracking_method_name="{tracking_method_name}"'
    )
    query = PersonBboxValid * TrackingBbox * VideoInfo & method_filt & "num_tracks > 1"
    if filt is not None:
        query = query & filt

    results = []
    for row in query.fetch(as_dict=True):
        results.append(
            {
                "key": {
                    "video_project": row["video_project"],
                    "filename": row["filename"],
                    "tracking_method": row["tracking_method"],
                },
                "keep_tracks": row["keep_tracks"],
                "video_subject_id": row["video_subject_id"],
                "num_tracks": row["num_tracks"],
                "tracks": row["tracks"],
                "image_width": row["width"],
                "image_height": row["height"],
            }
        )

    return results


def evaluate_selector(
    selector_fn: Callable[[list[list[dict]], int, int], SelectionResult],
    ground_truth: list[dict],
    thresholds: list[float] | None = None,
) -> EvaluationMetrics:
    """Evaluate a selection function against human annotations.

    Args:
        selector_fn: Function with signature (tracks, width, height) -> SelectionResult.
        ground_truth: Output of load_ground_truth().
        thresholds: Confidence thresholds for precision/coverage curves.

    Returns:
        EvaluationMetrics with accuracy and threshold-based metrics.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    correct = 0
    total = 0
    by_num_tracks: dict[int, list[bool]] = {}
    confidences: list[float] = []
    correctness: list[bool] = []

    for entry in ground_truth:
        if entry["video_subject_id"] < 0:
            continue

        keep = entry["keep_tracks"]
        if len(keep) == 0:
            continue

        result = selector_fn(
            entry["tracks"], entry["image_width"], entry["image_height"]
        )

        is_correct = result.predicted_track_id in keep
        total += 1
        correct += int(is_correct)

        nt = min(entry["num_tracks"], 4)
        by_num_tracks.setdefault(nt, []).append(is_correct)

        confidences.append(result.confidence)
        correctness.append(is_correct)

    top1_accuracy = correct / total if total > 0 else 0.0

    accuracy_by_nt = {k: np.mean(v).item() for k, v in sorted(by_num_tracks.items())}

    conf_arr = np.array(confidences)
    corr_arr = np.array(correctness)

    precision_at = {}
    coverage_at = {}
    for t in thresholds:
        mask = conf_arr >= t
        n_above = mask.sum()
        coverage_at[t] = n_above / total if total > 0 else 0.0
        precision_at[t] = corr_arr[mask].mean().item() if n_above > 0 else 0.0

    return EvaluationMetrics(
        top1_accuracy=top1_accuracy,
        total=total,
        correct=correct,
        accuracy_by_num_tracks=accuracy_by_nt,
        precision_at_thresholds=precision_at,
        coverage_at_thresholds=coverage_at,
    )


def auto_annotate(
    filt: Any,
    confidence_threshold: float = 0.8,
    tracking_method_name: str = "MMDet_deepsort",
    weights: dict[str, float] | None = None,
    dry_run: bool = False,
) -> AutoAnnotationSummary:
    """Auto-annotate unannotated multi-person videos with high-confidence selections.

    Args:
        filt: DataJoint restriction (e.g., project/session filter).
        confidence_threshold: Minimum confidence to auto-insert.
        tracking_method_name: Tracking method to operate on.
        weights: Feature weights for heuristic scorer.
        dry_run: If True, compute scores but don't insert into database.

    Returns:
        AutoAnnotationSummary with counts and per-video details.
    """
    from pose_pipeline import (
        PersonBboxValid,
        TrackingBbox,
        TrackingBboxMethodLookup,
        VideoInfo,
    )

    method_filt = (
        TrackingBboxMethodLookup & f'tracking_method_name="{tracking_method_name}"'
    )
    unannotated = (
        TrackingBbox & filt & method_filt & "num_tracks > 1"
    ) - PersonBboxValid

    keys_data = (unannotated * VideoInfo).fetch(as_dict=True)

    auto_annotated = 0
    deferred = 0
    details = []

    for row in keys_data:
        result = heuristic_select(
            row["tracks"], row["width"], row["height"], weights=weights
        )

        entry = {
            "video_project": row["video_project"],
            "filename": row["filename"],
            "tracking_method": row["tracking_method"],
            "predicted_track_id": result.predicted_track_id,
            "confidence": result.confidence,
            "num_tracks": row["num_tracks"],
        }

        if result.confidence >= confidence_threshold:
            entry["action"] = "annotated"
            auto_annotated += 1

            if not dry_run:
                insert_key = {
                    "video_project": row["video_project"],
                    "filename": row["filename"],
                    "tracking_method": row["tracking_method"],
                    "video_subject_id": 0,
                    "keep_tracks": [result.predicted_track_id],
                }
                PersonBboxValid.insert1(insert_key)
                logger.info(
                    "Auto-annotated %s track=%d conf=%.3f",
                    row["filename"],
                    result.predicted_track_id,
                    result.confidence,
                )
        else:
            entry["action"] = "deferred"
            deferred += 1
            logger.info(
                "Deferred %s conf=%.3f < %.3f",
                row["filename"],
                result.confidence,
                confidence_threshold,
            )

        details.append(entry)

    logger.info(
        "Auto-annotation complete: %d annotated, %d deferred", auto_annotated, deferred
    )
    return AutoAnnotationSummary(
        auto_annotated=auto_annotated, deferred=deferred, details=details
    )
