"""Tests for bridging.py numpy math functions. No CUDA needed."""

import numpy as np
import pytest


# Import the pure-numpy functions directly to avoid pulling in DataJoint/CUDA deps.
# They live in pose_pipeline.wrappers.bridging but that module imports from pose_pipeline
# which requires DataJoint. So we import from the file path using importlib.
def _import_bridging_math():
    """Import math functions from bridging.py without triggering heavy deps (cv2, DataJoint)."""
    import importlib.util
    import sys
    import types
    from pathlib import Path
    from unittest.mock import MagicMock

    src = (
        Path(__file__).resolve().parents[1]
        / "pose_pipeline"
        / "wrappers"
        / "bridging.py"
    )

    stubs = {
        "pose_pipeline": types.ModuleType("pose_pipeline"),
        "cv2": MagicMock(),
    }
    stubs["pose_pipeline"].Video = None
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)

    try:
        spec = importlib.util.spec_from_file_location("bridging", src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, orig in saved.items():
            if orig is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = orig

    return mod


_bridging = _import_bridging_math()
scale_align = _bridging.scale_align
point_stdev = _bridging.point_stdev
augmentation_noise = _bridging.augmentation_noise
noise_to_conf = _bridging.noise_to_conf


# --- Fixtures ---


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def poses_3d(rng: np.random.Generator) -> np.ndarray:
    """Shape: (N_people=3, 10_augmentations, J=17_joints, 3_coords)."""
    return rng.standard_normal((3, 10, 17, 3)).astype(np.float64)


# --- scale_align ---


def test_scale_align_shape_preserved(poses_3d: np.ndarray) -> None:
    result = scale_align(poses_3d)
    assert result.shape == poses_3d.shape


def test_scale_align_normalizes_scale(poses_3d: np.ndarray) -> None:
    result = scale_align(poses_3d)
    # After alignment, each augmentation's mean-square-scale should equal the group mean
    square_scales = np.mean(np.square(result), axis=(-2, -1), keepdims=True)
    mean_sq = np.mean(square_scales, axis=-3, keepdims=True)
    np.testing.assert_allclose(
        square_scales, np.broadcast_to(mean_sq, square_scales.shape), rtol=1e-6
    )


# --- point_stdev ---


def test_point_stdev_shape(poses_3d: np.ndarray) -> None:
    result = point_stdev(poses_3d, item_axis=1, coord_axis=-1)
    assert result.shape == (3, 17)


def test_point_stdev_zero_variance() -> None:
    constant = np.ones((2, 5, 10, 3))
    result = point_stdev(constant, item_axis=1, coord_axis=-1)
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


# --- augmentation_noise ---


def test_augmentation_noise_shape_and_type(poses_3d: np.ndarray) -> None:
    result = augmentation_noise(poses_3d)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 17)


def test_augmentation_noise_single_person() -> None:
    rng = np.random.default_rng(99)
    poses = rng.standard_normal((1, 10, 25, 3))
    result = augmentation_noise(poses)
    assert result.shape == (1, 25)
    assert np.all(np.isfinite(result))


# --- noise_to_conf ---


def test_noise_to_conf_range() -> None:
    x = np.array([0.0, 100.0, 200.0, 300.0, 1000.0])
    result = noise_to_conf(x, half_val=200, sharpness=50)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_noise_to_conf_at_half_val() -> None:
    result = noise_to_conf(np.array([200.0]), half_val=200, sharpness=50)
    np.testing.assert_allclose(result, 0.5)


def test_noise_to_conf_monotonic_decreasing() -> None:
    x = np.arange(0, 1000, 10, dtype=np.float64)
    result = noise_to_conf(x)
    diffs = np.diff(result)
    assert np.all(diffs <= 0), "noise_to_conf should be monotonically decreasing"


# --- TF equivalence regression test ---


def test_numpy_matches_tf_implementation() -> None:
    tf = pytest.importorskip("tensorflow")

    rng = np.random.default_rng(123)
    poses = rng.standard_normal((2, 10, 17, 3)).astype(np.float32)

    # Original TF implementation
    def tf_scale_align(p: np.ndarray) -> np.ndarray:
        sq = tf.reduce_mean(tf.square(p), axis=(-2, -1), keepdims=True)
        msq = tf.reduce_mean(sq, axis=-3, keepdims=True)
        return (p * tf.sqrt(msq / sq)).numpy()

    def tf_point_stdev(p: np.ndarray) -> np.ndarray:
        cv = tf.math.reduce_variance(p, axis=1, keepdims=True)
        avg = tf.sqrt(tf.reduce_sum(cv, axis=-1, keepdims=True))
        return tf.squeeze(avg, (1, -1)).numpy()

    tf_result = tf_point_stdev(tf_scale_align(poses))
    np_result = augmentation_noise(poses)

    np.testing.assert_allclose(np_result, tf_result, rtol=1e-5, atol=1e-6)
