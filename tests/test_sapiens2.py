# This script confirms that the Sapiens2 (JAX/Equinox) packages are installed
# and the PosePipeline wrapper can load and run inference.
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.gpu

# Smallest Sapiens2 variant that ships a pose head (0.1b is backbone-only)
VARIANT = "0.4b"

# Test video built from pose_demo.jpg (a skier): (x, y, w, h) PosePipeline-format bbox around
# the skier and per-frame presence
DEMO_IMAGE = os.path.join(os.path.dirname(__file__), "pose_demo.jpg")
NUM_FRAMES = 5
BBOX = (270.0, 40.0, 235.0, 355.0)
PRESENT = np.array([True, True, False, True, True])


@pytest.fixture(scope="module")
def pose_estimator():
    """Load the pose estimator once and share it across tests in this module."""
    from pose_pipeline.wrappers.sapiens2 import Sapiens2Estimator

    return Sapiens2Estimator(variant=VARIANT, tasks=["pose"], img_size=(1024, 768))


@pytest.fixture(scope="module")
def demo_video(tmp_path_factory):
    """Write a short video repeating pose_demo.jpg and return (path, BGR frames as decoded)."""
    image = cv2.imread(DEMO_IMAGE)
    path = str(tmp_path_factory.mktemp("sapiens2") / "test.mp4")

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (image.shape[1], image.shape[0]))
    for _ in range(NUM_FRAMES):
        writer.write(image)
    writer.release()

    # Re-read so reference comparisons use the same (lossy) decoded frames as the wrapper
    cap = cv2.VideoCapture(path)
    frames = [cap.read()[1] for _ in range(NUM_FRAMES)]
    cap.release()
    return path, frames


def test_sapiens2_eqx_import():
    """Verify sapiens2_eqx package is importable and exports expected API."""
    import sapiens2_eqx
    from sapiens2_eqx import model, inference

    assert VARIANT in sapiens2_eqx.VARIANTS, f"Variant {VARIANT} not found in sapiens2_eqx"
    assert hasattr(model, "Sapiens2Pose"), "Sapiens2Pose not found in sapiens2_eqx.model"
    assert hasattr(inference, "PoseEstimator"), "PoseEstimator not found in sapiens2_eqx.inference"
    assert hasattr(inference, "udp_decode"), "udp_decode not found in sapiens2_eqx.inference"


def test_sapiens2_wrapper_import():
    """Verify the PosePipeline sapiens2 wrapper is importable."""
    from pose_pipeline.wrappers.sapiens2 import Sapiens2Estimator, get_joint_names, NUM_KEYPOINTS

    assert NUM_KEYPOINTS == 308
    assert callable(Sapiens2Estimator)

    joint_names = get_joint_names()
    assert len(joint_names) == 308

    # Sapiens2 shares the Goliath 308 layout with Sapiens v1
    from pose_pipeline.wrappers.sapiens2 import get_joint_names as sapiens_joint_names

    assert joint_names == sapiens_joint_names()
    assert get_joint_names(normalize=False) == sapiens_joint_names(normalize=False)


def test_sapiens2_unknown_task():
    """Unknown task names should fail before any model is loaded."""
    from pose_pipeline.wrappers.sapiens2 import Sapiens2Estimator

    with pytest.raises(ValueError, match="Unknown Sapiens2 task"):
        Sapiens2Estimator(variant=VARIANT, tasks=["depth"])


def test_sapiens2_pose_model_load():
    """Load the smallest Sapiens2 pose model and verify it initializes."""
    from sapiens2_eqx.model import Sapiens2Pose

    # Try pretrained (requires HF_TOKEN / hf auth login for private repo), fall back to PyTorch conversion
    try:
        model = Sapiens2Pose.from_pretrained(variant=VARIANT)
    except (OSError, EnvironmentError):
        model = Sapiens2Pose.from_pytorch(variant=VARIANT)
    assert model is not None, "Sapiens2Pose model failed to load"


def test_sapiens2_pose_inference(pose_estimator):
    """Run pose inference on a dummy image through the Sapiens2Estimator wrapper."""
    import jax.numpy as jnp
    from sapiens2_eqx.inference import preprocess_image
    from pose_pipeline.wrappers.sapiens2 import _batched_pose_step

    assert "pose" in pose_estimator.models, "Pose model not initialized"

    # Create a dummy input image (H, W, 3) and preprocess it into a batch of 1
    dummy_img = np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8)
    input_tensor = jnp.stack([preprocess_image(dummy_img, img_size=pose_estimator.img_size)])

    # Run JIT-compiled inference
    keypoints, scores, heatmap_size = _batched_pose_step(pose_estimator.models["pose"], input_tensor)
    keypoints = np.array(keypoints)
    scores = np.array(scores)

    assert keypoints.shape == (1, 308, 2), f"Expected (1, 308, 2) keypoints, got shape {keypoints.shape}"
    assert scores.shape == (1, 308), f"Expected (1, 308) keypoint scores, got {scores.shape}"
    assert tuple(heatmap_size) == (256, 192), f"Expected 1/4-resolution heatmaps, got {heatmap_size}"
    assert np.isfinite(keypoints).all() and np.isfinite(scores).all()


def test_sapiens2_predict_video(pose_estimator, demo_video):
    """predict_video returns (N, 308, 3) keypoints, all NaN for frames without a tracked person."""
    path, _ = demo_video
    bboxes = np.array([BBOX] * NUM_FRAMES)

    # Frame 2 is marked not present; frame 4 is present but its bbox is NaN (person not tracked)
    bboxes[4] = np.nan
    tracked = PRESENT & ~np.isnan(bboxes).any(axis=1)

    # batch_size=4 over 5 frames exercises both a full and a padded batch
    keypoints = pose_estimator.predict_video(path, bboxes, PRESENT, batch_size=4)["keypoints"]

    assert keypoints.shape == (NUM_FRAMES, 308, 3)
    assert np.isnan(keypoints[~tracked]).all(), "Frames without a tracked person should be all NaN"
    assert np.isfinite(keypoints[tracked]).all(), "Frames with a person should have no NaN"
    assert (keypoints[tracked, :, 2] > 0).any(axis=1).all(), "Frames with a person should have nonzero scores"

    # Keypoints are mapped back to the full frame, so the confident ones should land on the skier
    x, y, w, h = BBOX
    for frame_kpts in keypoints[tracked]:
        confident = frame_kpts[frame_kpts[:, 2] > 0.5, :2]
        assert len(confident) > 0, "Expected confident keypoints on the skier"
        inside = (confident[:, 0] >= x) & (confident[:, 0] <= x + w) & (confident[:, 1] >= y) & (confident[:, 1] <= y + h)
        assert inside.mean() > 0.9, f"Only {inside.mean():.1%} of confident keypoints inside the bbox"

    # Nose (Goliath index 0) should be on the skier's face, near (370, 80)
    nose = keypoints[0, 0, :2]
    assert np.linalg.norm(nose - np.array([370.0, 80.0])) < 30, f"Nose at {nose}"


def test_sapiens2_predict_video_too_few_frames(pose_estimator, demo_video):
    """More bboxes than video frames should raise rather than silently misalign keypoints."""
    path, _ = demo_video
    # All-NaN bboxes skip inference, so this only exercises frame reading
    bboxes = np.full((NUM_FRAMES + 2, 4), np.nan)
    present = np.ones(NUM_FRAMES + 2, dtype=bool)

    with pytest.raises(RuntimeError, match="Could not read frame"):
        pose_estimator.predict_video(path, bboxes, present, batch_size=4)


def test_sapiens2_matches_reference_estimator(pose_estimator, demo_video):
    """Wrapper keypoints match sapiens2_eqx's own predict_multi_person on the same frame and bbox."""
    from sapiens2_eqx.inference import PoseEstimator

    path, frames = demo_video
    bboxes = np.array([BBOX] * NUM_FRAMES)
    keypoints = pose_estimator.predict_video(path, bboxes, PRESENT, batch_size=4)["keypoints"]

    frame_idx = 3
    x, y, w, h = BBOX
    reference = PoseEstimator(pose_estimator.models["pose"]).predict_multi_person(
        cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB), np.array([[x, y, x + w, y + h]])
    )[0]

    # Batched (vmap) vs single-image compiled forwards can differ slightly, which may flip the
    # heatmap argmax for a few low-confidence keypoints, so compare robust statistics.
    dist = np.linalg.norm(reference["keypoints"] - keypoints[frame_idx, :, :2], axis=-1)
    assert np.median(dist) < 0.5, f"Median keypoint distance to reference {np.median(dist):.3f} px"
    assert np.mean(dist < 2.0) > 0.95, f"Only {np.mean(dist < 2.0):.1%} of keypoints within 2 px of reference"
    np.testing.assert_allclose(keypoints[frame_idx, :, 2], reference["scores"], atol=0.05)


def test_estimator_cache():
    """Verify _get_estimator returns the same instance on repeated calls."""
    from pose_pipeline.wrappers.sapiens2 import _get_estimator, _estimator_cache

    _estimator_cache.clear()

    est1 = _get_estimator("0.4b", ["pose"])
    est2 = _get_estimator("0.4b", ["pose"])
    assert est1 is est2, "Cache should return the same estimator instance"

    # Switching variant should evict the old entry
    est3 = _get_estimator("0.8b", ["pose"])
    assert est3 is not est1, "Different variant should create a new estimator"
    assert len(_estimator_cache) == 1, "Cache should only hold one entry at a time"

    _estimator_cache.clear()
