# This script confirms that the Sapiens (JAX/Equinox) packages are installed
# and the PosePipeline wrapper can load and run inference.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_sapiens_eqx_import():
    """Verify sapiens_eqx package is importable and exports expected API."""
    import sapiens_eqx

    assert hasattr(sapiens_eqx, "SapiensPose"), "SapiensPose not found in sapiens_eqx"
    assert hasattr(sapiens_eqx, "SapiensPoseEstimator"), "SapiensPoseEstimator not found"
    assert hasattr(sapiens_eqx, "GOLIATH_308_KEYPOINT_NAMES"), "Keypoint names not found"

    keypoint_names = sapiens_eqx.GOLIATH_308_KEYPOINT_NAMES
    assert len(keypoint_names) == 308, f"Expected 308 keypoints, got {len(keypoint_names)}"


def test_sapiens_wrapper_import():
    """Verify the PosePipeline sapiens wrapper is importable."""
    from pose_pipeline.wrappers.sapiens import SapiensEstimator, get_joint_names, NUM_KEYPOINTS

    assert NUM_KEYPOINTS == 308
    assert callable(SapiensEstimator)

    joint_names = get_joint_names()
    assert len(joint_names) == 308


def test_sapiens_pose_model_load():
    """Load the smallest Sapiens pose model and verify it initializes."""
    from sapiens_eqx import SapiensPose

    # Try pretrained (requires HF_TOKEN for private repo), fall back to PyTorch conversion
    try:
        model = SapiensPose.from_pretrained(variant="0.3b")
    except Exception:
        model = SapiensPose.from_pytorch(variant="0.3b")
    assert model is not None, "SapiensPose model failed to load"


def test_sapiens_pose_inference():
    """Run pose inference on a dummy image through the SapiensEstimator wrapper."""
    from pose_pipeline.wrappers.sapiens import SapiensEstimator

    estimator = SapiensEstimator(variant="0.3b", tasks=["pose"], img_size=(1024, 768))
    assert "pose" in estimator.estimators, "Pose estimator not initialized"

    # Create a dummy input image (H, W, 3) and preprocess it
    dummy_img = np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8)
    input_tensor = estimator.estimators["pose"].preprocess(dummy_img)

    # Run JIT-compiled inference
    keypoints, scores = estimator._jit_pose(input_tensor)
    keypoints = np.array(keypoints)
    scores = np.array(scores)

    assert keypoints.shape[-1] == 2, f"Expected (x, y) coords, got shape {keypoints.shape}"
    assert scores.shape[-1] == 308, f"Expected 308 keypoint scores, got {scores.shape}"
