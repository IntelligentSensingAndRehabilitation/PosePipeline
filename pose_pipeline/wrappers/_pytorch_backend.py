"""
PyTorch backend for SAM-3D-Body.

Uses the original sam_3d_body package from Meta/Facebook Research.
"""

from typing import Optional

import numpy as np
import torch

from .sam3d_backend import SAM3DBackend, SAM3DResult

# Global model cache to avoid reloading
_pytorch_model_cache = {}


def _to_numpy(x):
    """Convert torch tensor to numpy array, handling CPU/CUDA devices."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


class PyTorchSAM3DBackend(SAM3DBackend):
    """PyTorch backend for SAM-3D-Body using sam_3d_body package."""

    def __init__(self):
        self._estimator = None
        self._device = None
        self._repo_id = None

    def load(self, repo_id: str, device: str = "cuda") -> None:
        """Load SAM-3D-Body model from HuggingFace.

        Args:
            repo_id: HuggingFace repository ID (e.g., "facebook/sam-3d-body-dinov3")
            device: Device for inference ('cuda' or 'cpu')
        """
        cache_key = (repo_id, device)
        if cache_key in _pytorch_model_cache:
            self._estimator = _pytorch_model_cache[cache_key]
            self._device = device
            self._repo_id = repo_id
            return

        # Import here to avoid import errors when PyTorch backend not available
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
        from sam_3d_body.build_models import _hf_download

        print(f"Loading SAM-3D-Body (PyTorch) from {repo_id}...")

        # Download checkpoint from HuggingFace
        ckpt_path, mhr_path = _hf_download(repo_id)

        # Load model with explicit device
        model, model_cfg = load_sam_3d_body(
            checkpoint_path=ckpt_path,
            mhr_path=mhr_path,
            device=device,
        )

        # Create estimator without detector (we use pre-computed bboxes)
        self._estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )

        self._device = device
        self._repo_id = repo_id
        _pytorch_model_cache[cache_key] = self._estimator

        print(f"SAM-3D-Body (PyTorch) loaded on {device}")

    def predict(
        self, image_rgb: np.ndarray, bbox_xyxy: np.ndarray
    ) -> Optional[SAM3DResult]:
        """Run inference on a single image with bounding box.

        Args:
            image_rgb: (H, W, 3) RGB image as numpy array
            bbox_xyxy: (4,) bounding box as [x1, y1, x2, y2]

        Returns:
            SAM3DResult or None if inference failed
        """
        if self._estimator is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        bbox_array = bbox_xyxy.reshape(1, 4)

        try:
            outputs = self._estimator.process_one_image(
                image_rgb,
                bboxes=bbox_array,
                bbox_thr=0.5,
                use_mask=False,
                inference_type="full",
            )

            if len(outputs) == 0:
                return None

            person = outputs[0]

            return SAM3DResult(
                vertices=_to_numpy(person["pred_vertices"]),
                keypoints_3d=_to_numpy(person["pred_keypoints_3d"]),
                keypoints_2d=_to_numpy(person["pred_keypoints_2d"]),
                camera_t=_to_numpy(person["pred_cam_t"]),
                focal_length=float(_to_numpy(person["focal_length"])),
                body_pose_params=_to_numpy(person["body_pose_params"]),
                hand_pose_params=_to_numpy(person["hand_pose_params"]),
                shape_params=_to_numpy(person["shape_params"]),
                global_rot=_to_numpy(person["global_rot"]),
                mesh_faces=self.mesh_faces,
            )

        except (RuntimeError, ValueError) as e:
            print(f"PyTorch inference error: {e}")
            return None

    @property
    def mesh_faces(self) -> np.ndarray:
        """Get mesh face indices."""
        if self._estimator is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._estimator.faces

    @property
    def backend_name(self) -> str:
        """Return backend identifier."""
        return "pytorch"
