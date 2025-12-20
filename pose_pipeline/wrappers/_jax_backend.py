"""
JAX/Equinox backend for SAM-3D-Body.

Uses the sam3d_body_eqx package - a pure JAX port of Meta's SAM-3D-Body.
"""

from typing import Optional

import numpy as np

from .sam3d_backend import SAM3DBackend, SAM3DResult

# Global model cache to avoid reloading
_jax_model_cache = {}


def _to_numpy_jax(x):
    """Convert JAX array to numpy array."""
    if x is None:
        return None
    return np.asarray(x)


class JAXSAM3DBackend(SAM3DBackend):
    """JAX/Equinox backend for SAM-3D-Body using sam3d_body_eqx package."""

    def __init__(self):
        self._estimator = None
        self._warmup_done = False
        self._repo_id = None

    def load(self, repo_id: str, device: str = "cuda") -> None:
        """Load SAM-3D-Body model.

        Note: JAX handles device placement automatically based on available
        accelerators. The device parameter is accepted for API compatibility
        but not directly used.

        Args:
            repo_id: HuggingFace repository ID (ignored - uses native weights)
            device: Device hint ('cuda' or 'cpu') - for info only
        """
        # Use a fixed cache key since JAX handles devices automatically
        cache_key = "default"
        if cache_key in _jax_model_cache:
            self._estimator = _jax_model_cache[cache_key]
            self._repo_id = repo_id
            return

        # Import here to avoid import errors when JAX backend not available
        from sam3d_body_eqx import SAM3DBodyEstimator

        print("Loading SAM-3D-Body (JAX/Equinox) from HuggingFace...")

        # Load pretrained model (uses native .eqx format, no PyTorch required)
        self._estimator = SAM3DBodyEstimator.from_pretrained()

        self._repo_id = repo_id
        self._warmup_done = False
        _jax_model_cache[cache_key] = self._estimator

        # Check JAX device
        import jax

        devices = jax.devices()
        device_info = devices[0].device_kind if devices else "unknown"
        print(f"SAM-3D-Body (JAX) loaded on {device_info}")

    def _warmup(self, image_rgb: np.ndarray, bbox_xyxy: np.ndarray) -> None:
        """Run warmup inference for JIT compilation.

        JAX compiles functions on first call. This warmup ensures
        compilation overhead doesn't affect benchmarks.
        """
        if self._warmup_done:
            return

        print("Running JAX warmup (JIT compilation)...")
        # Run one inference to trigger JIT compilation
        _ = self._estimator.predict(image_rgb, bbox=bbox_xyxy, use_iterative=True)

        # Block until computation completes
        import jax

        jax.block_until_ready(_)

        self._warmup_done = True
        print("JAX warmup complete.")

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

        # Trigger JIT warmup on first call
        self._warmup(image_rgb, bbox_xyxy)

        try:
            result = self._estimator.predict(
                image_rgb,
                bbox=bbox_xyxy,
                use_iterative=True,
            )

            if result is None:
                return None

            # Map JAX output keys to unified format
            # Note: JAX uses 'body_pose' instead of 'body_pose_params', etc.
            return SAM3DResult(
                vertices=_to_numpy_jax(result.get("pred_vertices")),
                keypoints_3d=_to_numpy_jax(result.get("pred_keypoints_3d")),
                keypoints_2d=_to_numpy_jax(result.get("pred_keypoints_2d")),
                camera_t=_to_numpy_jax(result.get("pred_cam_t")),
                focal_length=float(result.get("focal_length", 0.0)),
                body_pose_params=_to_numpy_jax(result.get("body_pose")),
                hand_pose_params=_to_numpy_jax(result.get("hand")),
                shape_params=_to_numpy_jax(result.get("shape")),
                global_rot=_to_numpy_jax(result.get("global_rot")),
                mesh_faces=self.mesh_faces,
            )

        except (RuntimeError, ValueError) as e:
            print(f"JAX inference error: {e}")
            return None

    @property
    def mesh_faces(self) -> np.ndarray:
        """Get mesh face indices."""
        if self._estimator is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        # Access faces from the MHR model
        return np.asarray(self._estimator.model.head_pose.mhr.faces)

    @property
    def backend_name(self) -> str:
        """Return backend identifier."""
        return "jax"

    def get_block_fn(self):
        """Return a function to block until JAX computation completes.

        Useful for accurate benchmarking.
        """
        import jax

        def block_fn(result):
            if result is not None:
                # Block on the vertices array to ensure computation is complete
                jax.block_until_ready(result.vertices)

        return block_fn
