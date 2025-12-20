"""
SAM-3D-Body backend abstraction layer.

Provides unified interface for JAX/Equinox and PyTorch backends with
auto-detection, manual override, and benchmarking support.
"""

import importlib.util
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, Optional

import numpy as np

# Type alias for backend selection
BackendType = Literal["auto", "jax", "pytorch"]


def is_jax_backend_available() -> bool:
    """Check if sam3d_body_eqx (JAX/Equinox) is importable."""
    return importlib.util.find_spec("sam3d_body_eqx") is not None


def is_pytorch_backend_available() -> bool:
    """Check if sam_3d_body (PyTorch) is importable."""
    return importlib.util.find_spec("sam_3d_body") is not None


def get_backend(backend: BackendType = "auto") -> str:
    """Resolve backend selection.

    Args:
        backend: "auto" (default - prefer JAX), "jax", or "pytorch"

    Returns:
        Resolved backend name: "jax" or "pytorch"

    Raises:
        ImportError: If requested backend is not available
    """
    if backend == "auto":
        # Prefer JAX backend if available
        if is_jax_backend_available():
            return "jax"
        elif is_pytorch_backend_available():
            return "pytorch"
        else:
            raise ImportError(
                "No SAM-3D-Body backend available. "
                "Install either sam3d-body-eqx (JAX) or sam-3d-body (PyTorch)."
            )
    elif backend == "jax":
        if not is_jax_backend_available():
            raise ImportError(
                "JAX backend requested but sam3d_body_eqx is not installed. "
                "Install with: pip install -e packages/Sam3dBodyEqx"
            )
        return "jax"
    elif backend == "pytorch":
        if not is_pytorch_backend_available():
            raise ImportError(
                "PyTorch backend requested but sam_3d_body is not installed."
            )
        return "pytorch"
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'auto', 'jax', or 'pytorch'.")


@dataclass
class SAM3DResult:
    """Unified result structure for SAM-3D-Body outputs."""

    vertices: np.ndarray  # (N_vertices, 3) mesh vertices
    keypoints_3d: np.ndarray  # (70, 3) 3D joint positions
    keypoints_2d: np.ndarray  # (70, 2) 2D projected keypoints
    camera_t: np.ndarray  # (3,) camera translation
    focal_length: float  # focal length
    body_pose_params: np.ndarray  # body pose parameters
    hand_pose_params: np.ndarray  # hand pose parameters
    shape_params: np.ndarray  # shape parameters
    global_rot: np.ndarray  # (3,) global rotation
    mesh_faces: np.ndarray  # (N_faces, 3) mesh face indices


class SAM3DBackend(ABC):
    """Abstract base class for SAM-3D-Body backends."""

    @abstractmethod
    def load(self, repo_id: str, device: str = "cuda") -> None:
        """Load the model.

        Args:
            repo_id: HuggingFace repository ID or checkpoint path
            device: Device for inference ('cuda' or 'cpu')
        """
        pass

    @abstractmethod
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
        pass

    @property
    @abstractmethod
    def mesh_faces(self) -> np.ndarray:
        """Get mesh face indices (shared across all predictions)."""
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return backend identifier ('jax' or 'pytorch')."""
        pass


@dataclass
class BenchmarkResult:
    """Results from benchmarking inference."""

    backend: str
    warmup_frames: int
    timed_frames: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    fps: float
    times_ms: List[float] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Benchmark ({self.backend}):\n"
            f"  Warmup: {self.warmup_frames} frames\n"
            f"  Timed: {self.timed_frames} frames\n"
            f"  Mean: {self.mean_ms:.2f} ms ({self.fps:.1f} FPS)\n"
            f"  Std: {self.std_ms:.2f} ms\n"
            f"  Min/Max: {self.min_ms:.2f} / {self.max_ms:.2f} ms\n"
            f"  Median: {self.median_ms:.2f} ms"
        )


def benchmark_inference(
    inference_fn: Callable[[], Any],
    warmup_runs: int = 5,
    timed_runs: int = 10,
    backend_name: str = "unknown",
    block_fn: Optional[Callable[[Any], None]] = None,
) -> BenchmarkResult:
    """Run benchmark with warmup and timing.

    Args:
        inference_fn: Function to benchmark (no arguments, returns result)
        warmup_runs: Number of warmup runs (for JIT compilation)
        timed_runs: Number of timed runs for statistics
        backend_name: Name of backend for reporting
        block_fn: Optional function to block until computation completes.
                  For JAX, use: lambda x: jax.block_until_ready(x)
                  For PyTorch with CUDA: lambda x: torch.cuda.synchronize()

    Returns:
        BenchmarkResult with timing statistics
    """
    # Warmup runs (JIT compilation for JAX, CUDA warmup for PyTorch)
    print(f"Running {warmup_runs} warmup frames...")
    for i in range(warmup_runs):
        result = inference_fn()
        if block_fn is not None:
            block_fn(result)

    # Timed runs
    print(f"Running {timed_runs} timed frames...")
    times_ms = []
    for i in range(timed_runs):
        start = time.perf_counter()
        result = inference_fn()
        if block_fn is not None:
            block_fn(result)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)  # Convert to ms

    times_arr = np.array(times_ms)
    mean_ms = float(times_arr.mean())

    return BenchmarkResult(
        backend=backend_name,
        warmup_frames=warmup_runs,
        timed_frames=timed_runs,
        mean_ms=mean_ms,
        std_ms=float(times_arr.std()),
        min_ms=float(times_arr.min()),
        max_ms=float(times_arr.max()),
        median_ms=float(np.median(times_arr)),
        fps=1000.0 / mean_ms if mean_ms > 0 else 0.0,
        times_ms=times_ms,
    )
