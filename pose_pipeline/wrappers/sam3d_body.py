# SAM-3D-Body wrapper for PosePipeline
# Meta's state-of-the-art single-image full-body 3D human mesh recovery model
# Uses MHR (Momentum Human Rig) representation instead of SMPL
#
# Supports two backends:
#   - JAX/Equinox (sam3d_body_eqx): pip install -e packages/Sam3dBodyEqx
#   - PyTorch (sam_3d_body): pip install -e sam-3d-body/
#
# By default, prefers JAX backend if available (faster, no PyTorch runtime needed)

import os
import time
from typing import Literal, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .sam3d_backend import (
    BackendType,
    BenchmarkResult,
    SAM3DBackend,
    SAM3DResult,
    benchmark_inference,
    get_backend,
    is_jax_backend_available,
    is_pytorch_backend_available,
)

# Lazy-loaded backend classes
_backend_classes = {}


def _get_backend_class(backend_name: str) -> type:
    """Lazily import and return backend class."""
    if backend_name not in _backend_classes:
        if backend_name == "jax":
            from ._jax_backend import JAXSAM3DBackend

            _backend_classes["jax"] = JAXSAM3DBackend
        elif backend_name == "pytorch":
            from ._pytorch_backend import PyTorchSAM3DBackend

            _backend_classes["pytorch"] = PyTorchSAM3DBackend
    return _backend_classes[backend_name]


def _append_invalid_frame(results):
    """Append placeholder values for an invalid frame."""
    results["frame_valid"].append(False)
    results["vertices"].append(None)
    results["keypoints_3d"].append(None)
    results["keypoints_2d"].append(None)
    results["camera_t"].append(None)
    results["focal_length"].append(None)
    results["body_pose_params"].append(None)
    results["hand_pose_params"].append(None)
    results["shape_params"].append(None)
    results["global_rot"].append(None)


def _append_result(results, result: SAM3DResult):
    """Append SAM3DResult to results accumulators."""
    results["frame_valid"].append(True)
    results["vertices"].append(result.vertices)
    results["keypoints_3d"].append(result.keypoints_3d)
    results["keypoints_2d"].append(result.keypoints_2d)
    results["camera_t"].append(result.camera_t)
    results["focal_length"].append(result.focal_length)
    results["body_pose_params"].append(result.body_pose_params)
    results["hand_pose_params"].append(result.hand_pose_params)
    results["shape_params"].append(result.shape_params)
    results["global_rot"].append(result.global_rot)


def load_sam3d_body(
    repo_id: str = "facebook/sam-3d-body-dinov3",
    device: str = "cuda",
    backend: BackendType = "auto",
) -> SAM3DBackend:
    """
    Load SAM-3D-Body model with specified backend.

    Args:
        repo_id: HuggingFace repository ID (used by PyTorch backend)
        device: Device for inference ('cuda' or 'cpu')
        backend: 'auto' (prefer JAX), 'jax', or 'pytorch'

    Returns:
        SAM3DBackend instance (either JAX or PyTorch implementation)
    """
    resolved_backend = get_backend(backend)
    backend_cls = _get_backend_class(resolved_backend)
    backend_instance = backend_cls()
    backend_instance.load(repo_id, device)
    return backend_instance


def process_sam3d_body(
    key,
    repo_id: str = "facebook/sam-3d-body-dinov3",
    backend: BackendType = "auto",
    benchmark: bool = False,
    benchmark_warmup: int = 5,
    benchmark_frames: int = 20,
) -> dict:
    """
    Process video with SAM-3D-Body model.

    Uses pre-computed bounding boxes from PosePipeline's tracking stage.
    SAM-3D-Body expects full images with bboxes and handles cropping internally.
    Outputs MHR (Momentum Human Rig) format natively.

    Args:
        key: DataJoint key with video/person references
        repo_id: HuggingFace repository ID for the model
        backend: 'auto' (prefer JAX), 'jax', or 'pytorch'
        benchmark: If True, collect timing statistics (printed at end)
        benchmark_warmup: Warmup frames before timing (for JIT compilation)
        benchmark_frames: Number of frames to time for statistics

    Returns:
        dict with MHR outputs:
            - vertices: [N_frames, N_vertices, 3] mesh vertices
            - keypoints_3d: [N_frames, 70, 3] 3D joint positions
            - keypoints_2d: [N_frames, 70, 2] 2D projected keypoints
            - camera_t: [N_frames, 3] camera translations
            - focal_length: [N_frames] focal lengths
            - body_pose_params: [N_frames, ...] MHR body pose parameters
            - hand_pose_params: [N_frames, ...] MHR hand pose parameters
            - shape_params: [N_frames, ...] MHR shape parameters
            - mesh_faces: mesh face indices (same for all frames)
    """
    from pose_pipeline import PersonBbox, Video, VideoInfo

    # Get video and bounding boxes
    video_path, bboxes_dj, present_dj = (Video * PersonBbox & key).fetch1(
        "video", "bbox", "present"
    )
    num_frames, height, width = (VideoInfo & key).fetch1(
        "num_frames", "height", "width"
    )

    # Load model with selected backend
    # Check for CUDA availability (works for both PyTorch and JAX)
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        # No PyTorch - JAX will auto-detect GPU
        device = "cuda"

    sam3d_backend = load_sam3d_body(repo_id=repo_id, device=device, backend=backend)
    print(f"Using SAM-3D-Body backend: {sam3d_backend.backend_name}")

    # Get mesh faces (same for all frames)
    mesh_faces = sam3d_backend.mesh_faces

    # Initialize result accumulators
    results = {
        "vertices": [],
        "keypoints_3d": [],
        "keypoints_2d": [],
        "camera_t": [],
        "focal_length": [],
        "body_pose_params": [],
        "hand_pose_params": [],
        "shape_params": [],
        "global_rot": [],
        "frame_valid": [],
    }

    # Benchmarking setup
    timing_data = [] if benchmark else None
    warmup_remaining = benchmark_warmup if benchmark else 0

    # Open video with proper resource management
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        for idx in tqdm(range(num_frames), desc=f"Processing SAM-3D-Body ({sam3d_backend.backend_name})"):
            ret, frame = cap.read()
            if not ret:
                # Video ended early - append placeholders for remaining frames
                for _ in range(idx, num_frames):
                    _append_invalid_frame(results)
                break

            # Check if person is present in this frame
            if not present_dj[idx]:
                _append_invalid_frame(results)
                continue

            # Get bounding box for this frame
            # Note: PosePipeline's "TLHW" is actually XYWH format [x, y, width, height]
            bbox_xywh = bboxes_dj[idx]

            # Convert XYWH to XYXY (x1, y1, x2, y2) format for SAM-3D-Body
            bbox_xyxy = np.array(
                [
                    bbox_xywh[0],  # x1 (left)
                    bbox_xywh[1],  # y1 (top)
                    bbox_xywh[0] + bbox_xywh[2],  # x2 = x1 + width
                    bbox_xywh[1] + bbox_xywh[3],  # y2 = y1 + height
                ]
            )

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                # Time inference if benchmarking
                if benchmark and warmup_remaining <= 0 and len(timing_data) < benchmark_frames:
                    start_time = time.perf_counter()
                    result = sam3d_backend.predict(frame_rgb, bbox_xyxy)
                    # For JAX, we need to block until computation completes
                    if sam3d_backend.backend_name == "jax" and result is not None:
                        import jax
                        jax.block_until_ready(result.vertices)
                    end_time = time.perf_counter()
                    timing_data.append((end_time - start_time) * 1000)  # ms
                else:
                    result = sam3d_backend.predict(frame_rgb, bbox_xyxy)
                    if warmup_remaining > 0:
                        warmup_remaining -= 1

                if result is not None:
                    _append_result(results, result)
                else:
                    _append_invalid_frame(results)

            except (RuntimeError, ValueError) as e:
                print(f"Error processing frame {idx}: {e}")
                _append_invalid_frame(results)

    finally:
        cap.release()

    # Clean up video file (DataJoint fetches to temp file)
    try:
        os.remove(video_path)
    except OSError:
        pass  # File may already be removed or inaccessible

    # Print benchmark results if collected
    if benchmark and timing_data:
        times_arr = np.array(timing_data)
        print(f"\n{'='*50}")
        print(f"Benchmark Results ({sam3d_backend.backend_name})")
        print(f"{'='*50}")
        print(f"  Warmup frames: {benchmark_warmup}")
        print(f"  Timed frames: {len(timing_data)}")
        print(f"  Mean: {times_arr.mean():.2f} ms ({1000/times_arr.mean():.1f} FPS)")
        print(f"  Std: {times_arr.std():.2f} ms")
        print(f"  Min/Max: {times_arr.min():.2f} / {times_arr.max():.2f} ms")
        print(f"  Median: {np.median(times_arr):.2f} ms")
        print(f"{'='*50}\n")

    # Convert lists to arrays, handling None values
    def stack_with_nans(data_list, shape_if_none=None):
        """Stack list into array, using NaN for None values."""
        if all(x is None for x in data_list):
            return None

        # Find first non-None to get shape (or detect scalar)
        is_scalar = False
        for x in data_list:
            if x is not None:
                if hasattr(x, "shape"):
                    shape_if_none = x.shape
                else:
                    # Scalar value (e.g., focal_length is a float)
                    is_scalar = True
                break

        if is_scalar:
            # Handle list of scalars
            result = np.full(len(data_list), np.nan)
            for i, x in enumerate(data_list):
                if x is not None:
                    result[i] = x
            return result

        if shape_if_none is None:
            return None

        result = np.full((len(data_list), *shape_if_none), np.nan)
        for i, x in enumerate(data_list):
            if x is not None:
                result[i] = x
        return result

    # Stack results
    final_results = {
        "vertices": stack_with_nans(results["vertices"]),
        "keypoints_3d": stack_with_nans(results["keypoints_3d"]),
        "keypoints_2d": stack_with_nans(results["keypoints_2d"]),
        "camera_t": stack_with_nans(results["camera_t"]),
        "focal_length": stack_with_nans(results["focal_length"]),
        "body_pose_params": stack_with_nans(results["body_pose_params"]),
        "hand_pose_params": stack_with_nans(results["hand_pose_params"]),
        "shape_params": stack_with_nans(results["shape_params"]),
        "global_rot": stack_with_nans(results["global_rot"]),
        "mesh_faces": mesh_faces,
        "frame_valid": np.array(results["frame_valid"]),
    }

    final_results.update(key)

    return final_results


def get_sam3d_callback(key, mesh_color=(0.65098039, 0.74117647, 0.85882353)):
    """
    Get visualization callback for SAM-3D-Body mesh overlay.

    Args:
        key: DataJoint key to fetch SAM3DBody results
        mesh_color: RGB tuple for mesh color (default: light blue)

    Returns:
        Callback function for video_overlay
    """
    from sam_3d_body.visualization.renderer import Renderer

    from pose_pipeline import SAM3DBody

    # Fetch results
    data = (SAM3DBody & key).fetch1()
    vertices = data["vertices"]
    faces = data["mesh_faces"]
    camera_t = data["camera_t"]
    focal_length = data["focal_length"]
    frame_valid = data.get("frame_valid", np.ones(len(vertices), dtype=bool))

    # Cache renderer per focal length to avoid recreation
    _renderer_cache = {}

    def overlay(image, idx):
        """Overlay mesh on frame."""
        if not frame_valid[idx] or np.any(np.isnan(vertices[idx])):
            return image

        # Get focal length for this frame
        fl = focal_length[idx] if not np.isnan(focal_length[idx]) else 1000.0

        # Get or create renderer for this focal length
        if fl not in _renderer_cache:
            _renderer_cache[fl] = Renderer(focal_length=fl, faces=faces)
        renderer = _renderer_cache[fl]

        # Render mesh overlay
        rendered = renderer(
            vertices[idx],
            camera_t[idx],
            image.copy(),
            mesh_base_color=mesh_color,
            scene_bg_color=(1, 1, 1),
        )

        return (rendered * 255).astype(np.uint8)

    return overlay
