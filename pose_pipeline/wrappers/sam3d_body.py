"""
Unified SAM-3D-Body wrapper for PosePipeline.
Consolidates JAX and PyTorch backends into a single, dictionary-driven API.
"""

import os
import time
from typing import Optional, Literal, Dict, Tuple, Any
import cv2
import numpy as np

# Types for backend selection and results
BackendType = Literal["jax", "pytorch"]

def is_jax_available() -> bool:
    """Check if the JAX/Equinox backend package is installed."""
    import importlib.util
    return importlib.util.find_spec("sam3d_body_eqx") is not None

def is_pytorch_available() -> bool:
    """Check if the original PyTorch backend package is installed."""
    import importlib.util
    return importlib.util.find_spec("sam_3d_body") is not None

def process_sam3d_pytorch(
    video_path: str,
    bboxes: np.ndarray,
    present: np.ndarray,
    repo_id: str = "facebook/sam-3d-body-dinov3",
    device: str = "cuda"
) -> Dict[str, np.ndarray]:
    """
    Run SAM3D inference using the PyTorch backend.

    Args:
        video_path: Path to the local MP4 video file.
        bboxes: (N_frames, 4) array of bounding boxes in [x, y, w, h] format.
        present: (N_frames,) boolean mask indicating if the person is present.
        repo_id: HuggingFace repository ID for the model weights.
        device: Torch device string (e.g., "cuda", "cpu").

    Returns:
        Standardized dictionary containing stacked NumPy arrays for all outputs.
        Keys include: vertices, keypoints_3d, keypoints_2d, camera_t, focal_length,
        body_pose_params, hand_pose_params, shape_params, global_rot, and mesh_faces.
    """
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    from sam_3d_body.build_models import _hf_download
    
    # Download and load model
    ckpt_path, mhr_path = _hf_download(repo_id)
    model, model_cfg = load_sam_3d_body(checkpoint_path=ckpt_path, mhr_path=mhr_path, device=device)
    estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg)
    
    num_frames = len(bboxes)
    results = {
        "vertices": [], "keypoints_3d": [], "keypoints_2d": [],
        "camera_t": [], "focal_length": [], "body_pose_params": [],
        "hand_pose_params": [], "shape_params": [], "global_rot": []
    }
    
    cap = cv2.VideoCapture(video_path)
    try:
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret or not present[i]:
                for k in results: results[k].append(None)
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bbox_xywh = bboxes[i]
            # Convert to XYXY for the PyTorch estimator
            bbox_xyxy = np.array([
                bbox_xywh[0], bbox_xywh[1],
                bbox_xywh[0] + bbox_xywh[2],
                bbox_xywh[1] + bbox_xywh[3]
            ]).reshape(1, 4)
            
            outputs = estimator.process_one_image(
                frame_rgb, bboxes=bbox_xyxy, bbox_thr=0.5, use_mask=False, inference_type="full"
            )
            
            if not outputs:
                for k in results: results[k].append(None)
                continue
                
            p = outputs[0]
            # Convert torch tensors to numpy and store
            results["vertices"].append(p["pred_vertices"].detach().cpu().numpy())
            results["keypoints_3d"].append(p["pred_keypoints_3d"].detach().cpu().numpy())
            results["keypoints_2d"].append(p["pred_keypoints_2d"].detach().cpu().numpy())
            results["camera_t"].append(p["pred_cam_t"].detach().cpu().numpy())
            results["focal_length"].append(float(p["focal_length"].detach().cpu().numpy()))
            results["body_pose_params"].append(p["body_pose_params"].detach().cpu().numpy())
            results["hand_pose_params"].append(p["hand_pose_params"].detach().cpu().numpy())
            results["shape_params"].append(p["shape_params"].detach().cpu().numpy())
            results["global_rot"].append(p["global_rot"].detach().cpu().numpy())
    finally:
        cap.release()
        
    def stack(l):
        if all(x is None for x in l): return None
        first = next(x for x in l if x is not None)
        shape = first.shape if hasattr(first, "shape") else ()
        out = np.full((len(l), *shape), np.nan)
        for i, x in enumerate(l):
            if x is not None: out[i] = x
        return out

    final = {k: stack(v) for k, v in results.items()}
    final["mesh_faces"] = estimator.faces
    return final

def process_sam3d_jax(
    video_path: str,
    bboxes: np.ndarray,
    present: np.ndarray,
    repo_id: str = "facebook/sam-3d-body-dinov3",
) -> Dict[str, np.ndarray]:
    """
    Run SAM3D inference using the JAX/Equinox backend.

    Args:
        video_path: Path to the local MP4 video file.
        bboxes: (N_frames, 4) array of bounding boxes in [x, y, w, h] format.
        present: (N_frames,) boolean mask indicating if the person is present.
        repo_id: HuggingFace repository ID for the model weights.

    Returns:
        Standardized dictionary containing stacked NumPy arrays for all outputs.
    """
    from sam3d_body_eqx.inference import SAM3DBodyEstimator
    from sam3d_body_eqx.inference.utils import stack_sequence_results
    
    # Load JAX estimator (uses helper function in sam3d_body_eqx)
    estimator = SAM3DBodyEstimator.from_pretrained()
    
    results_list = []
    # Use the estimator's sequence processing generator
    generator = estimator.predict_video_with_bboxes(
        input_path=video_path, bboxes=bboxes, present_mask=present, show_progress=True
    )
    
    for _, _, frame_results in generator:
        res = frame_results[0] if frame_results else None
        if res:
            # Map JAX-specific output keys to the standardized PosePipeline API
            results_list.append({
                "vertices": np.asarray(res["pred_vertices"]),
                "keypoints_3d": np.asarray(res["pred_keypoints_3d"]),
                "keypoints_2d": np.asarray(res["pred_keypoints_2d"]),
                "camera_t": np.asarray(res["pred_cam_t"]),
                "focal_length": float(res["focal_length"]),
                "body_pose_params": np.asarray(res["body_pose"]),
                "hand_pose_params": np.asarray(res["hand"]),
                "shape_params": np.asarray(res["shape"]),
                "global_rot": np.asarray(res["global_rot"]),
            })
        else:
            results_list.append(None)
            
    # Consolidate frames using the JAX-helper stack utility
    final = stack_sequence_results(results_list)
    final["mesh_faces"] = np.asarray(estimator.model.head_pose.mhr.faces)
    return final

def process_sam3d_body(
    key: Dict[str, Any], 
    repo_id: str = "facebook/sam-3d-body-dinov3", 
    backend: BackendType = "jax"
) -> Dict[str, np.ndarray]:
    """
    Main entry point for SAM3D processing in DataJoint pipelines.

    Args:
        key: DataJoint primary key dictionary.
        repo_id: HuggingFace repository ID for model weights.
        backend: Explicit backend selection: "jax" or "pytorch".

    Returns:
        Standardized results dictionary ready for DataJoint insertion.
    """
    from pose_pipeline import PersonBbox, Video
    
    if backend == "jax" and not is_jax_available():
        raise ImportError("JAX backend requested but sam3d_body_eqx not installed.")
    if backend == "pytorch" and not is_pytorch_available():
        raise ImportError("PyTorch backend requested but sam_3d_body not installed.")

    # 1. Fetch data from parent tables
    video_path, bboxes, present = (Video * PersonBbox & key).fetch1("video", "bbox", "present")
    
    # 2. Dispatch to specific backend implementation
    if backend == "jax":
        results = process_sam3d_jax(video_path, bboxes, present, repo_id=repo_id)
    else:
        results = process_sam3d_pytorch(video_path, bboxes, present, repo_id=repo_id, device="cuda")
        
    # 3. Finalize and return
    results.update(key)
    results["frame_valid"] = present
    
    # Cleanup DataJoint temp file
    if "tmp" in str(video_path) and os.path.exists(video_path):
        try: os.remove(video_path)
        except: pass
        
    return results

def get_sam3d_callback(key: Dict[str, Any], mesh_color: Tuple[float, float, float] = (0.65, 0.74, 0.86)):
    """
    Create a visualization callback for rendering SAM3D mesh overlays.

    Args:
        key: DataJoint key to fetch results.
        mesh_color: Normalized RGB tuple (0.0 to 1.0).

    Returns:
        A function: overlay(image, frame_index) -> visualized_image.
    """
    from sam_3d_body.visualization.renderer import Renderer
    from pose_pipeline import SAM3DBody
    
    data = (SAM3DBody & key).fetch1()
    vertices, faces, camera_t, focal_length = data["vertices"], data["mesh_faces"], data["camera_t"], data["focal_length"]
    valid = data.get("frame_valid", np.ones(len(vertices), dtype=bool))
    
    _cache = {}
    def overlay(image, idx):
        if not valid[idx] or np.any(np.isnan(vertices[idx])):
            return image
        
        fl = focal_length[idx] if not np.isnan(focal_length[idx]) else 1000.0
        if fl not in _cache:
            _cache[fl] = Renderer(focal_length=fl, faces=faces)
        
        rendered = _cache[fl](vertices[idx], camera_t[idx], image.copy(), mesh_base_color=mesh_color)
        return (rendered * 255).astype(np.uint8)
        
    return overlay
