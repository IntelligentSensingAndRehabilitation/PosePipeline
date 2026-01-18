"""
Unified SAM-3D-Body wrapper for PosePipeline.
Consolidates JAX and PyTorch backends into a single, dictionary-driven API.
"""

import os
import time
from typing import Optional, Literal, Dict, Tuple, Any
import cv2
import numpy as np
from tqdm import tqdm


MHR70_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "right_thumb4",
    "right_thumb3",
    "right_thumb2",
    "right_thumb_third_joint",
    "right_forefinger4",
    "right_forefinger3",
    "right_forefinger2",
    "right_forefinger_third_joint",
    "right_middle_finger4",
    "right_middle_finger3",
    "right_middle_finger2",
    "right_middle_finger_third_joint",
    "right_ring_finger4",
    "right_ring_finger3",
    "right_ring_finger2",
    "right_ring_finger_third_joint",
    "right_pinky_finger4",
    "right_pinky_finger3",
    "right_pinky_finger2",
    "right_pinky_finger_third_joint",
    "right_wrist",
    "left_thumb4",
    "left_thumb3",
    "left_thumb2",
    "left_thumb_third_joint",
    "left_forefinger4",
    "left_forefinger3",
    "left_forefinger2",
    "left_forefinger_third_joint",
    "left_middle_finger4",
    "left_middle_finger3",
    "left_middle_finger2",
    "left_middle_finger_third_joint",
    "left_ring_finger4",
    "left_ring_finger3",
    "left_ring_finger2",
    "left_ring_finger_third_joint",
    "left_pinky_finger4",
    "left_pinky_finger3",
    "left_pinky_finger2",
    "left_pinky_finger_third_joint",
    "left_wrist",
    "left_olecranon",
    "right_olecranon",
    "left_cubital_fossa",
    "right_cubital_fossa",
    "left_acromion",
    "right_acromion",
    "neck",
]


def get_joint_names(normalize=True):
    """Return MHR 70 joint names.

    Args:
        normalize: If True (default), convert to Title Case (left_hip -> Left Hip)
                   to match normalized_joint_name_dictionary convention used elsewhere.
                   If False, return original naming (lowercase with underscores).
    """

    if normalize:
        names = [name.replace("_", " ").title() for name in MHR70_KEYPOINT_NAMES]
        return list(names)
    else:
        return list(MHR70_KEYPOINT_NAMES)

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
        for i in tqdm(range(num_frames), desc="SAM3D (Torch)"):
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
            
            # Original PyTorch code handles conversion to numpy internally
            outputs = estimator.process_one_image(
                frame_rgb, bboxes=bbox_xyxy, bbox_thr=0.5, use_mask=False, inference_type="full"
            )
            
            if not outputs:
                for k in results: results[k].append(None)
                continue
                
            p = outputs[0]
            # Store outputs (already NumPy arrays)
            results["vertices"].append(p.get("pred_vertices"))
            results["keypoints_3d"].append(p.get("pred_keypoints_3d"))
            results["keypoints_2d"].append(p.get("pred_keypoints_2d"))
            results["camera_t"].append(p.get("pred_cam_t"))
            results["focal_length"].append(float(p.get("focal_length", 0.0)))
            results["body_pose_params"].append(p.get("body_pose_params"))
            results["hand_pose_params"].append(p.get("hand_pose_params"))
            results["shape_params"].append(p.get("shape_params"))
            results["global_rot"].append(p.get("global_rot"))
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
    use_hands: bool = False,
    batch_size: int = 4,
) -> Dict[str, np.ndarray]:
    """
    Run SAM3D inference using the JAX/Equinox backend with batched processing.

    Args:
        video_path: Path to the local MP4 video file.
        bboxes: (N_frames, 4) array of bounding boxes in [x, y, w, h] format.
        present: (N_frames,) boolean mask indicating if the person is present.
        use_hands: Whether to enable hand refinement pipeline.
        batch_size: Number of frames to process at once (default 4 for best performance).

    Returns:
        Standardized dictionary containing stacked NumPy arrays for all outputs.
    """
    from sam3d_body_eqx.inference import SAM3DBodyEstimator
    from sam3d_body_eqx.inference.utils import stack_sequence_results

    # Load JAX estimator
    estimator = SAM3DBodyEstimator.from_pretrained()

    # Convert bboxes from [x, y, w, h] to [x1, y1, x2, y2] format
    # The estimator expects XYXY format
    bboxes_xyxy = np.column_stack([
        bboxes[:, 0],                      # x1
        bboxes[:, 1],                      # y1
        bboxes[:, 0] + bboxes[:, 2],       # x2 = x + w
        bboxes[:, 1] + bboxes[:, 3],       # y2 = y + h
    ])

    results_list = []
    # Use the batched video processing for faster inference
    generator = estimator.predict_video_batched(
        input_path=video_path,
        bboxes=bboxes_xyxy,
        present_mask=present,
        batch_size=batch_size,
        use_hands=use_hands,
        show_progress=True,
    )

    for _, res in generator:
        if res is not None:
            # Map JAX-specific output keys to the standardized PosePipeline API
            results_list.append({
                "vertices": np.asarray(res["pred_vertices"]) if res.get("pred_vertices") is not None else None,
                "keypoints_3d": np.asarray(res["pred_keypoints_3d"]),
                "keypoints_2d": np.asarray(res["pred_keypoints_2d"]) if res.get("pred_keypoints_2d") is not None else None,
                "camera_t": np.asarray(res["pred_cam_t"]) if res.get("pred_cam_t") is not None else None,
                "focal_length": float(res["focal_length"]) if res.get("focal_length") is not None else None,
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
    method_name: str = "jax"
) -> Dict[str, np.ndarray]:
    """
    Main entry point for SAM3D processing in DataJoint pipelines.

    Args:
        key: DataJoint primary key dictionary.
        method_name: Explicit backend selection: "jax" or "torch_dinov3".

    Returns:
        Standardized results dictionary ready for DataJoint insertion.
    """
    from pose_pipeline import PersonBbox, Video
    
    if method_name in ("jax", "jax_hands") and not is_jax_available():
        raise ImportError("JAX backend requested but sam3d_body_eqx not installed.")
    if method_name == "torch_dinov3" and not is_pytorch_available():
        raise ImportError("PyTorch backend requested but sam_3d_body not installed.")

    # 1. Fetch data from parent tables
    video_path, bboxes, present = (Video * PersonBbox & key).fetch1("video", "bbox", "present")

    # 2. Dispatch based on method name
    if method_name == "jax":
        results = process_sam3d_jax(video_path, bboxes, present, use_hands=False)
    elif method_name == "jax_hands":
        results = process_sam3d_jax(video_path, bboxes, present, use_hands=True)
    elif method_name == "torch_dinov3":
        results = process_sam3d_pytorch(video_path, bboxes, present, repo_id="facebook/sam-3d-body-dinov3")
    else:
        raise ValueError(f"Unknown SAM3D method: {method_name}")
        
    # 3. Finalize and return
    results.update(key)
    results["frame_valid"] = present
    
    # Cleanup DataJoint temp file
    if "tmp" in str(video_path) and os.path.exists(video_path):
        os.remove(video_path)

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
    from sam3d_body_eqx.visualization.mesh import render_mesh
    from pose_pipeline import SAM3DBody

    data = (SAM3DBody & key).fetch1()
    vertices, faces, camera_t, focal_length = data["vertices"], data["mesh_faces"], data["camera_t"], data["focal_length"]
    valid = data.get("frame_valid", np.ones(len(vertices), dtype=bool))

    def overlay(image, idx):
        if not valid[idx] or np.any(np.isnan(vertices[idx])):
            return image

        fl = focal_length[idx] if not np.isnan(focal_length[idx]) else 1000.0

        rendered = render_mesh(
            image=image,
            vertices=vertices[idx],
            faces=faces,
            camera_translation=camera_t[idx],
            focal_length=fl,
            mesh_color=mesh_color,
        )
        return rendered

    return overlay