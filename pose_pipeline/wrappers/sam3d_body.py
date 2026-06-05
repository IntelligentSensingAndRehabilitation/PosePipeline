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
import json
from pathlib import Path


from sam3d_body_eqx.visualization.skeleton import MHR70_KEYPOINT_NAMES


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


sam_vertex_movi_names = [
    "backneck",
    "upperback",
    "clavicle",
    "sternum",
    "umbilicus",
    "lfronthead",
    "lbackhead",
    "lback",
    "lshom",
    "lupperarm",
    "lelbm",
    "lforearm",
    "lwrithumbside",
    "lwripinkieside",
    "lfin",
    "lasis",
    "lpsis",
    "lfrontthigh",
    "lthigh",
    "lknem",
    "lankm",
    "LHeel",
    "lfifthmetatarsal",
    "LBigToe",
    "lcheek",
    "lbreast",
    "lelbinner",
    "lwaist",
    "lthumb",
    "lfrontinnerthigh",
    "linnerknee",
    "lshin",
    "lfirstmetatarsal",
    "lfourthtoe",
    "lscapula",
    "lbum",
    "rfronthead",
    "rbackhead",
    "rback",
    "rshom",
    "rupperarm",
    "relbm",
    "rforearm",
    "rwrithumbside",
    "rwripinkieside",
    "rfin",
    "rasis",
    "rpsis",
    "rfrontthigh",
    "rthigh",
    "rknem",
    "rankm",
    "RHeel",
    "rfifthmetatarsal",
    "RBigToe",
    "rcheek",
    "rbreast",
    "relbinner",
    "rwaist",
    "rthumb",
    "rfrontinnerthigh",
    "rinnerknee",
    "rshin",
    "rfirstmetatarsal",
    "rfourthtoe",
    "rscapula",
    "rbum",
    "Head",
    "mhip",
    "CHip",
    "Neck",
    "LAnkle",
    "LElbow",
    "LHip",
    "LHand",
    "LKnee",
    "LShoulder",
    "LWrist",
    "LFoot",
    "RAnkle",
    "RElbow",
    "RHip",
    "RHand",
    "RKnee",
    "RShoulder",
    "RWrist",
    "RFoot",
]

sam_kinematic_node_names = ['body_world',
 'root',
 'l_upleg',
 'l_lowleg',
 'l_foot',
 'l_talocrural',
 'l_subtalar',
 'l_transversetarsal',
 'l_ball',
 'l_lowleg_twist1_proc',
 'l_lowleg_twist2_proc',
 'l_lowleg_twist3_proc',
 'l_lowleg_twist4_proc',
 'l_upleg_twist0_proc',
 'l_upleg_twist1_proc',
 'l_upleg_twist2_proc',
 'l_upleg_twist3_proc',
 'l_upleg_twist4_proc',
 'r_upleg',
 'r_lowleg',
 'r_foot',
 'r_talocrural',
 'r_subtalar',
 'r_transversetarsal',
 'r_ball',
 'r_lowleg_twist1_proc',
 'r_lowleg_twist2_proc',
 'r_lowleg_twist3_proc',
 'r_lowleg_twist4_proc',
 'r_upleg_twist0_proc',
 'r_upleg_twist1_proc',
 'r_upleg_twist2_proc',
 'r_upleg_twist3_proc',
 'r_upleg_twist4_proc',
 'c_spine0',
 'c_spine1',
 'c_spine2',
 'c_spine3',
 'r_clavicle',
 'r_uparm',
 'r_lowarm',
 'r_wrist_twist',
 'r_wrist',
 'r_pinky0',
 'r_pinky1',
 'r_pinky2',
 'r_pinky3',
 'r_pinky_null',
 'r_ring1',
 'r_ring2',
 'r_ring3',
 'r_ring_null',
 'r_middle1',
 'r_middle2',
 'r_middle3',
 'r_middle_null',
 'r_index1',
 'r_index2',
 'r_index3',
 'r_index_null',
 'r_thumb0',
 'r_thumb1',
 'r_thumb2',
 'r_thumb3',
 'r_thumb_null',
 'r_lowarm_twist1_proc',
 'r_lowarm_twist2_proc',
 'r_lowarm_twist3_proc',
 'r_lowarm_twist4_proc',
 'r_uparm_twist0_proc',
 'r_uparm_twist1_proc',
 'r_uparm_twist2_proc',
 'r_uparm_twist3_proc',
 'r_uparm_twist4_proc',
 'l_clavicle',
 'l_uparm',
 'l_lowarm',
 'l_wrist_twist',
 'l_wrist',
 'l_pinky0',
 'l_pinky1',
 'l_pinky2',
 'l_pinky3',
 'l_pinky_null',
 'l_ring1',
 'l_ring2',
 'l_ring3',
 'l_ring_null',
 'l_middle1',
 'l_middle2',
 'l_middle3',
 'l_middle_null',
 'l_index1',
 'l_index2',
 'l_index3',
 'l_index_null',
 'l_thumb0',
 'l_thumb1',
 'l_thumb2',
 'l_thumb3',
 'l_thumb_null',
 'l_lowarm_twist1_proc',
 'l_lowarm_twist2_proc',
 'l_lowarm_twist3_proc',
 'l_lowarm_twist4_proc',
 'l_uparm_twist0_proc',
 'l_uparm_twist1_proc',
 'l_uparm_twist2_proc',
 'l_uparm_twist3_proc',
 'l_uparm_twist4_proc',
 'c_neck',
 'c_neck_twist1_proc',
 'c_neck_twist0_proc',
 'c_head',
 'c_jaw',
 'c_teeth',
 'c_jaw_null',
 'c_tongue0',
 'c_tongue1',
 'c_tongue2',
 'c_tongue3',
 'c_tongue4',
 'r_eye',
 'r_eye_null',
 'l_eye',
 'l_eye_null',
 'c_head_null']

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
        "camera_t": [], "focal_length": [], "body_pose_params": [],
        "hand_pose_params": [], "shape_params": [], "scale_params": [],
        "global_rot": [],
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
            results["camera_t"].append(p.get("pred_cam_t"))
            results["focal_length"].append(float(p.get("focal_length", 0.0)))
            results["body_pose_params"].append(p.get("body_pose_params"))
            results["hand_pose_params"].append(p.get("hand_pose_params"))
            results["shape_params"].append(p.get("shape_params"))
            results["scale_params"].append(p.get("scale_params", p.get("scale")))
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
                "camera_t": np.asarray(res["pred_cam_t"]) if res.get("pred_cam_t") is not None else None,
                "focal_length": float(res["focal_length"]) if res.get("focal_length") is not None else None,
                "body_pose_params": np.asarray(res["body_pose"]),
                "hand_pose_params": np.asarray(res["hand"]),
                "shape_params": np.asarray(res["shape"]),
                "scale_params": np.asarray(res["scale"]),
                "global_rot": np.asarray(res["global_rot"]),
            })
        else:
            results_list.append(None)

    # Consolidate frames using the JAX-helper stack utility
    final = stack_sequence_results(results_list)
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
    
    if method_name in ("jax", "jax_hands", "jax_hands2") and not is_jax_available():
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
    elif method_name == "jax_hands2":
        results = process_sam3d_jax(video_path, bboxes, present, use_hands=True, batch_size=16)
    elif method_name == "torch_dinov3":
        results = process_sam3d_pytorch(video_path, bboxes, present, repo_id="facebook/sam-3d-body-dinov3")
    else:
        raise ValueError(f"Unknown SAM3D method: {method_name}")
        
    # 3. Finalize and return
    results.update(key)
    results["frame_valid"] = present
    
    # Cleanup DataJoint temp file
    if os.path.exists(video_path):
        os.remove(video_path)

    return results


def compute_sam3d_geometry(
    body_pose_params: np.ndarray,
    shape_params: np.ndarray,
    scale_params: np.ndarray,
    hand_pose_params: np.ndarray,
    global_rot: np.ndarray,
    camera_t: Optional[np.ndarray] = None,
    focal_length: Optional[np.ndarray] = None,
    image_size: Optional[Tuple[int, int]] = None,
    depth_tolerance: float = 0.05,
    return_vertices: bool = True,
    return_joints: bool = True,
) -> Dict[str, np.ndarray]:
    """Reconstruct MHR mesh vertices and kinematic joints from stored minimal parameters.

    Delegates to SAM3DBodyEstimator.compute_geometry() (requires sam3d_body_eqx).
    Vertices/joints are returned in body/root space. Apply camera_t separately for
    2D projection.

    When camera_t, focal_length, and image_size are provided, self-occlusion
    visibility masks are also returned (requires pyrender + trimesh).

    Args:
        body_pose_params:  (N, 133) body pose Euler angles (XYZ).
        shape_params:      (N, 45) shape PCA coefficients.
        scale_params:      (N, 28) scale PCA coefficients.
        hand_pose_params:  (N, 108) hand pose: columns 0:54 left, 54:108 right (continuous).
        global_rot:        (N, 3) global rotation (ZYX Euler).
        camera_t:          (N, 3) camera translations — required for visibility.
        focal_length:      (N,) focal lengths in pixels — required for visibility.
        image_size:        (H, W) image dimensions — required for visibility.
        depth_tolerance:   Occlusion tolerance in metres (default 5 cm).
        return_vertices:   Whether to include mesh vertices (N, 18439, 3) in output.
        return_joints:     Whether to include kinematic tree joints (N, 127, 3) in output.

    Returns:
        dict with keys: 'keypoints_3d' always; 'vertices' and/or 'joints' when requested;
        'keypoints_visibility', 'joints_visibility', 'vertices_visibility' when camera
        params are provided.
    """
    from sam3d_body_eqx.inference import SAM3DBodyEstimator

    estimator = SAM3DBodyEstimator.from_pretrained()
    return estimator.compute_geometry(
        body_pose_params=body_pose_params,
        shape_params=shape_params,
        scale_params=scale_params,
        hand_pose_params=hand_pose_params,
        global_rot=global_rot,
        camera_t=camera_t,
        focal_length=focal_length,
        image_size=image_size,
        depth_tolerance=depth_tolerance,
        return_vertices=return_vertices,
        return_joints=return_joints,
    )


def get_sam3d_callback(key: Dict[str, Any], mesh_color: Tuple[float, float, float] = (0.65, 0.74, 0.86)):
    """
    Create a visualization callback for rendering SAM3D mesh overlays.

    Args:
        key: DataJoint key to fetch results.
        mesh_color: Normalized RGB tuple (0.0 to 1.0).

    Returns:
        A function: overlay(image, frame_index) -> visualized_image.
    """
    from sam3d_body_eqx.inference import SAM3DBodyEstimator
    from sam3d_body_eqx.visualization.mesh import render_mesh
    from pose_pipeline import SAM3DBody

    sam3d_entry = SAM3DBody & key
    data = sam3d_entry.fetch1("camera_t", "focal_length", "frame_valid")
    geom = sam3d_entry.fetch_geometry(return_vertices=True, return_joints=False)
    vertices = geom["vertices"]
    estimator = SAM3DBodyEstimator.from_pretrained()
    faces = np.asarray(estimator.model.head_pose.mhr.faces)
    camera_t, focal_length = data["camera_t"], data["focal_length"]
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