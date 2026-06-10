"""
Unified SAM-3D-Body wrapper for PosePipeline.
Consolidates JAX and PyTorch backends into a single, dictionary-driven API.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import cv2
import numpy as np
from tqdm import tqdm


MHR70_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
    "right_thumb4", "right_thumb3", "right_thumb2", "right_thumb_third_joint",
    "right_forefinger4", "right_forefinger3", "right_forefinger2", "right_forefinger_third_joint",
    "right_middle_finger4", "right_middle_finger3", "right_middle_finger2", "right_middle_finger_third_joint",
    "right_ring_finger4", "right_ring_finger3", "right_ring_finger2", "right_ring_finger_third_joint",
    "right_pinky_finger4", "right_pinky_finger3", "right_pinky_finger2", "right_pinky_finger_third_joint",
    "right_wrist",
    "left_thumb4", "left_thumb3", "left_thumb2", "left_thumb_third_joint",
    "left_forefinger4", "left_forefinger3", "left_forefinger2", "left_forefinger_third_joint",
    "left_middle_finger4", "left_middle_finger3", "left_middle_finger2", "left_middle_finger_third_joint",
    "left_ring_finger4", "left_ring_finger3", "left_ring_finger2", "left_ring_finger_third_joint",
    "left_pinky_finger4", "left_pinky_finger3", "left_pinky_finger2", "left_pinky_finger_third_joint",
    "left_wrist",
    "left_olecranon", "right_olecranon",
    "left_cubital_fossa", "right_cubital_fossa",
    "left_acromion", "right_acromion",
    "neck",
]


def get_joint_names(normalize=True):
    """Return MHR 70 joint names. Works for both JAX and PyTorch backends.

    Args:
        normalize: If True (default), convert to Title Case (left_hip -> Left Hip)
                   to match normalized_joint_name_dictionary convention used elsewhere.
                   If False, return original naming (lowercase with underscores).
    """
    if normalize:
        return [name.replace("_", " ").title() for name in MHR70_KEYPOINT_NAMES]
    else:
        return list(MHR70_KEYPOINT_NAMES)


# TODO: investigate name discrepancies vs normalized_joint_name_dictionary["bml_movi_87"]:
# - Formatting: indices 21,23,52,54 and 71-86 use abbreviated forms (LHeel, LAnkle, etc.)
#   vs full names (Left Heel, Left Ankle, etc.) in bml_movi_87.
# - Semantic: index 69 "CHip" vs bml "Pelvis" (likely synonym; central hip = pelvis centre).
# - Semantic: index 70 "Neck" vs bml "Sternum" — anatomically distinct; needs verification
#   before these lists can be unified.
SAM_VERTEX_MOVI_NAMES = [
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

SAM_KINEMATIC_NODE_NAMES = ['body_world',
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


# ---------------------------------------------------------------------------
# MHR projection / marker utilities (pure NumPy — backend-agnostic)
# Copied from sam3d_body_eqx.mhr.mhr_utils so these work without the JAX
# package installed.
# ---------------------------------------------------------------------------

def project_to_2d_mhr_batched(
    points_3d: np.ndarray,    # (T, 3)
    camera_t: np.ndarray,     # (T, 3)
    focal_length: np.ndarray, # (T,)
    image_size: tuple,
) -> np.ndarray:
    """Project 3D points to 2D using MHR weak-perspective camera model.

    Args:
        points_3d:      3D keypoint for one keypoint type in MHR camera frame per frame (T, 3).
        camera_t:       Camera translation (T, 3) per frame
        focal_length:   Focal length per frame (T,)
        image_size:     (height, width) of the image.

    Returns:
        points_2d:      Pixel coordinates (T, 2) in (x, y) / (col, row) order.
    """
    h, w = image_size

    points_cam = points_3d + camera_t                          # (T, 3)
    z = points_cam[:, 2:3] + 1e-8                              # (T, 1)
    f = focal_length[:, None]                                  # (T, 1)

    points_2d = f * points_cam[:, :2] / z                      # (T, 2)
    points_2d = points_2d + np.array([w / 2.0, h / 2.0])       # (T, 2)

    return points_2d


def load_mhr_mapping(name: str = "final", data_dir=None) -> dict:
    """Load an MHR keypoint-to-vertex mapping from a JSON file.

    Args:
        name: Mapping name ('with_kinematic', 'ideal_biomech_sites', etc.).
        data_dir: Path to the directory containing the mapping JSON files.
                  If None, defaults to the 'mhr/data' folder next to this file.

    Returns:
        The mapping dictionary.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "mhr" / "data"
    data_dir = Path(data_dir)

    file_path = data_dir / f"kp_vertex_mapping_{name}.json"

    if not file_path.exists():
        raise FileNotFoundError(f"MHR mapping file not found: {file_path}")

    with open(file_path, "r") as f:
        return json.load(f)


def extract_markers(
    mapping: dict,
    vertices: np.ndarray,
    keypoints_3d: np.ndarray,
    kinematic_nodes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract virtual markers from MHR predictions.

    Args:
        mapping:        MHR mapping dict (from load_mhr_mapping).
        vertices:       MHR mesh vertices       (T, V, 3)
        keypoints_3d:   MHR 3D keypoints        (T, K, 3)
        kinematic_nodes: Optional kinematic tree joint positions (T, 127, 3).
                         Required only when any marker uses match_type='kinematic_node'.

    Returns:
        Virtual markers (T, K, 3) in the same frame as the inputs.
    """
    markers = []
    for name, m in mapping.items():
        if not isinstance(m, dict):
            continue
        match_type = m.get("match_type", "joint")

        if match_type == "vertex":
            markers.append(vertices[..., m["index"], :])            # (T,3)

        elif match_type == "kinematic_node":
            if kinematic_nodes is None:
                raise ValueError(
                    f"Marker '{name}' requires 'kinematic_nodes' but none was provided."
                )
            markers.append(kinematic_nodes[..., m["index"], :])     # (T,3)

        elif match_type == "sam3d_kp":
            markers.append(keypoints_3d[..., m["index"], :])        # (T,3)

        else:
            raise ValueError(
                f"Marker '{name}' has unrecognized match_type='{match_type}'. "
                f"Accepted types are: 'vertex', 'kinematic_node', 'sam3d_kp'."
            )

    return np.stack(markers, axis=-2)  # (T, K, 3)


def extract_markers_2d(
    mapping: dict,
    vertices: np.ndarray,
    keypoints_2d: np.ndarray,
    camera_t: np.ndarray,
    focal_length,
    image_size: tuple,
    kinematic_nodes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract 2D markers handling both single-frame and batched inputs.

    Args:
        mapping:        MHR mapping dict (from load_mhr_mapping).
        vertices:       MHR mesh vertices        (V,3) or (T, V, 3).
        keypoints_2d:   MHR 2D keypoints         (K, 2) or (T, K, 2).
        camera_t:       Camera translation        (3,) or (T, 3).
        focal_length:   Focal length scalar or array (scalar or (T,)).
        image_size:     (height, width) of the image.
        kinematic_nodes: Optional kinematic tree joint positions (127, 3) or (T, 127, 3).
                         Required only when any marker uses match_type='kinematic_node'.

    Returns:
        markers_2d: (K, 2) or (T, K, 2) matching the input dimensionality.
    """
    batched = vertices.ndim == 3
    if not batched:
        vertices = vertices[None]           # (1, V, 3)
        keypoints_2d = keypoints_2d[None]   # (1, K, 2)
        camera_t = camera_t[None]           # (1, 3)
        focal_length = np.atleast_1d(focal_length)      # (1,)

    markers = []
    for name, m in mapping.items():
        if not isinstance(m, dict):
            continue
        match_type = m.get("match_type", "joint")

        if match_type == "vertex":
            pts_3d = vertices[:, m["index"], :]                    # (T, 3)
            marker = project_to_2d_mhr_batched(
                pts_3d, camera_t, focal_length, image_size)        # (T, 2)

        elif match_type == "kinematic_node":
            if kinematic_nodes is None:
                raise ValueError(
                    f"Marker '{name}' uses match_type='kinematic_node' but 'kinematic_nodes' was not provided."
                )
            pts_3d = kinematic_nodes[:, m["index"], :]              # (T, 3)
            marker = project_to_2d_mhr_batched(
                pts_3d, camera_t, focal_length, image_size)         # (T, 2)

        elif match_type == "sam3d_kp":
            marker = keypoints_2d[:, m["index"], :]                 # (T, 2)

        else:
            raise ValueError(
                f"Marker '{name}' has unrecognized match_type='{match_type}'. "
                f"Accepted types are: 'vertex', 'kinematic_node', 'sam3d_kp'."
            )

        markers.append(marker)

    result = np.stack(markers, axis=1)                               # (T, K, 2)
    return result[0] if not batched else result                      # (K, 2) or (T, K, 2)


# ---------------------------------------------------------------------------
# High-level fetch helpers — accept a DataJoint SAM3DBody restriction and
# return keypoints-with-confidence arrays ready for pipeline insertion.
# ---------------------------------------------------------------------------

def _project_all_keypoints_2d(
    points_3d: np.ndarray,    # (T, K, 3)
    camera_t: np.ndarray,     # (T, 3)
    focal_length: np.ndarray, # (T,)
    image_size: tuple,
) -> np.ndarray:              # (T, K, 2)
    """Vectorised pinhole projection for a full keypoint set."""
    h, w = image_size
    points_cam = points_3d + camera_t[:, None, :]          # (T, K, 3)
    z = points_cam[:, :, 2:3] + 1e-8                       # (T, K, 1)
    f = focal_length[:, None, None]                         # (T, 1, 1)
    points_2d = f * points_cam[:, :, :2] / z               # (T, K, 2)
    return points_2d + np.array([w / 2.0, h / 2.0])        # (T, K, 2)


def _with_nan_confidence(points: np.ndarray) -> np.ndarray:
    """Append a NaN-based confidence channel (1.0 = all coords finite)."""
    conf = (~np.isnan(points).any(axis=-1, keepdims=True)).astype(float)
    return np.concatenate([points, conf], axis=-1)



def fetch_sam3d_joints_2d(sam3d_entry, image_size: tuple) -> np.ndarray:
    """Project MHR 70 keypoints to 2D pixel coordinates.

    Returns:
        (T, 70, 3) — (x, y, confidence)
    """
    assert len(sam3d_entry) == 1, f"Expected exactly one SAM3DBody entry, got {len(sam3d_entry)}. Only one sam3d_method should be populated per video."
    geom = sam3d_entry.fetch_geometry(return_vertices=False, return_joints=False)
    camera_t, focal_length = sam3d_entry.fetch1("camera_t", "focal_length")
    kp2d = _project_all_keypoints_2d(geom["keypoints_3d"], camera_t, focal_length, image_size)
    return _with_nan_confidence(kp2d)


def fetch_sam3d_movi87_2d(sam3d_entry, image_size: tuple) -> np.ndarray:
    """Project MoVi-87 virtual markers to 2D pixel coordinates.

    Returns:
        (T, 87, 3) — (x, y, confidence)
    """
    assert len(sam3d_entry) == 1, f"Expected exactly one SAM3DBody entry, got {len(sam3d_entry)}. Only one sam3d_method should be populated per video."
    mapping = load_mhr_mapping("with_kinematic")
    geom = sam3d_entry.fetch_geometry(return_vertices=True, return_joints=True)
    camera_t, focal_length = sam3d_entry.fetch1("camera_t", "focal_length")
    kp2d = _project_all_keypoints_2d(geom["keypoints_3d"], camera_t, focal_length, image_size)
    markers_2d = extract_markers_2d(
        mapping, geom["vertices"], kp2d, camera_t, focal_length, image_size, geom["joints"]
    )
    return _with_nan_confidence(markers_2d)


def fetch_sam3d_ideal_2d(sam3d_entry, image_size: tuple) -> np.ndarray:
    """Project ideal biomechanical site markers to 2D pixel coordinates.

    Returns:
        (T, N, 3) — (x, y, confidence)
    """
    assert len(sam3d_entry) == 1, f"Expected exactly one SAM3DBody entry, got {len(sam3d_entry)}. Only one sam3d_method should be populated per video."
    mapping = load_mhr_mapping("ideal_biomech_sites")
    geom = sam3d_entry.fetch_geometry(return_vertices=True, return_joints=True)
    camera_t, focal_length = sam3d_entry.fetch1("camera_t", "focal_length")
    kp2d = _project_all_keypoints_2d(geom["keypoints_3d"], camera_t, focal_length, image_size)
    markers_2d = extract_markers_2d(
        mapping, geom["vertices"], kp2d, camera_t, focal_length, image_size, geom["joints"]
    )
    return _with_nan_confidence(markers_2d)


def fetch_sam3d_kinematic_nodes_2d(sam3d_entry, image_size: tuple) -> np.ndarray:
    """Project 127 kinematic tree nodes to 2D pixel coordinates.

    Returns:
        (T, 127, 3) — (x, y, confidence)
    """
    assert len(sam3d_entry) == 1, f"Expected exactly one SAM3DBody entry, got {len(sam3d_entry)}. Only one sam3d_method should be populated per video."
    geom = sam3d_entry.fetch_geometry(return_vertices=False, return_joints=True)
    camera_t, focal_length = sam3d_entry.fetch1("camera_t", "focal_length")
    kp2d = _project_all_keypoints_2d(geom["joints"], camera_t, focal_length, image_size)
    return _with_nan_confidence(kp2d)


def fetch_sam3d_joints_3d(sam3d_entry) -> np.ndarray:
    """Fetch MHR 70 keypoints in mm with NaN-based confidence.

    Returns:
        (T, 70, 4) — (x, y, z, confidence) in mm
    """
    assert len(sam3d_entry) == 1, f"Expected exactly one SAM3DBody entry, got {len(sam3d_entry)}. Only one sam3d_method should be populated per video."
    kp3d = sam3d_entry.fetch_geometry(return_vertices=False, return_joints=False)["keypoints_3d"]
    return _with_nan_confidence(kp3d * 1000)


def fetch_sam3d_movi87_3d(sam3d_entry) -> np.ndarray:
    """Fetch MoVi-87 virtual markers in mm with NaN-based confidence.

    Returns:
        (T, 87, 4) — (x, y, z, confidence) in mm
    """
    assert len(sam3d_entry) == 1, f"Expected exactly one SAM3DBody entry, got {len(sam3d_entry)}. Only one sam3d_method should be populated per video."
    mapping = load_mhr_mapping("with_kinematic")
    geom = sam3d_entry.fetch_geometry(return_vertices=True, return_joints=True)
    markers = extract_markers(mapping, geom["vertices"], geom["keypoints_3d"], geom["joints"])
    return _with_nan_confidence(markers * 1000)


def fetch_sam3d_ideal_3d(sam3d_entry) -> np.ndarray:
    """Fetch ideal biomechanical site markers in mm with NaN-based confidence.

    Returns:
        (T, N, 4) — (x, y, z, confidence) in mm
    """
    assert len(sam3d_entry) == 1, f"Expected exactly one SAM3DBody entry, got {len(sam3d_entry)}. Only one sam3d_method should be populated per video."
    mapping = load_mhr_mapping("ideal_biomech_sites")
    geom = sam3d_entry.fetch_geometry(return_vertices=True, return_joints=True)
    markers = extract_markers(mapping, geom["vertices"], geom["keypoints_3d"], geom["joints"])
    return _with_nan_confidence(markers * 1000)


def fetch_sam3d_kinematic_nodes_3d(sam3d_entry) -> np.ndarray:
    """Fetch 127 kinematic tree nodes in mm with NaN-based confidence.

    Returns:
        (T, 127, 4) — (x, y, z, confidence) in mm
    """
    assert len(sam3d_entry) == 1, f"Expected exactly one SAM3DBody entry, got {len(sam3d_entry)}. Only one sam3d_method should be populated per video."
    kp3d = sam3d_entry.fetch_geometry(return_vertices=False, return_joints=True)["joints"]
    return _with_nan_confidence(kp3d * 1000)


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
        method_name: Backend/variant selection:
            "jax"          — JAX backend, no hand refinement (default).
            "jax_hands"    — JAX backend, with hand refinement (batch_size=4).
            "jax_hands2"   — JAX backend, with hand refinement (batch_size=16).
            "torch_dinov3" — PyTorch backend (facebook/sam-3d-body-dinov3).

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


# ---------------------------------------------------------------------------
# Self-occlusion helpers (pure numpy + pyrender/trimesh, no JAX dependency)
#
# Coordinate conventions:
#   Sam3d camera frame : X=right, Y=down,  Z=forward  (OpenCV)
#   Pyrender frame     : X=right, Y=up,    Z=back      (OpenGL)
#   Transform          : negate Y and Z when going Sam3d → Pyrender.
# ---------------------------------------------------------------------------

def _project_pinhole(
    pts_3d: np.ndarray,
    focal_length: float,
    image_size: Tuple[int, int],
) -> np.ndarray:
    """Pinhole projection (no distortion). Returns (K, 2) pixel coords."""
    H, W = image_size
    cx, cy = W / 2.0, H / 2.0
    z = pts_3d[:, 2]
    safe_z = np.where(z > 0, z, 1e-6)
    x_px = focal_length * pts_3d[:, 0] / safe_z + cx
    y_px = focal_length * pts_3d[:, 1] / safe_z + cy
    return np.stack([x_px, y_px], axis=-1)


def _render_depth_buffer(
    vertices_cam: np.ndarray,
    faces: np.ndarray,
    focal_length: float,
    image_size: Tuple[int, int],
) -> np.ndarray:
    """Render mesh depth buffer via pyrender (off-screen). Returns (H, W) float32."""
    import pyrender
    import trimesh

    H, W = image_size
    cx, cy = W / 2.0, H / 2.0

    verts_render = vertices_cam.copy()
    verts_render[:, 1] *= -1
    verts_render[:, 2] *= -1

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
    mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts_render, faces.copy()))
    scene.add(mesh)
    camera = pyrender.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=cx, cy=cy,
        znear=0.01, zfar=20.0,
    )
    scene.add(camera, pose=np.eye(4))

    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    try:
        _, depth = renderer.render(scene)
    finally:
        renderer.delete()

    return depth


def _compute_point_visibility(
    pts_3d_cam: np.ndarray,
    pts_2d: np.ndarray,
    depth_buffer: np.ndarray,
    depth_tolerance: float = 0.05,
) -> np.ndarray:
    """Vectorized depth test. Returns (K,) bool — True = visible."""
    H, W = depth_buffer.shape
    pt_depth = pts_3d_cam[:, 2]
    px = np.clip(np.round(pts_2d[:, 0]).astype(np.int32), 0, W - 1)
    py = np.clip(np.round(pts_2d[:, 1]).astype(np.int32), 0, H - 1)
    in_bounds = (
        (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < W) &
        (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < H)
    )
    buf_depth = depth_buffer[py, px]
    depth_ok = (buf_depth == 0) | (pt_depth <= buf_depth + depth_tolerance)
    return in_bounds & depth_ok


def _compute_visibility_batch(
    vertices_cam: np.ndarray,
    faces: np.ndarray,
    focal_length: float,
    image_size: Tuple[int, int],
    keypoints_3d_cam: np.ndarray,
    keypoints_2d: np.ndarray,
    joints_3d_cam: Optional[np.ndarray] = None,
    joints_2d: Optional[np.ndarray] = None,
    depth_tolerance: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Render depth buffer once, test keypoints/joints/vertices against it.

    Returns dict with 'keypoints_visibility' (Kp,), optionally 'joints_visibility'
    (Kj,), and 'vertices_visibility' (V,) boolean arrays.
    """
    depth_buffer = _render_depth_buffer(vertices_cam, faces, focal_length, image_size)
    out: Dict[str, np.ndarray] = {}
    out["keypoints_visibility"] = _compute_point_visibility(
        keypoints_3d_cam, keypoints_2d, depth_buffer, depth_tolerance
    )
    if joints_3d_cam is not None and joints_2d is not None:
        out["joints_visibility"] = _compute_point_visibility(
            joints_3d_cam, joints_2d, depth_buffer, depth_tolerance
        )
    verts_2d = _project_pinhole(vertices_cam, focal_length, image_size)
    out["vertices_visibility"] = _compute_point_visibility(
        vertices_cam, verts_2d, depth_buffer, depth_tolerance
    )
    return out


@lru_cache(maxsize=2)
def _load_sam3d_torch_model(repo_id: str = "facebook/sam-3d-body-dinov3", device: str = "cuda"):
    """Load (and cache) just the SAM3D-Body torch model, for geometry reconstruction/rendering."""
    from sam_3d_body import load_sam_3d_body
    from sam_3d_body.build_models import _hf_download

    ckpt_path, mhr_path = _hf_download(repo_id)
    model, _ = load_sam_3d_body(checkpoint_path=ckpt_path, mhr_path=mhr_path, device=device)
    return model


def compute_sam3d_geometry_jax(
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
    """Reconstruct MHR mesh vertices and kinematic joints using the JAX/Equinox backend.

    Delegates to SAM3DBodyEstimator.compute_geometry() (requires sam3d_body_eqx).
    Vertices/joints are returned in body/root space. Apply camera_t separately for
    2D projection.

    When camera_t, focal_length, and image_size are provided, self-occlusion
    visibility masks are also returned (requires pyrender + trimesh).

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


def compute_sam3d_geometry_torch(
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
    device: str = "cuda",
    repo_id: str = "facebook/sam-3d-body-dinov3",
) -> Dict[str, np.ndarray]:
    """Reconstruct MHR mesh vertices and kinematic joints using the PyTorch backend.

    Runs the stored minimal parameters back through MHRHead.mhr_forward() (decoupled
    from the image pipeline — requires sam_3d_body). Vertices/joints are returned in
    body/root space, matching compute_sam3d_geometry_jax's convention; apply camera_t
    separately for 2D projection.

    When camera_t, focal_length, and image_size are provided, self-occlusion visibility
    masks are computed via pyrender depth-buffer testing (requires pyrender + trimesh).

    Returns:
        dict with keys: 'keypoints_3d' (N, 70, 3) always; 'vertices' (N, 18439, 3) and/or
        'joints' (N, 127, 3) when requested; 'keypoints_visibility' (N, 70),
        'joints_visibility' (N, 127), and 'vertices_visibility' (N, 18439) bool arrays
        when camera params are provided.
    """
    import torch

    model = _load_sam3d_torch_model(repo_id=repo_id, device=device)

    compute_vis = (
        camera_t is not None
        and focal_length is not None
        and image_size is not None
    )
    need_vertices = return_vertices or compute_vis

    n = len(body_pose_params)
    to_tensor = lambda arr: torch.as_tensor(np.asarray(arr), dtype=torch.float32, device=device)

    with torch.no_grad():
        verts, keypoints, joints = model.head_pose.mhr_forward(
            global_trans=torch.zeros((n, 3), dtype=torch.float32, device=device),
            global_rot=to_tensor(global_rot),
            body_pose_params=to_tensor(body_pose_params),
            hand_pose_params=to_tensor(hand_pose_params),
            scale_params=to_tensor(scale_params),
            shape_params=to_tensor(shape_params),
            expr_params=torch.zeros((n, model.head_pose.num_face_comps), dtype=torch.float32, device=device),
            return_keypoints=True,
            return_joint_coords=True,
        )

    vertices_all = verts.cpu().numpy() if need_vertices else None
    keypoints_3d = keypoints[:, :70].cpu().numpy()
    joints_all = joints.cpu().numpy()

    result = {"keypoints_3d": keypoints_3d}
    if return_vertices:
        result["vertices"] = vertices_all
    if return_joints:
        result["joints"] = joints_all

    if compute_vis:
        camera_t = np.asarray(camera_t)
        focal_length = np.asarray(focal_length)
        faces = model.head_pose.faces.cpu().numpy()

        kp_vis = np.zeros((n, keypoints_3d.shape[1]), dtype=bool)
        jt_vis = np.zeros((n, joints_all.shape[1]), dtype=bool) if return_joints else None
        vt_vis = np.zeros((n, vertices_all.shape[1]), dtype=bool)

        for i in range(n):
            if np.any(np.isnan(camera_t[i])) or np.isnan(focal_length[i]):
                continue

            cam_t_i = camera_t[i]
            fl_i = float(focal_length[i])

            kp3d_cam = keypoints_3d[i] + cam_t_i
            verts_cam = vertices_all[i] + cam_t_i
            j3d_cam = (joints_all[i] + cam_t_i) if return_joints else None

            kp2d = _project_pinhole(kp3d_cam, fl_i, image_size)
            j2d = _project_pinhole(j3d_cam, fl_i, image_size) if j3d_cam is not None else None

            vis = _compute_visibility_batch(
                vertices_cam=verts_cam,
                faces=faces,
                focal_length=fl_i,
                image_size=image_size,
                keypoints_3d_cam=kp3d_cam,
                keypoints_2d=kp2d,
                joints_3d_cam=j3d_cam,
                joints_2d=j2d,
                depth_tolerance=depth_tolerance,
            )
            kp_vis[i] = vis["keypoints_visibility"]
            if jt_vis is not None:
                jt_vis[i] = vis["joints_visibility"]
            vt_vis[i] = vis["vertices_visibility"]

        result["keypoints_visibility"] = kp_vis
        if jt_vis is not None:
            result["joints_visibility"] = jt_vis
        if return_vertices:
            result["vertices_visibility"] = vt_vis

    # Normalize to Y-down convention (matching JAX backend) so all downstream
    # consumers project correctly without backend-specific handling.
    for k in ("keypoints_3d", "vertices", "joints"):
        if result.get(k) is not None:
            result[k] = result[k].copy()
            result[k][..., 1] *= -1
            result[k][..., 2] *= -1

    return result


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
    method_name: str = "jax",
) -> Dict[str, np.ndarray]:
    """Reconstruct MHR mesh vertices and kinematic joints from stored minimal parameters.

    Dispatches to the JAX/Equinox or PyTorch backend based on `method_name`, mirroring
    process_sam3d_body's dispatch — so geometry is reconstructed with the same family of
    model that produced the stored parameters (both decode the same MHR rig/checkpoint,
    so either backend can in principle reconstruct either's output, but using the native
    backend avoids requiring both JAX and PyTorch to be installed).

    Vertices/joints are returned in body/root space. Apply camera_t separately for
    2D projection.

    Args:
        body_pose_params:  (N, 133) body pose Euler angles (XYZ).
        shape_params:      (N, 45) shape PCA coefficients.
        scale_params:      (N, 28) scale PCA coefficients.
        hand_pose_params:  (N, 108) hand pose: columns 0:54 left, 54:108 right (continuous).
        global_rot:        (N, 3) global rotation (ZYX Euler).
        camera_t:          (N, 3) camera translations — used for visibility.
        focal_length:      (N,) focal lengths in pixels — used for visibility.
        image_size:        (H, W) image dimensions — used for visibility.
        depth_tolerance:   Occlusion tolerance in metres (default 5 cm).
        return_vertices:   Whether to include mesh vertices (N, 18439, 3) in output.
        return_joints:     Whether to include kinematic tree joints (N, 127, 3) in output.
        method_name:       Which backend produced (and should reconstruct) the geometry:
                           "jax"/"jax_hands"/"jax_hands2" -> sam3d_body_eqx, "torch_dinov3" -> sam_3d_body.

    Returns:
        dict with keys: 'keypoints_3d' always; 'vertices' and/or 'joints' when requested;
        'keypoints_visibility', 'joints_visibility', 'vertices_visibility' when camera
        params are provided (both backends).
    """
    impl = compute_sam3d_geometry_torch if method_name == "torch_dinov3" else compute_sam3d_geometry_jax
    return impl(
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


def _get_sam3d_method_name(sam3d_entry) -> str:
    from pose_pipeline import SAM3DBodyMethodLookup

    return (SAM3DBodyMethodLookup & sam3d_entry).fetch1("sam3d_method_name")


def get_sam3d_callback(key: Dict[str, Any], mesh_color: Tuple[float, float, float] = (0.65, 0.74, 0.86)):
    """
    Create a visualization callback for rendering SAM3D mesh overlays.

    Dispatches to the JAX or PyTorch renderer depending on which backend produced the
    entry, so that e.g. torch_dinov3 results can be visualized without sam3d_body_eqx.

    Args:
        key: DataJoint key to fetch results.
        mesh_color: Normalized RGB tuple (0.0 to 1.0).

    Returns:
        A function: overlay(image, frame_index) -> visualized_image.
    """
    from pose_pipeline import SAM3DBody

    sam3d_entry = SAM3DBody & key
    method_name = _get_sam3d_method_name(sam3d_entry)
    camera_t, focal_length, frame_valid = sam3d_entry.fetch1("camera_t", "focal_length", "frame_valid")
    geom = sam3d_entry.fetch_geometry(return_vertices=True, return_joints=False)
    vertices = geom["vertices"]
    valid = frame_valid if frame_valid is not None else np.ones(len(vertices), dtype=bool)

    if method_name == "torch_dinov3":
        from sam_3d_body.visualization.renderer import Renderer

        model = _load_sam3d_torch_model()
        faces = model.head_pose.faces.cpu().numpy()
        renderer = Renderer(focal_length=1000.0, faces=faces)

        def render(image, verts, cam_t, fl):
            renderer.focal_length = fl
            out = renderer(verts, cam_t, image, mesh_base_color=mesh_color)
            return (np.clip(out, 0, 255) if out.dtype == np.uint8 else np.clip(out * 255, 0, 255).astype(np.uint8))
    else:
        from sam3d_body_eqx.inference import SAM3DBodyEstimator
        from sam3d_body_eqx.visualization.mesh import render_mesh

        estimator = SAM3DBodyEstimator.from_pretrained()
        faces = np.asarray(estimator.model.head_pose.mhr.faces)

        def render(image, verts, cam_t, fl):
            return render_mesh(
                image=image,
                vertices=verts,
                faces=faces,
                camera_translation=cam_t,
                focal_length=fl,
                mesh_color=mesh_color,
            )

    def overlay(image, idx):
        if not valid[idx] or np.any(np.isnan(vertices[idx])):
            return image

        fl = focal_length[idx] if not np.isnan(focal_length[idx]) else 1000.0
        return render(image, vertices[idx], camera_t[idx], fl)

    return overlay