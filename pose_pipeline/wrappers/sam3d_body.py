# SAM-3D-Body wrapper for PosePipeline
# Meta's state-of-the-art single-image full-body 3D human mesh recovery model
# Uses MHR (Momentum Human Rig) representation instead of SMPL
#
# Installation: sam-3d-body must be installed and importable
#   git clone https://github.com/facebookresearch/sam-3d-body.git
#   pip install -e sam-3d-body/  # or add to PYTHONPATH

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

# Global model cache to avoid reloading
_sam3d_model_cache = {}


def load_sam3d_body(repo_id="facebook/sam-3d-body-dinov3", device="cuda"):
    """
    Load SAM-3D-Body model from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID
        device: torch device ('cuda' or 'cpu')

    Returns:
        SAM3DBodyEstimator instance
    """
    cache_key = (repo_id, device)
    if cache_key in _sam3d_model_cache:
        return _sam3d_model_cache[cache_key]

    # Use lower-level functions to properly pass device parameter
    # (load_sam_3d_body_hf doesn't forward kwargs to load_sam_3d_body)
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    from sam_3d_body.build_models import _hf_download

    print(f"Loading SAM-3D-Body model from {repo_id}...")

    # Download checkpoint from HuggingFace
    ckpt_path, mhr_path = _hf_download(repo_id)

    # Load model with explicit device
    model, model_cfg = load_sam_3d_body(
        checkpoint_path=ckpt_path,
        mhr_path=mhr_path,
        device=device
    )

    # Create estimator without detector (we'll provide bboxes)
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,  # We use pre-computed bboxes from PosePipeline
        human_segmentor=None,
        fov_estimator=None,
    )

    _sam3d_model_cache[cache_key] = estimator
    return estimator


def process_sam3d_body(key, repo_id="facebook/sam-3d-body-dinov3"):
    """
    Process video with SAM-3D-Body model.

    Uses pre-computed bounding boxes from PosePipeline's tracking stage.
    SAM-3D-Body expects full images with bboxes and handles cropping internally.
    Outputs MHR (Momentum Human Rig) format natively.

    Args:
        key: DataJoint key with video/person references
        repo_id: HuggingFace repository ID for the model

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
    from pose_pipeline import Video, VideoInfo, PersonBbox

    # Get video and bounding boxes
    video_path, bboxes_dj, present_dj = (Video * PersonBbox & key).fetch1("video", "bbox", "present")
    num_frames, height, width = (VideoInfo & key).fetch1('num_frames', 'height', 'width')

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    estimator = load_sam3d_body(repo_id=repo_id, device=device)

    # Get mesh faces (same for all frames)
    mesh_faces = estimator.faces

    # Initialize result accumulators
    results = {
        'vertices': [],
        'keypoints_3d': [],
        'keypoints_2d': [],
        'camera_t': [],
        'focal_length': [],
        'body_pose_params': [],
        'hand_pose_params': [],
        'shape_params': [],
        'global_rot': [],
        'frame_valid': [],
    }

    # Open video
    cap = cv2.VideoCapture(video_path)

    for idx in tqdm(range(num_frames), desc="Processing SAM-3D-Body"):
        ret, frame = cap.read()
        if not ret:
            break

        # Check if person is present in this frame
        if not present_dj[idx]:
            # Mark as invalid and append placeholders
            results['frame_valid'].append(False)
            results['vertices'].append(None)
            results['keypoints_3d'].append(None)
            results['keypoints_2d'].append(None)
            results['camera_t'].append(None)
            results['focal_length'].append(None)
            results['body_pose_params'].append(None)
            results['hand_pose_params'].append(None)
            results['shape_params'].append(None)
            results['global_rot'].append(None)
            continue

        # Get bounding box for this frame (TLHW format from PosePipeline)
        bbox_tlhw = bboxes_dj[idx]

        # Convert TLHW to TLBR (x1, y1, x2, y2) format for SAM-3D-Body
        bbox_tlbr = np.array([
            bbox_tlhw[0],  # x1 (left)
            bbox_tlhw[1],  # y1 (top)
            bbox_tlhw[0] + bbox_tlhw[2],  # x2 (right)
            bbox_tlhw[1] + bbox_tlhw[3],  # y2 (bottom)
        ]).reshape(1, 4)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # Run inference with pre-computed bbox
            outputs = estimator.process_one_image(
                frame_rgb,
                bboxes=bbox_tlbr,
                bbox_thr=0.5,
                use_mask=False,
                inference_type="full",  # Full body + hands
            )

            if len(outputs) > 0:
                person = outputs[0]  # Take first (and should be only) person

                results['frame_valid'].append(True)
                results['vertices'].append(person['pred_vertices'])
                results['keypoints_3d'].append(person['pred_keypoints_3d'])
                results['keypoints_2d'].append(person['pred_keypoints_2d'])
                results['camera_t'].append(person['pred_cam_t'])
                results['focal_length'].append(person['focal_length'])
                results['body_pose_params'].append(person['body_pose_params'])
                results['hand_pose_params'].append(person['hand_pose_params'])
                results['shape_params'].append(person['shape_params'])
                results['global_rot'].append(person['global_rot'])
            else:
                # No detection in this frame
                results['frame_valid'].append(False)
                results['vertices'].append(None)
                results['keypoints_3d'].append(None)
                results['keypoints_2d'].append(None)
                results['camera_t'].append(None)
                results['focal_length'].append(None)
                results['body_pose_params'].append(None)
                results['hand_pose_params'].append(None)
                results['shape_params'].append(None)
                results['global_rot'].append(None)

        except Exception as e:
            print(f"Error processing frame {idx}: {e}")
            results['frame_valid'].append(False)
            results['vertices'].append(None)
            results['keypoints_3d'].append(None)
            results['keypoints_2d'].append(None)
            results['camera_t'].append(None)
            results['focal_length'].append(None)
            results['body_pose_params'].append(None)
            results['hand_pose_params'].append(None)
            results['shape_params'].append(None)
            results['global_rot'].append(None)

    cap.release()

    # Clean up video file (DataJoint fetches to temp file)
    try:
        os.remove(video_path)
    except OSError:
        pass  # File may already be removed or inaccessible

    # Convert lists to arrays, handling None values
    def stack_with_nans(data_list, shape_if_none=None):
        """Stack list into array, using NaN for None values."""
        if all(x is None for x in data_list):
            return None

        # Find first non-None to get shape
        for x in data_list:
            if x is not None:
                shape_if_none = x.shape
                break

        if shape_if_none is None:
            return None

        result = np.full((len(data_list), *shape_if_none), np.nan)
        for i, x in enumerate(data_list):
            if x is not None:
                result[i] = x
        return result

    # Stack results
    final_results = {
        'vertices': stack_with_nans(results['vertices']),
        'keypoints_3d': stack_with_nans(results['keypoints_3d']),
        'keypoints_2d': stack_with_nans(results['keypoints_2d']),
        'camera_t': stack_with_nans(results['camera_t']),
        'focal_length': stack_with_nans(results['focal_length']),
        'body_pose_params': stack_with_nans(results['body_pose_params']),
        'hand_pose_params': stack_with_nans(results['hand_pose_params']),
        'shape_params': stack_with_nans(results['shape_params']),
        'global_rot': stack_with_nans(results['global_rot']),
        'mesh_faces': mesh_faces,
        'frame_valid': np.array(results['frame_valid']),
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
    vertices = data['vertices']
    faces = data['mesh_faces']
    camera_t = data['camera_t']
    focal_length = data['focal_length']
    frame_valid = data.get('frame_valid', np.ones(len(vertices), dtype=bool))

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
