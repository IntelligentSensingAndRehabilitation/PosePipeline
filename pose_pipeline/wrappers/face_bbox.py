import numpy as np
import cv2
from typing import Tuple
import datajoint as dj
import os

from pose_pipeline.pipeline import FaceBbox


def check_mmpose_available():
    """Check if MMPose and MMDetection are available"""
    try:
        import mmpose
        import mmdet
        import mmcv
        return True, None
    except ImportError as e:
        return False, str(e)


def mmpose_face_det(key: dict, method: str = "RTMPose_Face") -> Tuple[int, np.ndarray]:
    """
    Detect faces using RTMPose face detector
    
    Args:
        key: DataJoint key containing video information
        method: Detection method (default "RTMPose_Face")
        
    Returns:
        Tuple of (num_faces, face_bboxes)
        - num_faces: number of detected faces
        - face_bboxes: array of shape (frames, max_faces, 5) where last dimension is [x1, y1, x2, y2, confidence]
    """
    try:
        from mmpose.apis import init_model, inference_topdown
        from mmdet.apis import init_detector, inference_detector
    except ImportError:
        error_msg = """
❌ MMPose not available: No module named 'mmpose'

Suggested alternatives:
- MediaPipe_Face (installed) - Use estimation_method=1
- TopDown_Wholebody (uses existing pose data) - Use estimation_method=2  
- PersonBbox_Wholebody (uses person detections) - Use detection_method=2

Installation guide: /home/vscode/workspace/MMPOSE_INSTALLATION_GUIDE.md
Quick install: pip install mmpose mmdet mmcv
        """
        raise ImportError(error_msg)
    
    from pose_pipeline.pipeline import Video
    from pose_pipeline.utils.video import get_frames_cv2
    from pose_pipeline.env import MODEL_DATA_DIR

    # Get video path
    video_path = (Video & key).fetch1("video_path")
    
    if method == "RTMPose_Face":
        # RTMPose face detection config and checkpoint
        config_url = "https://raw.githubusercontent.com/open-mmlab/mmpose/dev-1.x/configs/face_2d_keypoint/rtmpose/coco_wholebody_face/rtmpose-m_8xb32-60e_coco-wholebody-face-256x256.py"
        checkpoint_url = "https://download.openmmlab.com/mmpose/v1/face_2d_keypoint/rtmpose/coco-wholebody-face/rtmpose-m_simcc-coco-wholebody-face_pt-aic-coco_60e-256x256-5bda63ce.pth"
        
        # Define local paths
        config_dir = os.path.join(MODEL_DATA_DIR, "mmpose/configs/face_2d_keypoint/rtmpose/coco_wholebody_face/")
        checkpoint_dir = os.path.join(MODEL_DATA_DIR, "mmpose/checkpoints/")
        
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        config_file = os.path.join(config_dir, "rtmpose-m_8xb32-60e_coco-wholebody-face-256x256.py")
        checkpoint_file = os.path.join(checkpoint_dir, "rtmpose-m_simcc-coco-wholebody-face_pt-aic-coco_60e-256x256-5bda63ce.pth")
        
        # Download config if not exists
        if not os.path.exists(config_file):
            print(f"Downloading config from {config_url}")
            os.system(f"wget -O '{config_file}' '{config_url}'")
        
        # Download checkpoint if not exists
        if not os.path.exists(checkpoint_file):
            print(f"Downloading checkpoint from {checkpoint_url}")
            os.system(f"wget -O '{checkpoint_file}' '{checkpoint_url}'")
        
        # Initialize RTMPose face model
        try:
            face_model = init_model(config_file, checkpoint_file, device='cuda:0')
        except:
            face_model = init_model(config_file, checkpoint_file, device='cpu')
        
        # We need a person detector to get rough face regions first
        # Use RTMDet person detector
        person_config_url = "https://raw.githubusercontent.com/open-mmlab/mmdetection/master/configs/rtmdet/rtmdet_m_8xb32-300e_coco.py"
        person_checkpoint_url = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
        
        person_config_dir = os.path.join(MODEL_DATA_DIR, "mmdet/configs/rtmdet/")
        person_checkpoint_dir = os.path.join(MODEL_DATA_DIR, "mmdet/checkpoints/")
        
        os.makedirs(person_config_dir, exist_ok=True)
        os.makedirs(person_checkpoint_dir, exist_ok=True)
        
        person_config_file = os.path.join(person_config_dir, "rtmdet_m_8xb32-300e_coco.py")
        person_checkpoint_file = os.path.join(person_checkpoint_dir, "rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth")
        
        # Download person detector config if not exists
        if not os.path.exists(person_config_file):
            print(f"Downloading person config from {person_config_url}")
            os.system(f"wget -O '{person_config_file}' '{person_config_url}'")
        
        # Download person detector checkpoint if not exists
        if not os.path.exists(person_checkpoint_file):
            print(f"Downloading person checkpoint from {person_checkpoint_url}")
            os.system(f"wget -O '{person_checkpoint_file}' '{person_checkpoint_url}'")
        
        try:
            person_model = init_detector(person_config_file, person_checkpoint_file, device='cuda:0')
        except:
            person_model = init_detector(person_config_file, person_checkpoint_file, device='cpu')
            
    else:
        raise ValueError(f"Unsupported face detection method: {method}")
    
    # Get video frames
    frames = get_frames_cv2(video_path)
    num_frames = len(frames)
    
    face_detections = []
    max_faces = 0
    
    for frame in frames:
        # First detect persons to get approximate head regions
        person_results = inference_detector(person_model, frame)
        
        # Extract person bboxes
        if hasattr(person_results, 'pred_instances'):
            # MMDet 3.x format
            person_bboxes = person_results.pred_instances.bboxes.cpu().numpy()
            person_scores = person_results.pred_instances.scores.cpu().numpy()
            person_labels = person_results.pred_instances.labels.cpu().numpy()
            
            # Filter for person class (class 0 in COCO)
            person_mask = person_labels == 0
            person_bboxes = person_bboxes[person_mask]
            person_scores = person_scores[person_mask]
        else:
            # MMDet 2.x format  
            person_bboxes = person_results[0][:, :4] if len(person_results[0]) > 0 else np.empty((0, 4))
            person_scores = person_results[0][:, 4] if len(person_results[0]) > 0 else np.empty((0,))
        
        frame_face_bboxes = []
        
        # For each detected person, estimate face region and run face pose estimation
        for person_bbox, person_score in zip(person_bboxes, person_scores):
            if person_score > 0.5:  # Confidence threshold
                # Estimate face region from person bbox (top portion of person)
                x1, y1, x2, y2 = person_bbox
                face_y1 = y1
                face_y2 = y1 + (y2 - y1) * 0.3  # Top 30% of person bbox
                face_x1 = x1 + (x2 - x1) * 0.1  # Add some horizontal margin
                face_x2 = x2 - (x2 - x1) * 0.1
                
                # Ensure face bbox is within frame bounds
                h, w = frame.shape[:2]
                face_bbox = [
                    max(0, face_x1),
                    max(0, face_y1), 
                    min(w, face_x2),
                    min(h, face_y2)
                ]
                
                # Run face pose estimation to validate this is actually a face
                try:
                    face_results = inference_topdown(face_model, frame, [face_bbox])
                    
                    if len(face_results) > 0 and hasattr(face_results[0], 'pred_instances'):
                        pred_instances = face_results[0].pred_instances
                        if len(pred_instances.keypoints) > 0:
                            # If face keypoints detected successfully, use this as a face bbox
                            keypoints = pred_instances.keypoints[0]
                            scores = pred_instances.keypoint_scores[0]
                            
                            # Calculate face confidence based on keypoint visibility
                            visible_scores = scores[scores > 0.3]
                            if len(visible_scores) > 10:  # Need reasonable number of visible keypoints
                                face_confidence = np.mean(visible_scores)
                                
                                if face_confidence > 0.5:  # Face detection confidence threshold
                                    frame_face_bboxes.append([face_x1, face_y1, face_x2, face_y2, face_confidence])
                            
                except Exception as e:
                    print(f"Warning: Face pose estimation failed for bbox {face_bbox}: {e}")
                    continue
        
        # Convert to numpy array
        if len(frame_face_bboxes) > 0:
            frame_detections = np.array(frame_face_bboxes)
        else:
            frame_detections = np.empty((0, 5))
        
        face_detections.append(frame_detections)
        max_faces = max(max_faces, len(frame_detections))
    
    # Pad detections to consistent shape
    padded_detections = np.zeros((num_frames, max_faces, 5))
    
    for i, detections in enumerate(face_detections):
        if len(detections) > 0:
            padded_detections[i, :len(detections)] = detections
    
    return max_faces, padded_detections


def make_bbox_from_keypoints(keypoints: np.ndarray, face_indices: list = None) -> np.ndarray:
    """
    Create face bboxes from facial keypoints in wholebody pose estimation
    
    Args:
        keypoints: Array of shape (frames, keypoints, 3) containing pose keypoints
        face_indices: List of keypoint indices that correspond to face landmarks
        
    Returns:
        face_bboxes: Array of shape (frames, 1, 5) containing face bounding boxes
    """
    if face_indices is None:
        # Face landmark indices for MMPose COCOWholeBody format (method 1: MMPoseWholebody)
        # Face keypoints are indices 23-90 in COCOWholeBody format (68 facial landmarks)
        face_indices = list(range(23, 91))
    
    num_frames = keypoints.shape[0]
    face_bboxes = np.zeros((num_frames, 1, 5))
    
    for frame_idx in range(num_frames):
        frame_keypoints = keypoints[frame_idx]
        
        # Extract face keypoints
        face_kpts = frame_keypoints[face_indices]
        valid_kpts = face_kpts[face_kpts[:, 2] > 0.3]  # Filter by confidence
        
        if len(valid_kpts) > 4:  # Need at least 4 points for a reasonable bbox
            x_coords = valid_kpts[:, 0]
            y_coords = valid_kpts[:, 1]
            
            x1, x2 = np.min(x_coords), np.max(x_coords)
            y1, y2 = np.min(y_coords), np.max(y_coords)
            
            # Add some padding
            padding = 0.15
            width = x2 - x1
            height = y2 - y1
            x1 -= width * padding
            x2 += width * padding
            y1 -= height * padding
            y2 += height * padding
            
            # Average confidence of face keypoints
            confidence = np.mean(valid_kpts[:, 2])
            
            face_bboxes[frame_idx, 0] = [x1, y1, x2, y2, confidence]
    
    return face_bboxes


def make_bbox_from_person_bbox(person_bboxes: np.ndarray) -> np.ndarray:
    """
    Create face bboxes from person bounding boxes by estimating head region
    
    Args:
        person_bboxes: Array of shape (frames, max_persons, 5) containing person bboxes
        
    Returns:
        face_bboxes: Array of shape (frames, max_faces, 5) containing face bounding boxes
    """
    num_frames, max_persons, _ = person_bboxes.shape
    face_bboxes = np.zeros((num_frames, max_persons, 5))  # Assume max one face per person
    
    for frame_idx in range(num_frames):
        frame_person_bboxes = person_bboxes[frame_idx]
        
        for person_idx, person_bbox in enumerate(frame_person_bboxes):
            x1, y1, x2, y2, confidence = person_bbox
            
            if confidence > 0.3:  # Only process confident person detections
                # Estimate face region as top portion of person bbox
                face_y1 = y1
                face_y2 = y1 + (y2 - y1) * 0.25  # Top 25% of person bbox
                face_x1 = x1 + (x2 - x1) * 0.15  # Add horizontal margin
                face_x2 = x2 - (x2 - x1) * 0.15
                
                # Use person confidence as face confidence (slightly reduced)
                face_bboxes[frame_idx, person_idx] = [face_x1, face_y1, face_x2, face_y2, confidence * 0.8]
    
    return face_bboxes