import os
import cv2
import numpy as np
import datajoint as dj
from pose_pipeline import Video
from pose_pipeline.pipeline import FaceBbox
from tqdm import tqdm


def mmpose_FPE(key: dict, method: str = "RTMPose_Face") -> np.ndarray:
    """
    Estimate facial landmarks using MMPose
    
    Args:
        key: DataJoint key containing video and bbox information
        method: Pose estimation method
        
    Returns:
        keypoints_2d: Array of shape (frames, 68, 3) containing facial landmarks
    """
    try:
        from mmpose.apis import inference_topdown, init_model
    except ImportError:
        error_msg = """
❌ MMPose not available: No module named 'mmpose'

Suggested alternatives:
- MediaPipe_Face (installed) - Use estimation_method=1
- TopDown_Wholebody (uses existing pose data) - Use estimation_method=2

Installation guide: /home/vscode/workspace/MMPOSE_INSTALLATION_GUIDE.md
Quick install: pip install mmpose mmdet mmcv
        """
        raise ImportError(error_msg)
    
    from pose_pipeline import MODEL_DATA_DIR
    
    # Model configuration based on method - use your existing configs
    if method == "RTMPose_Face":
        # Use the same config/checkpoint you have in face_bbox.py
        pose_model_cfg = os.path.join(
            MODEL_DATA_DIR,
            "mmpose/configs/face_2d_keypoint/rtmpose/coco_wholebody_face/rtmpose-m_8xb32-60e_coco-wholebody-face-256x256.py"
        )
        pose_model_ckpt = os.path.join(
            MODEL_DATA_DIR,
            "mmpose/checkpoints/rtmpose-m_simcc-coco-wholebody-face_pt-aic-coco_60e-256x256-5bda63ce.pth"
        )
    else:
        raise ValueError(f"Unsupported face estimation method: {method}")
    
    # Try GPU first, fallback to CPU
    try:
        model = init_model(pose_model_cfg, pose_model_ckpt, device="cuda:0")
    except:
        model = init_model(pose_model_cfg, pose_model_ckpt, device="cpu")
    
    # Get video and face bboxes
    video = Video.get_robust_reader(key, return_cap=False)
    bboxes = (FaceBbox & key).fetch1("bboxes")
    
    cap = cv2.VideoCapture(video)
    results = []
    
    for bbox in tqdm(bboxes):
        ret, frame = cap.read()
        assert ret and frame is not None
        
        # Filter valid bboxes (confidence > 0)
        valid_bboxes = bbox[bbox[:, 4] > 0]
        
        if len(valid_bboxes) > 0:
            # Use the highest confidence face bbox
            best_bbox_idx = np.argmax(valid_bboxes[:, 4])
            face_bbox = valid_bboxes[best_bbox_idx:best_bbox_idx+1]  # Keep as 2D array
            
            # Run pose estimation
            pose_results = inference_topdown(model, frame, face_bbox)
            
            if pose_results and len(pose_results) > 0:
                # Get prediction instances from mmpose results
                pred_instances = pose_results[0].pred_instances
                keypoints = pred_instances.keypoints
                keypoint_scores = pred_instances.keypoint_scores
                
                # Concatenate keypoints and scores
                keypoints_2d = np.concatenate((keypoints[0, :, :], keypoint_scores.T), axis=-1)
            else:
                # No face detected, return zeros
                keypoints_2d = np.zeros((68, 3))
        else:
            # No valid bboxes, return zeros
            keypoints_2d = np.zeros((68, 3))
        
        results.append(keypoints_2d)
    
    cap.release()
    os.remove(video)
    
    return np.array(results)


def face_estimation_from_wholebody(key: dict, face_indices: list = None) -> np.ndarray:
    """
    Extract facial landmarks from wholebody pose estimation results
    
    Args:
        key: DataJoint key
        face_indices: List of keypoint indices corresponding to face landmarks
        
    Returns:
        keypoints_2d: Array of facial landmarks
    """
    from pose_pipeline.pipeline import TopDownPerson
    
    if face_indices is None:
        # Default face landmark indices for MMPose COCOWholeBody format (method 1: MMPoseWholebody)
        # Face keypoints are indices 23-90 in COCOWholeBody format (68 facial landmarks)
        face_indices = list(range(23, 91))
    
    try:
        # Get wholebody keypoints (using MMPoseWholebody method 1)
        wholebody_keypoints = (TopDownPerson & key & "top_down_method=1").fetch1("keypoints")
        
        # Extract face keypoints
        face_keypoints = wholebody_keypoints[:, face_indices, :]
        
        return face_keypoints
        
    except Exception as e:
        raise Exception(f"Could not extract face keypoints from wholebody estimation (method 1: MMPoseWholebody): {e}")


def overlay_face_keypoints(video, output_file, keypoints, bboxes):
    """Process a video and create overlay of facial keypoints

    Args:
    video (str): filename for source (from key)
    output_file (str): output filename
    keypoints (list): list of list of keypoints
    bboxes (list): list of face bounding boxes
    """
    from pose_pipeline.utils.visualization import draw_keypoints

    # Get video details
    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_size = (int(w), int(h))

    # set writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, output_size)
    
    # process every frame
    for frame_idx in tqdm(range(total_frames)):
        success, frame = cap.read()
        if not success:
            break
        
        keypoints_2d = keypoints[frame_idx][:, :]
        frame = draw_keypoints(frame, keypoints_2d, threshold=0.2, color=(0, 255, 255))  # Yellow for face
        
        # Draw face bounding boxes
        for bbox in bboxes[frame_idx]:
            if bbox[4] > 0:  # Only draw if confidence > 0
                frame = cv2.rectangle(
                    frame,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (255, 0, 0),  # Blue color for face bbox
                    2,
                )

        out.write(frame)
    
    # cleanup
    out.release()
    cap.release()


def mediapipe_FPE(key: dict) -> np.ndarray:
    """
    Estimate facial landmarks using MediaPipe
    
    Args:
        key: DataJoint key containing video information
        
    Returns:
        keypoints_2d: Array of shape (frames, 468, 3) containing facial landmarks
    """
    import mediapipe as mp
    from pose_pipeline.utils.video import get_frames_cv2
    
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    
    # Get video path
    video_path = (Video & key).fetch1("video_path")
    frames = get_frames_cv2(video_path)
    
    num_frames = len(frames)
    num_landmarks = 468  # MediaPipe face mesh landmarks
    
    keypoints_2d = np.zeros((num_frames, num_landmarks, 3))
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        for frame_idx, frame in enumerate(frames):
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                # Get the first (most confident) face
                face_landmarks = results.multi_face_landmarks[0]
                
                # Extract landmark coordinates
                height, width = frame.shape[:2]
                for i, landmark in enumerate(face_landmarks.landmark):
                    x = landmark.x * width
                    y = landmark.y * height
                    z = landmark.z  # Relative depth
                    
                    keypoints_2d[frame_idx, i] = [x, y, 1.0]  # Set confidence to 1.0
    
    return keypoints_2d
