"""
Sapiens All-Modes Visualization Script.

Runs all four Sapiens modes (pose, depth, normal, seg) on a video
and produces a 2x2 tiled visualization.

Usage:
    python sapiens_all_modes.py <video_project> <filename>

Example:
    python sapiens_all_modes.py CONTROL_TEST p504_20240511_094008.20150962
"""

import sys
import cv2
import numpy as np
from tqdm import tqdm
from pose_pipeline.pipeline import Video, PersonBbox
from pose_pipeline.wrappers.sapiens import SapiensEstimator
from sapiens_eqx.inference.demo_utils import box_to_center_scale, get_affine_transform

# Sapiens Goliath Palette (RGB) - 28 classes
GOLIATH_PALETTE = np.array([
    [50, 50, 50],      # Background
    [255, 218, 0],     # Apparel
    [128, 200, 255],   # Face_Neck
    [255, 0, 109],     # Hair
    [189, 0, 204],     # Left_Foot
    [255, 0, 218],     # Left_Hand
    [0, 160, 204],     # Left_Lower_Arm
    [0, 255, 145],     # Left_Lower_Leg
    [204, 0, 131],     # Left_Shoe
    [182, 0, 255],     # Left_Sock
    [255, 109, 0],     # Left_Upper_Arm
    [0, 255, 255],     # Left_Upper_Leg
    [72, 0, 255],      # Lower_Clothing
    [204, 131, 0],     # Right_Foot
    [255, 0, 0],       # Right_Hand
    [72, 255, 0],      # Right_Lower_Arm
    [189, 204, 0],     # Right_Lower_Leg
    [182, 255, 0],     # Right_Shoe
    [102, 0, 204],     # Right_Sock
    [32, 72, 204],     # Right_Upper_Arm
    [0, 145, 255],     # Right_Upper_Leg
    [14, 204, 0],      # Torso
    [0, 128, 72],      # Upper_Clothing
    [204, 0, 43],      # Lower_Lip
    [235, 205, 119],   # Upper_Lip
    [115, 227, 112],   # Lower_Teeth
    [157, 113, 143],   # Upper_Teeth
    [132, 93, 50],     # Tongue
], dtype=np.uint8)

# Goliath 308 keypoint indices (from skeleton.py):
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
# 9: left_hip, 10: right_hip, 11: left_knee, 12: right_knee
# 13: left_ankle, 14: right_ankle
# 15: left_big_toe, 16: left_small_toe, 17: left_heel
# 18: right_big_toe, 19: right_small_toe, 20: right_heel
# 41: right_wrist, 62: left_wrist
# 69: neck

# Skeleton connections following COCO-style pattern
SKELETON_CONNECTIONS = [
    # Face
    (0, 1), (0, 2),         # nose -> eyes
    (1, 3), (2, 4),         # eyes -> ears

    # Neck to shoulders
    (69, 5), (69, 6),       # neck -> shoulders
    (69, 0),                # neck -> nose

    # Torso
    (5, 6),                 # shoulder to shoulder
    (5, 9), (6, 10),        # shoulders -> hips
    (9, 10),                # hip to hip

    # Left arm (green)
    (5, 7), (7, 62),        # left_shoulder -> left_elbow -> left_wrist

    # Right arm (orange)
    (6, 8), (8, 41),        # right_shoulder -> right_elbow -> right_wrist

    # Left leg (green)
    (9, 11), (11, 13),      # left_hip -> left_knee -> left_ankle

    # Right leg (orange)
    (10, 12), (12, 14),     # right_hip -> right_knee -> right_ankle

    # Left foot
    (13, 15), (13, 17),     # left_ankle -> big_toe, heel

    # Right foot
    (14, 18), (14, 20),     # right_ankle -> big_toe, heel
]

# Colors for skeleton (BGR): left=green, right=orange, center=blue
SKELETON_COLORS = [
    # Face (blue)
    (255, 153, 51), (255, 153, 51),   # nose -> eyes
    (255, 153, 51), (255, 153, 51),   # eyes -> ears

    # Neck (blue)
    (255, 153, 51), (255, 153, 51),   # neck -> shoulders
    (255, 153, 51),                    # neck -> nose

    # Torso (blue)
    (255, 153, 51),                    # shoulder to shoulder
    (255, 153, 51), (255, 153, 51),   # shoulders -> hips
    (255, 153, 51),                    # hip to hip

    # Left arm (green)
    (0, 255, 0), (0, 255, 0),

    # Right arm (orange)
    (0, 128, 255), (0, 128, 255),

    # Left leg (green)
    (0, 255, 0), (0, 255, 0),

    # Right leg (orange)
    (0, 128, 255), (0, 128, 255),

    # Left foot (green)
    (0, 255, 0), (0, 255, 0),

    # Right foot (orange)
    (0, 128, 255), (0, 128, 255),
]


def get_inverse_transform(bbox, img_size):
    """Compute the inverse affine transform for a bbox."""
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x + w, y + h
    center, scale = box_to_center_scale(x1, y1, x2, y2)
    trans = get_affine_transform(center, scale, (img_size[1], img_size[0]))
    inv_trans = cv2.invertAffineTransform(trans)
    return inv_trans


def visualize_pose(frame, keypoints, present, bbox, img_size, alpha=0.8):
    """Draw pose keypoints on frame with colored skeleton."""
    if not present or keypoints is None or np.isnan(keypoints).all():
        return frame

    frame = frame.copy()

    # Keypoints are already in original image coordinates from the wrapper
    kpts = keypoints[:, :2]
    scores = keypoints[:, 2]

    # Draw skeleton connections first (so keypoints are on top)
    for idx, (i, j) in enumerate(SKELETON_CONNECTIONS):
        if i < len(kpts) and j < len(kpts):
            if scores[i] > 0.3 and scores[j] > 0.3:
                pt1 = kpts[i]
                pt2 = kpts[j]
                if not np.isnan(pt1).any() and not np.isnan(pt2).any():
                    x1, y1 = int(pt1[0]), int(pt1[1])
                    x2, y2 = int(pt2[0]), int(pt2[1])
                    color = SKELETON_COLORS[idx] if idx < len(SKELETON_COLORS) else (0, 255, 255)
                    cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    # Draw keypoints
    for i, (pt, score) in enumerate(zip(kpts, scores)):
        if score > 0.3 and not np.isnan(pt).any():
            x, y = int(pt[0]), int(pt[1])
            # Color keypoints: left=green, right=orange, center=white
            if i in [1, 3, 5, 7, 9, 11, 13, 15, 16, 17, 62]:  # left side
                color = (0, 255, 0)
            elif i in [2, 4, 6, 8, 10, 12, 14, 18, 19, 20, 41]:  # right side
                color = (0, 128, 255)
            else:  # center
                color = (255, 255, 255)
            cv2.circle(frame, (x, y), 4, color, -1)
            cv2.circle(frame, (x, y), 4, (0, 0, 0), 1)  # black outline

    return frame


def visualize_depth(frame, depth_crop, present, bbox, img_size, alpha=0.6):
    """Render depth map with mean-color background (no original frame blending)."""
    if not present or depth_crop is None:
        return np.zeros_like(frame)

    height, width = frame.shape[:2]
    inv_trans = get_inverse_transform(bbox, img_size)

    # Resize depth to crop size
    depth_resized = cv2.resize(depth_crop, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)

    # Normalize depth to 0-255
    valid_mask = depth_resized > 0
    if valid_mask.any():
        d_min, d_max = depth_resized[valid_mask].min(), depth_resized[valid_mask].max()
        if d_max > d_min:
            depth_norm = ((depth_resized - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth_resized, dtype=np.uint8)
    else:
        return np.zeros_like(frame)

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)

    # Compute mean color of valid region for background fill
    mean_color = depth_colored[valid_mask].mean(axis=0).astype(np.uint8)

    # Warp back to original coordinates with mean color as border
    warped_depth = cv2.warpAffine(
        depth_colored, inv_trans, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(int(c) for c in mean_color)
    )

    # Create mask and fill remaining background with mean color
    warped_mask = cv2.warpAffine(
        valid_mask.astype(np.uint8) * 255, inv_trans, (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Fill background with mean color
    result = warped_depth.copy()
    result[warped_mask == 0] = mean_color

    return result


def visualize_normal(frame, normal_crop, present, bbox, img_size, alpha=0.6):
    """Render normal map with mean-color background (no original frame blending)."""
    if not present or normal_crop is None:
        return np.zeros_like(frame)

    height, width = frame.shape[:2]
    inv_trans = get_inverse_transform(bbox, img_size)

    # normal_crop is (3, H, W) in range [-1, 1], convert to (H, W, 3) RGB
    normal_resized = cv2.resize(
        normal_crop.transpose(1, 2, 0),
        (img_size[1], img_size[0]),
        interpolation=cv2.INTER_LINEAR
    )

    # Convert normals to RGB: map [-1, 1] to [0, 255]
    normal_rgb = ((normal_resized + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    normal_bgr = cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR)

    # Create valid mask
    valid_mask = np.abs(normal_resized).sum(axis=2) > 0.1

    # Compute mean color of valid region for background fill
    if valid_mask.any():
        mean_color = normal_bgr[valid_mask].mean(axis=0).astype(np.uint8)
    else:
        mean_color = np.array([128, 128, 128], dtype=np.uint8)  # neutral gray

    # Warp back to original coordinates with mean color as border
    warped_normal = cv2.warpAffine(
        normal_bgr, inv_trans, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(int(c) for c in mean_color)
    )

    # Create warped mask
    warped_mask = cv2.warpAffine(
        (valid_mask.astype(np.uint8) * 255), inv_trans, (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Fill background with mean color
    result = warped_normal.copy()
    result[warped_mask == 0] = mean_color

    return result


def visualize_seg(frame, seg_mask, present, bbox, img_size, alpha=0.6):
    """Overlay segmentation on frame."""
    if not present or seg_mask is None or seg_mask.max() >= 255:
        return frame

    frame = frame.copy()
    height, width = frame.shape[:2]
    inv_trans = get_inverse_transform(bbox, img_size)

    # Colorize the segmentation mask
    colored_mask = GOLIATH_PALETTE[seg_mask]  # (H, W, 3) RGB
    colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)

    # Warp back to original coordinates
    warped_mask = cv2.warpAffine(
        colored_mask, inv_trans, (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # Also warp the class mask to create blend mask
    warped_seg = cv2.warpAffine(
        seg_mask, inv_trans, (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Blend where segmentation is non-background
    blend_mask = (warped_seg > 0).astype(np.float32)[:, :, None]
    frame = (frame * (1 - alpha * blend_mask) + warped_mask * alpha * blend_mask).astype(np.uint8)

    return frame


def add_label(frame, text, position='top-left'):
    """Add a label to the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (255, 255, 255)
    bg_color = (0, 0, 0)

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    if position == 'top-left':
        x, y = 10, 30
    elif position == 'top-right':
        x, y = frame.shape[1] - text_width - 10, 30
    else:
        x, y = 10, 30

    # Draw background rectangle
    cv2.rectangle(frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), bg_color, -1)
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

    return frame


def create_tiled_video(video_path, results, bboxes, present, output_path,
                       img_size=(1024, 768), downsample=2):
    """Create a 2x2 tiled video with all four visualizations."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Downsampled dimensions
    tile_width = orig_width // downsample
    tile_height = orig_height // downsample

    # Output dimensions (2x2 grid)
    out_width = tile_width * 2
    out_height = tile_height * 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    num_frames = len(present)

    for i in tqdm(range(num_frames), desc="Creating tiled video"):
        ret, frame = cap.read()
        if not ret:
            break

        bbox = bboxes[i]
        is_present = present[i]

        # Create four visualizations
        frame_pose = visualize_pose(
            frame.copy(),
            results.get('keypoints', [None] * num_frames)[i] if 'keypoints' in results else None,
            is_present, bbox, img_size
        )

        frame_depth = visualize_depth(
            frame.copy(),
            results.get('depth', [None] * num_frames)[i] if 'depth' in results else None,
            is_present, bbox, img_size
        )

        frame_normal = visualize_normal(
            frame.copy(),
            results.get('normal', [None] * num_frames)[i] if 'normal' in results else None,
            is_present, bbox, img_size
        )

        frame_seg = visualize_seg(
            frame.copy(),
            results.get('segmentation', [None] * num_frames)[i] if 'segmentation' in results else None,
            is_present, bbox, img_size
        )

        # Add labels
        frame_pose = add_label(frame_pose, "POSE")
        frame_depth = add_label(frame_depth, "DEPTH")
        frame_normal = add_label(frame_normal, "NORMAL")
        frame_seg = add_label(frame_seg, "SEGMENTATION")

        # Downsample
        frame_pose = cv2.resize(frame_pose, (tile_width, tile_height))
        frame_depth = cv2.resize(frame_depth, (tile_width, tile_height))
        frame_normal = cv2.resize(frame_normal, (tile_width, tile_height))
        frame_seg = cv2.resize(frame_seg, (tile_width, tile_height))

        # Create 2x2 tile
        top_row = np.hstack([frame_pose, frame_seg])
        bottom_row = np.hstack([frame_depth, frame_normal])
        tiled_frame = np.vstack([top_row, bottom_row])

        out.write(tiled_frame)

    cap.release()
    out.release()
    print(f"Tiled video saved to: {output_path}")


def run_all_modes(video_project, filename, variant="0.3b", tracking_method=21, video_subject_id=0):
    """Run all four Sapiens modes on a video and create tiled visualization.

    Runs each model sequentially to avoid GPU memory issues.
    """
    import gc

    # Build key
    key = {
        'video_project': video_project,
        'filename': filename,
        'tracking_method': tracking_method,
        'video_subject_id': video_subject_id
    }

    print(f"Processing: {key}")

    # Fetch video and bboxes
    video_path, bboxes, present = (Video * PersonBbox & key).fetch1("video", "bbox", "present")
    print(f"Video path: {video_path}")
    print(f"Frames: {len(present)}, Present: {present.sum()}")

    # Run each task sequentially to avoid OOM
    all_results = {}
    img_size = (1024, 768)
    tasks = ["pose", "seg", "depth", "normal"]

    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Running {task.upper()} model ({variant})...")
        print(f"{'='*50}")

        # Create estimator for single task
        estimator = SapiensEstimator(variant=variant, tasks=[task], img_size=img_size)

        # Run prediction
        results = estimator.predict_video(video_path, bboxes, present, batch_size=4)

        # Store results
        all_results.update(results)

        # Print summary for this task
        for key_name, value in results.items():
            if value is not None:
                if isinstance(value, np.ndarray):
                    print(f"  {key_name}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, list):
                    non_none = sum(1 for v in value if v is not None)
                    print(f"  {key_name}: list of {len(value)} items, {non_none} non-None")

        # Clear memory
        del estimator
        gc.collect()

    # Create output filename
    output_path = f"{filename}_sapiens_all_modes.mp4"

    # Create tiled video
    print(f"\n{'='*50}")
    print("Creating tiled visualization...")
    print(f"{'='*50}")
    create_tiled_video(video_path, all_results, bboxes, present, output_path,
                       img_size=img_size, downsample=2)

    print(f"\nDone! Output: {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nAvailable arguments:")
        print("  video_project  - The project name (e.g., CONTROL_TEST)")
        print("  filename       - The video filename")
        print("  [variant]      - Model variant (default: 0.3b)")
        print("  [tracking_method] - Tracking method ID (default: 21)")
        sys.exit(1)

    video_project = sys.argv[1]
    filename = sys.argv[2]
    variant = sys.argv[3] if len(sys.argv) > 3 else "0.3b"
    tracking_method = int(sys.argv[4]) if len(sys.argv) > 4 else 21

    run_all_modes(video_project, filename, variant=variant, tracking_method=tracking_method)
