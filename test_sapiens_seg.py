"""
Test script for Sapiens segmentation on a p504 trial.
Outputs a video with segmentation overlay.
"""

import cv2
import numpy as np
from tqdm import tqdm
from pose_pipeline.pipeline import Video, PersonBbox
from pose_pipeline.wrappers.sapiens import SapiensEstimator

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


def create_segmentation_video(video_path, segmentation, bboxes, present, output_path,
                              img_size=(1024, 768), alpha=0.5):
    """Create a video with segmentation overlay.

    Uses inverse affine transform to correctly map segmentation back to original
    image coordinates, accounting for the bbox expansion done during inference.
    """
    from sapiens_eqx.inference.demo_utils import box_to_center_scale, get_affine_transform

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    num_frames = len(segmentation)

    for i in tqdm(range(num_frames), desc="Writing video"):
        ret, frame = cap.read()
        if not ret:
            break

        if present[i] and segmentation[i].max() < 255:
            # Get bounding box and compute the SAME transform used during inference
            x, y, w, h = bboxes[i]
            x1, y1, x2, y2 = x, y, x + w, y + h
            center, scale = box_to_center_scale(x1, y1, x2, y2)
            trans = get_affine_transform(center, scale, (img_size[1], img_size[0]))
            inv_trans = cv2.invertAffineTransform(trans)

            # Colorize the segmentation mask
            seg_mask = segmentation[i]
            colored_mask = GOLIATH_PALETTE[seg_mask]  # (H, W, 3) RGB
            colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)

            # Warp the colored mask back to original image coordinates
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

        out.write(frame)

    cap.release()
    out.release()
    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    # Pick a trial from p504
    key = {
        'video_project': 'CONTROL_TEST',
        'filename': 'p504_20240511_094008.20150962',
        'tracking_method': 21,
        'video_subject_id': 0
    }

    print(f"Running segmentation on: {key}")

    # Fetch video and bboxes
    video_path, bboxes, present = (Video * PersonBbox & key).fetch1("video", "bbox", "present")
    print(f"Video path: {video_path}")
    print(f"Bboxes shape: {bboxes.shape}")
    print(f"Present shape: {present.shape}, sum: {present.sum()}")

    # Create estimator with 0.3b segmentation model
    print("\nInitializing SapiensEstimator with 0.3b seg model...")
    estimator = SapiensEstimator(variant="0.3b", tasks=["seg"])

    # Run prediction
    print("\nRunning prediction...")
    results = estimator.predict_video(video_path, bboxes, present, batch_size=4)

    # Check results
    seg = results["segmentation"]
    print(f"\nSegmentation output shape: {seg.shape}")
    print(f"Unique classes in output: {np.unique(seg)}")
    print(f"Non-255 (valid) frames: {(seg != 255).any(axis=(1,2)).sum()}")

    # Create output video
    output_path = "p504_segmentation_output.mp4"
    print(f"\nCreating segmentation video...")
    create_segmentation_video(video_path, seg, bboxes, present, output_path, alpha=0.6)

    print("\nDone!")
