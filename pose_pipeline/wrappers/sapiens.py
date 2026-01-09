"""
Sapiens wrapper for PosePipeline.
Supports Pose, Depth, Normal, and Segmentation estimation using the JAX/Equinox backend.
"""

import os
import cv2
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from tqdm import tqdm
from typing import Dict, Any, List, Tuple

NUM_KEYPOINTS = 308

def get_joint_names():
    from sapiens_eqx import GOLIATH_308_KEYPOINT_NAMES
    return GOLIATH_308_KEYPOINT_NAMES


def visualize_depth_map(depth_crop: np.ndarray, seg_mask: np.ndarray = None, background_color: int = 100) -> np.ndarray:
    """
    Visualize depth map using official Sapiens approach.

    Uses COLORMAP_INFERNO with depth inversion (near camera = bright/warm, far = dark/cool).
    Background is filled with uniform gray.

    Args:
        depth_crop: Raw depth values (H, W)
        seg_mask: Optional segmentation mask (H, W), foreground > 0
        background_color: Background gray value (default 100)

    Returns:
        BGR image (H, W, 3)
    """
    H, W = depth_crop.shape

    # Determine foreground mask
    if seg_mask is not None:
        mask = seg_mask > 0
    else:
        mask = depth_crop > 0

    # Initialize with gray background
    result = np.full((H, W, 3), background_color, dtype=np.uint8)

    depth_foreground = depth_crop[mask]
    if len(depth_foreground) == 0:
        return result

    min_val, max_val = depth_foreground.min(), depth_foreground.max()
    if max_val <= min_val:
        return result

    # Invert: near camera = bright (high values), far = dark (low values)
    depth_normalized = 1 - ((depth_foreground - min_val) / (max_val - min_val))
    depth_normalized = (depth_normalized * 255).astype(np.uint8)

    # Apply INFERNO colormap
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
    result[mask] = depth_colored.reshape(-1, 3)

    return result


def visualize_normal_map(normal_crop: np.ndarray, seg_mask: np.ndarray = None) -> np.ndarray:
    """
    Visualize normal map using official Sapiens approach.

    Normalizes vectors to unit length and maps [-1, 1] to [0, 255].
    Background is black (masked regions set to -1 which maps to 0).

    Args:
        normal_crop: Raw normal vectors (3, H, W) in [-1, 1] range
        seg_mask: Optional segmentation mask (H, W), foreground > 0

    Returns:
        BGR image (H, W, 3)
    """
    # Transpose to (H, W, 3)
    normal_hwc = normal_crop.transpose(1, 2, 0)

    # Normalize to unit length
    normal_norm = np.linalg.norm(normal_hwc, axis=-1, keepdims=True)
    normal_normalized = normal_hwc / (normal_norm + 1e-5)

    # Apply mask for black background
    if seg_mask is not None:
        normal_normalized[seg_mask == 0] = -1  # Maps to black (0)
    else:
        # Fallback: areas with near-zero magnitude become black
        normal_normalized[normal_norm[..., 0] < 0.1] = -1

    # Convert [-1, 1] to [0, 255]
    normal_vis = ((normal_normalized + 1) / 2 * 255).astype(np.uint8)

    # RGB to BGR for OpenCV
    return normal_vis[:, :, ::-1]


class SapiensEstimator:
    """Unified Estimator for Sapiens tasks with JAX acceleration."""

    def __init__(self, variant: str = "2b", tasks: List[str] = ["pose"], img_size: Tuple[int, int] = (1024, 768)):
        from sapiens_eqx.model.sapiens_pose import SapiensPose
        from sapiens_eqx.model.sapiens_depth import SapiensDepth
        from sapiens_eqx.model.sapiens_normal import SapiensNormal
        from sapiens_eqx.model.sapiens_seg import SapiensSegmentation
        from sapiens_eqx.inference.pose_estimator import SapiensPoseEstimator
        from sapiens_eqx.inference.depth_estimator import SapiensDepthEstimator
        from sapiens_eqx.inference.normal_estimator import SapiensNormalEstimator
        from sapiens_eqx.inference.segmentation_estimator import SapiensSegmentationEstimator

        self.variant = variant
        self.tasks = tasks
        self.img_size = img_size
        self.token = os.environ.get("HF_TOKEN")

        self.estimators = {}

        if "pose" in tasks:
            try:
                model = SapiensPose.from_pretrained(variant=variant, token=self.token)
            except Exception:
                model = SapiensPose.from_pytorch(variant=variant)
            self.estimators["pose"] = SapiensPoseEstimator(model, img_size=img_size)

        if "depth" in tasks:
            try:
                model = SapiensDepth.from_pretrained(variant=variant, token=self.token)
            except Exception:
                model = SapiensDepth.from_pytorch(variant=variant)
            self.estimators["depth"] = SapiensDepthEstimator(model, img_size=img_size)

        if "normal" in tasks:
            try:
                model = SapiensNormal.from_pretrained(variant=variant, token=self.token)
            except Exception:
                model = SapiensNormal.from_pytorch(variant=variant)
            self.estimators["normal"] = SapiensNormalEstimator(model, img_size=img_size)

        if "seg" in tasks:
            try:
                model = SapiensSegmentation.from_pretrained(variant=variant, token=self.token)
            except Exception:
                model = SapiensSegmentation.from_pytorch(variant=variant)
            self.estimators["seg"] = SapiensSegmentationEstimator(model, img_size=img_size)

        # JIT the inference steps
        if "pose" in self.estimators:
            self._jit_pose = eqx.filter_jit(self._batched_pose_step)
        if "depth" in self.estimators:
            self._jit_depth = eqx.filter_jit(self._batched_depth_step)
        if "normal" in self.estimators:
            self._jit_normal = eqx.filter_jit(self._batched_normal_step)
        if "seg" in self.estimators:
            self._jit_seg = eqx.filter_jit(self._batched_seg_step)

    def _batched_pose_step(self, batch_tensor: jnp.ndarray):
        heatmaps = self.estimators["pose"].model(batch_tensor, inference=True)
        from sapiens_eqx.inference.post_processing import decode_udp_jax

        return jax.vmap(decode_udp_jax)(heatmaps)

    def _batched_depth_step(self, batch_tensor: jnp.ndarray):
        return self.estimators["depth"].model(batch_tensor, inference=True)

    def _batched_normal_step(self, batch_tensor: jnp.ndarray):
        return self.estimators["normal"].model(batch_tensor, inference=True)

    def _batched_seg_step(self, batch_tensor: jnp.ndarray):
        return self.estimators["seg"].model(batch_tensor, inference=True)

    def predict_video(self, video_path: str, bboxes: np.ndarray, present: np.ndarray, batch_size: int = 4):
        from sapiens_eqx.inference.demo_utils import box_to_center_scale, get_affine_transform

        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Ensure bboxes match video frames
        if len(bboxes) < num_frames:
            num_frames = len(bboxes)

        results = {t: [] for t in self.tasks}

        for i in tqdm(range(0, num_frames, batch_size), desc=f"Sapiens {self.variant}"):
            batch_frames = []
            batch_trans = []
            batch_indices = []

            for j in range(i, min(i + batch_size, num_frames)):
                ret, frame = cap.read()
                if not ret:
                    break

                if not present[j]:
                    batch_indices.append(None)
                    continue

                # Sapiens expected format [x1, y1, x2, y2]
                # PosePipeline bboxes are [x, y, w, h]
                x, y, w, h = bboxes[j]
                x1, y1, x2, y2 = x, y, x + w, y + h
                center, scale = box_to_center_scale(x1, y1, x2, y2)
                trans = get_affine_transform(center, scale, (self.img_size[1], self.img_size[0]))

                crop = cv2.warpAffine(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    trans,
                    (self.img_size[1], self.img_size[0]),
                    flags=cv2.INTER_LINEAR,
                )

                # Preprocess
                input_tensor = self.estimators[next(iter(self.estimators))].preprocess(crop)
                batch_frames.append(input_tensor[0])
                batch_trans.append(trans)
                batch_indices.append(len(batch_frames) - 1)

            if not batch_frames:
                for t in self.tasks:
                    results[t].extend([None] * (min(i + batch_size, num_frames) - i))
                continue

            # Padding for constant batch size to avoid JIT re-compilation
            actual_len = len(batch_frames)
            if actual_len < batch_size:
                padding = [jnp.zeros_like(batch_frames[0])] * (batch_size - actual_len)
                batch_tensor = jnp.stack(batch_frames + padding)
            else:
                batch_tensor = jnp.stack(batch_frames)

            # Inference
            batch_outputs = {}
            if "pose" in self.tasks:
                batch_outputs["pose"] = self._jit_pose(batch_tensor)
            if "depth" in self.tasks:
                batch_outputs["depth"] = self._jit_depth(batch_tensor)
            if "normal" in self.tasks:
                batch_outputs["normal"] = self._jit_normal(batch_tensor)
            if "seg" in self.tasks:
                batch_outputs["seg"] = self._jit_seg(batch_tensor)

            # Post-process and map back
            for idx_in_batch in range(min(i + batch_size, num_frames) - i):
                res_idx = batch_indices[idx_in_batch]
                if res_idx is None:
                    for t in self.tasks:
                        results[t].append(None)
                    continue

                trans = batch_trans[res_idx]
                inv_trans = cv2.invertAffineTransform(trans)

                if "pose" in self.tasks:
                    kpts_crop, scores = batch_outputs["pose"][0][res_idx], batch_outputs["pose"][1][res_idx]
                    kpts_crop = np.array(kpts_crop)
                    scores = np.array(scores)

                    # Scale back to crop size then to image
                    h_out, w_out = self.img_size[0] // 4, self.img_size[1] // 4
                    kpts_crop[:, 0] *= self.img_size[1] / w_out
                    kpts_crop[:, 1] *= self.img_size[0] / h_out

                    kpts_orig = cv2.transform(kpts_crop.reshape(-1, 1, 2), inv_trans).reshape(-1, 2)
                    results["pose"].append(np.concatenate([kpts_orig, scores[:, None]], axis=1))

                if "depth" in self.tasks:
                    # Output is (batch, 1, H, W)
                    depth_crop = np.array(batch_outputs["depth"][res_idx, 0])
                    results["depth"].append(depth_crop)

                if "normal" in self.tasks:
                    # Output is (batch, 3, H, W) - surface normals in range [-1, 1]
                    normal_crop = np.array(batch_outputs["normal"][res_idx])
                    results["normal"].append(normal_crop)

                if "seg" in self.tasks:
                    # Output is (batch, num_classes, H, W) logits
                    seg_logits = np.array(batch_outputs["seg"][res_idx])
                    # Get class predictions via argmax
                    seg_mask = np.argmax(seg_logits, axis=0).astype(np.uint8)
                    # Resize to crop size
                    seg_mask = cv2.resize(
                        seg_mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST
                    )
                    results["seg"].append(seg_mask)

        cap.release()

        # Standardize outputs
        final_results = {}
        if "pose" in self.tasks:
            # Stack into (N, NUM_KEYPOINTS, 3)
            K = NUM_KEYPOINTS
            stacked = np.full((num_frames, K, 3), np.nan)
            for idx, k in enumerate(results["pose"]):
                if k is not None:
                    stacked[idx] = k
            final_results["keypoints"] = stacked

        if "seg" in self.tasks:
            # Stack into (N, H, W) with 255 for missing frames
            H, W = self.img_size
            stacked = np.full((num_frames, H, W), 255, dtype=np.uint8)
            for idx, mask in enumerate(results["seg"]):
                if mask is not None:
                    stacked[idx] = mask
            final_results["segmentation"] = stacked

        if "depth" in self.tasks:
            # Return list of depth crops (variable output size from model)
            final_results["depth"] = results["depth"]

        if "normal" in self.tasks:
            # Return list of normal crops (variable output size from model)
            final_results["normal"] = results["normal"]

        return final_results


def sapiens_top_down_person(key: Dict[str, Any], variant: str = "1b", tasks=["pose"]) -> np.ndarray:
    """Entry point for DataJoint TopDownPerson table."""
    from pose_pipeline.pipeline import Video, PersonBbox

    video_path, bboxes, present = (Video * PersonBbox & key).fetch1("video", "bbox", "present")

    estimator = SapiensEstimator(variant=variant, tasks=tasks)
    results = estimator.predict_video(video_path, bboxes, present)

    if "tmp" in str(video_path) and os.path.exists(video_path):
        try:
            os.remove(video_path)
        except:
            pass

    return results["keypoints"]
