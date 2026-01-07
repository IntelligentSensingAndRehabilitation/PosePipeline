"""
Sapiens wrapper for PosePipeline.
Supports Pose, Depth, and Normal estimation using the JAX/Equinox backend.
"""

import os
import cv2
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Tuple

GOLIATH_308_KEYPOINT_NAMES = [
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
    "center_of_glabella",
    "center_of_nose_root",
    "tip_of_nose_bridge",
    "midpoint_1_of_nose_bridge",
    "midpoint_2_of_nose_bridge",
    "midpoint_3_of_nose_bridge",
    "center_of_labiomental_groove",
    "tip_of_chin",
    "upper_startpoint_of_r_eyebrow",
    "lower_startpoint_of_r_eyebrow",
    "end_of_r_eyebrow",
    "upper_midpoint_1_of_r_eyebrow",
    "lower_midpoint_1_of_r_eyebrow",
    "upper_midpoint_2_of_r_eyebrow",
    "upper_midpoint_3_of_r_eyebrow",
    "lower_midpoint_2_of_r_eyebrow",
    "lower_midpoint_3_of_r_eyebrow",
    "upper_startpoint_of_l_eyebrow",
    "lower_startpoint_of_l_eyebrow",
    "end_of_l_eyebrow",
    "upper_midpoint_1_of_l_eyebrow",
    "lower_midpoint_1_of_l_eyebrow",
    "upper_midpoint_2_of_l_eyebrow",
    "upper_midpoint_3_of_l_eyebrow",
    "lower_midpoint_2_of_l_eyebrow",
    "lower_midpoint_3_of_l_eyebrow",
    "l_inner_end_of_upper_lash_line",
    "l_outer_end_of_upper_lash_line",
    "l_centerpoint_of_upper_lash_line",
    "l_midpoint_2_of_upper_lash_line",
    "l_midpoint_1_of_upper_lash_line",
    "l_midpoint_6_of_upper_lash_line",
    "l_midpoint_5_of_upper_lash_line",
    "l_midpoint_4_of_upper_lash_line",
    "l_midpoint_3_of_upper_lash_line",
    "l_outer_end_of_upper_eyelid_line",
    "l_midpoint_6_of_upper_eyelid_line",
    "l_midpoint_2_of_upper_eyelid_line",
    "l_midpoint_5_of_upper_eyelid_line",
    "l_centerpoint_of_upper_eyelid_line",
    "l_midpoint_4_of_upper_eyelid_line",
    "l_midpoint_1_of_upper_eyelid_line",
    "l_midpoint_3_of_upper_eyelid_line",
    "l_midpoint_6_of_upper_crease_line",
    "l_midpoint_2_of_upper_crease_line",
    "l_midpoint_5_of_upper_crease_line",
    "l_centerpoint_of_upper_crease_line",
    "l_midpoint_4_of_upper_crease_line",
    "l_midpoint_1_of_upper_crease_line",
    "l_midpoint_3_of_upper_crease_line",
    "r_inner_end_of_upper_lash_line",
    "r_outer_end_of_upper_lash_line",
    "r_centerpoint_of_upper_lash_line",
    "r_midpoint_1_of_upper_lash_line",
    "r_midpoint_2_of_upper_lash_line",
    "r_midpoint_3_of_upper_lash_line",
    "r_midpoint_4_of_upper_lash_line",
    "r_midpoint_5_of_upper_lash_line",
    "r_midpoint_6_of_upper_lash_line",
    "r_outer_end_of_upper_eyelid_line",
    "r_midpoint_3_of_upper_eyelid_line",
    "r_midpoint_1_of_upper_eyelid_line",
    "r_midpoint_4_of_upper_eyelid_line",
    "r_centerpoint_of_upper_eyelid_line",
    "r_midpoint_5_of_upper_eyelid_line",
    "r_midpoint_2_of_upper_eyelid_line",
    "r_midpoint_6_of_upper_eyelid_line",
    "r_midpoint_3_of_upper_crease_line",
    "r_midpoint_1_of_upper_crease_line",
    "r_midpoint_4_of_upper_crease_line",
    "r_centerpoint_of_upper_crease_line",
    "r_midpoint_5_of_upper_crease_line",
    "r_midpoint_2_of_upper_crease_line",
    "r_midpoint_6_of_upper_crease_line",
    "l_inner_end_of_lower_lash_line",
    "l_outer_end_of_lower_lash_line",
    "l_centerpoint_of_lower_lash_line",
    "l_midpoint_2_of_lower_lash_line",
    "l_midpoint_1_of_lower_lash_line",
    "l_midpoint_6_of_lower_lash_line",
    "l_midpoint_5_of_lower_lash_line",
    "l_midpoint_4_of_lower_lash_line",
    "l_midpoint_3_of_lower_lash_line",
    "l_outer_end_of_lower_eyelid_line",
    "l_midpoint_6_of_lower_eyelid_line",
    "l_midpoint_2_of_lower_eyelid_line",
    "l_midpoint_5_of_lower_eyelid_line",
    "l_centerpoint_of_lower_eyelid_line",
    "l_midpoint_4_of_lower_eyelid_line",
    "l_midpoint_1_of_lower_eyelid_line",
    "l_midpoint_3_of_lower_eyelid_line",
    "r_inner_end_of_lower_lash_line",
    "r_outer_end_of_lower_lash_line",
    "r_centerpoint_of_lower_lash_line",
    "r_midpoint_1_of_lower_lash_line",
    "r_midpoint_2_of_lower_lash_line",
    "r_midpoint_3_of_lower_lash_line",
    "r_midpoint_4_of_lower_lash_line",
    "r_midpoint_5_of_lower_lash_line",
    "r_midpoint_6_of_lower_lash_line",
    "r_outer_end_of_lower_eyelid_line",
    "r_midpoint_3_of_lower_eyelid_line",
    "r_midpoint_1_of_lower_eyelid_line",
    "r_midpoint_4_of_lower_eyelid_line",
    "r_centerpoint_of_lower_eyelid_line",
    "r_midpoint_5_of_lower_eyelid_line",
    "r_midpoint_2_of_lower_eyelid_line",
    "r_midpoint_6_of_lower_eyelid_line",
    "tip_of_nose",
    "bottom_center_of_nose",
    "r_outer_corner_of_nose",
    "l_outer_corner_of_nose",
    "inner_corner_of_r_nostril",
    "outer_corner_of_r_nostril",
    "upper_corner_of_r_nostril",
    "inner_corner_of_l_nostril",
    "outer_corner_of_l_nostril",
    "upper_corner_of_l_nostril",
    "r_outer_corner_of_mouth",
    "l_outer_corner_of_mouth",
    "center_of_cupid_bow",
    "center_of_lower_outer_lip",
    "midpoint_1_of_upper_outer_lip",
    "midpoint_2_of_upper_outer_lip",
    "midpoint_1_of_lower_outer_lip",
    "midpoint_2_of_lower_outer_lip",
    "midpoint_3_of_upper_outer_lip",
    "midpoint_4_of_upper_outer_lip",
    "midpoint_5_of_upper_outer_lip",
    "midpoint_6_of_upper_outer_lip",
    "midpoint_3_of_lower_outer_lip",
    "midpoint_4_of_lower_outer_lip",
    "midpoint_5_of_lower_outer_lip",
    "midpoint_6_of_lower_outer_lip",
    "r_inner_corner_of_mouth",
    "l_inner_corner_of_mouth",
    "center_of_upper_inner_lip",
    "center_of_lower_inner_lip",
    "midpoint_1_of_upper_inner_lip",
    "midpoint_2_of_upper_inner_lip",
    "midpoint_1_of_lower_inner_lip",
    "midpoint_2_of_lower_inner_lip",
    "midpoint_3_of_upper_inner_lip",
    "midpoint_4_of_upper_inner_lip",
    "midpoint_5_of_upper_inner_lip",
    "midpoint_6_of_upper_inner_lip",
    "midpoint_3_of_lower_inner_lip",
    "midpoint_4_of_lower_inner_lip",
    "midpoint_5_of_lower_inner_lip",
    "midpoint_6_of_lower_inner_lip",
    "l_top_end_of_inferior_crus",
    "l_top_end_of_superior_crus",
    "l_start_of_antihelix",
    "l_end_of_antihelix",
    "l_midpoint_1_of_antihelix",
    "l_midpoint_1_of_inferior_crus",
    "l_midpoint_2_of_antihelix",
    "l_midpoint_3_of_antihelix",
    "l_point_1_of_inner_helix",
    "l_point_2_of_inner_helix",
    "l_point_3_of_inner_helix",
    "l_point_4_of_inner_helix",
    "l_point_5_of_inner_helix",
    "l_point_6_of_inner_helix",
    "l_point_7_of_inner_helix",
    "l_highest_point_of_antitragus",
    "l_bottom_point_of_tragus",
    "l_protruding_point_of_tragus",
    "l_top_point_of_tragus",
    "l_start_point_of_crus_of_helix",
    "l_deepest_point_of_concha",
    "l_tip_of_ear_lobe",
    "l_midpoint_between_22_15",
    "l_bottom_connecting_point_of_ear_lobe",
    "l_top_connecting_point_of_helix",
    "l_point_8_of_inner_helix",
    "r_top_end_of_inferior_crus",
    "r_top_end_of_superior_crus",
    "r_start_of_antihelix",
    "r_end_of_antihelix",
    "r_midpoint_1_of_antihelix",
    "r_midpoint_1_of_inferior_crus",
    "r_midpoint_2_of_antihelix",
    "r_midpoint_3_of_antihelix",
    "r_point_1_of_inner_helix",
    "r_point_8_of_inner_helix",
    "r_point_3_of_inner_helix",
    "r_point_4_of_inner_helix",
    "r_point_5_of_inner_helix",
    "r_point_6_of_inner_helix",
    "r_point_7_of_inner_helix",
    "r_highest_point_of_antitragus",
    "r_bottom_point_of_tragus",
    "r_protruding_point_of_tragus",
    "r_top_point_of_tragus",
    "r_start_point_of_crus_of_helix",
    "r_deepest_point_of_concha",
    "r_tip_of_ear_lobe",
    "r_midpoint_between_22_15",
    "r_bottom_connecting_point_of_ear_lobe",
    "r_top_connecting_point_of_helix",
    "r_point_2_of_inner_helix",
    "l_center_of_iris",
    "l_border_of_iris_3",
    "l_border_of_iris_midpoint_1",
    "l_border_of_iris_12",
    "l_border_of_iris_midpoint_4",
    "l_border_of_iris_9",
    "l_border_of_iris_midpoint_3",
    "l_border_of_iris_6",
    "l_border_of_iris_midpoint_2",
    "r_center_of_iris",
    "r_border_of_iris_3",
    "r_border_of_iris_midpoint_1",
    "r_border_of_iris_12",
    "r_border_of_iris_midpoint_4",
    "r_border_of_iris_9",
    "r_border_of_iris_midpoint_3",
    "r_border_of_iris_6",
    "r_border_of_iris_midpoint_2",
    "l_center_of_pupil",
    "l_border_of_pupil_3",
    "l_border_of_pupil_midpoint_1",
    "l_border_of_pupil_12",
    "l_border_of_pupil_midpoint_4",
    "l_border_of_pupil_9",
    "l_border_of_pupil_midpoint_3",
    "l_border_of_pupil_6",
    "l_border_of_pupil_midpoint_2",
    "r_center_of_pupil",
    "r_border_of_pupil_3",
    "r_border_of_pupil_midpoint_1",
    "r_border_of_pupil_12",
    "r_border_of_pupil_midpoint_4",
    "r_border_of_pupil_9",
    "r_border_of_pupil_midpoint_3",
    "r_border_of_pupil_6",
    "r_border_of_pupil_midpoint_2",
]


def is_sapiens_available() -> bool:
    """Check if the SapiensEqx package is installed."""
    import importlib.util

    return importlib.util.find_spec("sapiens_eqx") is not None


class SapiensEstimator:
    """Unified Estimator for Sapiens tasks with JAX acceleration."""

    def __init__(self, variant: str = "2b", tasks: List[str] = ["pose"], img_size: Tuple[int, int] = (1024, 768)):
        from sapiens_eqx.model.sapiens_pose import SapiensPose
        from sapiens_eqx.model.sapiens_depth import SapiensDepth
        from sapiens_eqx.model.sapiens_normal import SapiensNormal
        from sapiens_eqx.inference.pose_estimator import SapiensPoseEstimator
        from sapiens_eqx.inference.depth_estimator import SapiensDepthEstimator
        from sapiens_eqx.inference.normal_estimator import SapiensNormalEstimator

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

        # JIT the inference steps
        if "pose" in self.estimators:
            self._jit_pose = eqx.filter_jit(self._batched_pose_step)
        if "depth" in self.estimators:
            self._jit_depth = eqx.filter_jit(self._batched_depth_step)
        if "normal" in self.estimators:
            self._jit_normal = eqx.filter_jit(self._batched_normal_step)

    def _batched_pose_step(self, batch_tensor: jnp.ndarray):
        heatmaps = self.estimators["pose"].model(batch_tensor, inference=True)
        from sapiens_eqx.inference.post_processing import decode_udp_jax

        return jax.vmap(decode_udp_jax)(heatmaps)

    def _batched_depth_step(self, batch_tensor: jnp.ndarray):
        return self.estimators["depth"].model(batch_tensor, inference=True)

    def _batched_normal_step(self, batch_tensor: jnp.ndarray):
        return self.estimators["normal"].model(batch_tensor, inference=True)

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
                    depth_crop = np.array(batch_outputs["depth"][res_idx, 0])
                    # Resize to crop size
                    depth_crop = cv2.resize(
                        depth_crop, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR
                    )
                    # Warp back
                    # This is tricky for depth maps, usually we just keep the crop or warp back.
                    # For now, let's keep it consistent with keypoints if possible,
                    # but depth maps are full images.
                    # results["depth"].append(depth_crop) # Placeholder
                    results["depth"].append(None)  # Skipping complex depth warping for now

                if "normal" in self.tasks:
                    results["normal"].append(None)  # Skipping for now

        cap.release()

        # Standardize outputs
        final_results = {}
        if "pose" in self.tasks:
            # Stack into (N, 308, 3)
            K = 308
            stacked = np.full((num_frames, K, 3), np.nan)
            for idx, k in enumerate(results["pose"]):
                if k is not None:
                    stacked[idx] = k
            final_results["keypoints"] = stacked

        return final_results


def sapiens_top_down_person(key: Dict[str, Any], variant: str = "2b"):
    """Entry point for DataJoint TopDownPerson table."""
    from pose_pipeline.pipeline import Video, PersonBbox

    if not is_sapiens_available():
        raise ImportError("sapiens_eqx not installed.")

    video_path, bboxes, present = (Video * PersonBbox & key).fetch1("video", "bbox", "present")

    estimator = SapiensEstimator(variant=variant, tasks=["pose"])
    results = estimator.predict_video(video_path, bboxes, present)

    if "tmp" in str(video_path) and os.path.exists(video_path):
        try:
            os.remove(video_path)
        except:
            pass

    return results["keypoints"]
