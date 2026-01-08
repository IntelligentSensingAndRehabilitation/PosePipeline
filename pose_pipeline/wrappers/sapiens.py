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

from sapiens_eqx import GOLIATH_308_KEYPOINT_NAMES


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
