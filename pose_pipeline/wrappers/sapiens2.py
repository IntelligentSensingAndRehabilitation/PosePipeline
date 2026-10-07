"""
Sapiens2 wrapper for PosePipeline.
Supports Pose, Pointmap (depth), Normal, and Segmentation estimation using the
JAX/Equinox Sapiens2Eqx backend.
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

# Auto-select batch size based on model variant to avoid GPU OOM.
VARIANT_BATCH_SIZES = {"0.4b": 8, "0.8b": 4, "1b": 2}


def get_joint_names(normalize=True):
    """Return Sapiens Goliath 308 joint names.

    Args:
        normalize: If True (default), convert to Title Case (left_hip -> Left Hip)
                   to match normalized_joint_name_dictionary convention used elsewhere.
                   If False, return original Sapiens naming (lowercase with underscores).
    """
    from sapiens2_eqx import GOLIATH_308_KEYPOINT_NAMES

    if normalize:
        return [name.replace("_", " ").title() for name in GOLIATH_308_KEYPOINT_NAMES]
    return list(GOLIATH_308_KEYPOINT_NAMES)


# The model is passed as a JIT argument (not captured by closure) so its weights
# are treated as dynamic inputs rather than baked into the executable as constants.
@eqx.filter_jit
def _batched_pose_step(model, batch_tensor: jnp.ndarray):
    from sapiens2_eqx.inference import udp_decode

    heatmaps = jax.vmap(model)(batch_tensor)
    keypoints, scores = jax.vmap(udp_decode)(heatmaps)
    return keypoints, scores, heatmaps.shape[-2:]


@eqx.filter_jit
def _batched_dense_step(model, batch_tensor: jnp.ndarray):
    return jax.vmap(model)(batch_tensor)


class Sapiens2Estimator:
    """Unified Estimator for Sapiens2 tasks with JAX acceleration."""

    def __init__(self, variant: str = "0.4b", tasks: List[str] = ["pose"], img_size: Tuple[int, int] = (1024, 768)):
        from sapiens2_eqx.model import Sapiens2Pose, Sapiens2Pointmap, Sapiens2Normal, Sapiens2Seg

        self.variant = variant
        self.tasks = tasks
        self.img_size = img_size
        self.token = os.environ.get("HF_TOKEN")

        model_classes = {
            "pose": Sapiens2Pose,
            "pointmap": Sapiens2Pointmap,
            "normal": Sapiens2Normal,
            "seg": Sapiens2Seg,
        }

        self.models = {}
        for task in tasks:
            if task not in model_classes:
                raise ValueError(f"Unknown Sapiens2 task '{task}', expected one of {list(model_classes)}")
            model_cls = model_classes[task]
            try:
                model = model_cls.from_pretrained(variant=variant, img_size=img_size, token=self.token)
            except Exception:
                model = model_cls.from_pytorch(variant=variant, img_size=img_size)
            self.models[task] = model

    def predict_video(self, video_path: str, bboxes: np.ndarray, present: np.ndarray, batch_size: int = 4):
        from sapiens2_eqx.inference import preprocess_image
        from sapiens2_eqx.inference.pose_estimator import _box_to_center_scale, _get_affine_transform

        H, W = self.img_size

        cap = cv2.VideoCapture(video_path)
        try:
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Ensure bboxes match video frames
            if len(bboxes) < num_frames:
                num_frames = len(bboxes)

            results = {t: [None] * num_frames for t in self.tasks}

            for i in tqdm(range(0, num_frames, batch_size), desc=f"Sapiens2 {self.variant}"):
                batch_frames = []
                batch_trans = []
                batch_frame_idx = []

                for j in range(i, min(i + batch_size, num_frames)):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if not present[j]:
                        continue

                    # Sapiens2 expected format [x1, y1, x2, y2]
                    # PosePipeline bboxes are [x, y, w, h]
                    x, y, w, h = bboxes[j]
                    center, scale = _box_to_center_scale(x, y, x + w, y + h, aspect_ratio=W / H)
                    trans = _get_affine_transform(center, scale, output_size=(W, H))

                    crop = cv2.warpAffine(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        trans,
                        (W, H),
                        flags=cv2.INTER_LINEAR,
                    )

                    batch_frames.append(preprocess_image(crop, img_size=self.img_size))
                    batch_trans.append(trans)
                    batch_frame_idx.append(j)

                if not batch_frames:
                    continue

                # Padding for constant batch size to avoid JIT re-compilation
                actual_len = len(batch_frames)
                padding = [jnp.zeros_like(batch_frames[0])] * (batch_size - actual_len)
                batch_tensor = jnp.stack(batch_frames + padding)

                # Inference
                batch_outputs = {}
                for task, model in self.models.items():
                    if task == "pose":
                        batch_outputs[task] = _batched_pose_step(model, batch_tensor)
                    else:
                        batch_outputs[task] = _batched_dense_step(model, batch_tensor)

                # Post-process and map back
                for b, (frame_idx, trans) in enumerate(zip(batch_frame_idx, batch_trans)):
                    if "pose" in self.tasks:
                        kpts, scores, (h_out, w_out) = batch_outputs["pose"]
                        kpts_crop = np.array(kpts[b])
                        scores = np.array(scores[b])

                        # Scale heatmap coords back to crop size then to image
                        kpts_crop[:, 0] *= W / w_out
                        kpts_crop[:, 1] *= H / h_out

                        inv_trans = cv2.invertAffineTransform(trans)
                        kpts_orig = cv2.transform(kpts_crop.reshape(-1, 1, 2), inv_trans).reshape(-1, 2)
                        results["pose"][frame_idx] = np.concatenate([kpts_orig, scores[:, None]], axis=1)

                    if "pointmap" in self.tasks:
                        # Output is ((batch, 3, H, W) XYZ in camera coords, (batch, 1) scale)
                        pointmap, pm_scale = batch_outputs["pointmap"]
                        results["pointmap"][frame_idx] = {
                            "pointmap": np.array(pointmap[b]),
                            "scale": np.array(pm_scale[b]),
                        }

                    if "normal" in self.tasks:
                        # Output is (batch, 3, H, W) raw surface normals
                        results["normal"][frame_idx] = np.array(batch_outputs["normal"][b])

                    if "seg" in self.tasks:
                        # Output is (batch, num_classes, H, W) logits
                        seg_logits = np.array(batch_outputs["seg"][b])
                        seg_mask = np.argmax(seg_logits, axis=0).astype(np.uint8)
                        if seg_mask.shape != (H, W):
                            seg_mask = cv2.resize(seg_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                        results["seg"][frame_idx] = seg_mask

            # Standardize outputs
            final_results = {}
            if "pose" in self.tasks:
                # Stack into (N, NUM_KEYPOINTS, 3)
                stacked = np.full((num_frames, NUM_KEYPOINTS, 3), np.nan)
                for idx, k in enumerate(results["pose"]):
                    if k is not None:
                        stacked[idx] = k
                final_results["keypoints"] = stacked

            if "seg" in self.tasks:
                # Stack into (N, H, W) with 255 for missing frames
                stacked = np.full((num_frames, H, W), 255, dtype=np.uint8)
                for idx, mask in enumerate(results["seg"]):
                    if mask is not None:
                        stacked[idx] = mask
                final_results["segmentation"] = stacked

            if "pointmap" in self.tasks:
                # Return list of per-frame pointmap dicts (None for missing frames)
                final_results["pointmap"] = results["pointmap"]

            if "normal" in self.tasks:
                # Return list of normal crops (None for missing frames)
                final_results["normal"] = results["normal"]

            return final_results
        finally:
            cap.release()


_estimator_cache: Dict[str, Sapiens2Estimator] = {}


def _get_estimator(variant: str, tasks: List[str]) -> Sapiens2Estimator:
    """Return a cached Sapiens2Estimator, creating one if needed."""
    cache_key = f"{variant}_{'_'.join(sorted(tasks))}"
    if cache_key not in _estimator_cache:
        _estimator_cache.clear()  # only keep one variant loaded at a time
        _estimator_cache[cache_key] = Sapiens2Estimator(variant=variant, tasks=tasks)
    return _estimator_cache[cache_key]


def sapiens2_top_down_person(key: Dict[str, Any], variant: str = "1b", tasks=["pose"]) -> np.ndarray:
    """Entry point for DataJoint TopDownPerson table."""
    from pose_pipeline.pipeline import Video, PersonBbox

    video_path, bboxes, present = (Video * PersonBbox & key).fetch1("video", "bbox", "present")

    batch_size = VARIANT_BATCH_SIZES.get(variant, 2)

    estimator = _get_estimator(variant, tasks)
    results = estimator.predict_video(video_path, bboxes, present, batch_size=batch_size)

    # Clean up DataJoint temp file
    if "tmp" in str(video_path) and os.path.exists(video_path):
        os.remove(video_path)

    return results["keypoints"]
