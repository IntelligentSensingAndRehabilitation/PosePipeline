"""SAM3 video predictor wrapper for person tracking."""

from pathlib import Path

import numpy as np


def sam3_bounding_boxes(file_path: str) -> list[list[dict]]:
    """Run SAM3 person tracking and return per-frame bounding boxes.

    Returns standard tracking format: list[frames] of list[dict] with
    keys track_id, tlbr, tlhw, confidence.
    """
    import cv2
    import torch
    from sam3.model_builder import build_sam3_video_predictor

    cap = cv2.VideoCapture(file_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    predictor = build_sam3_video_predictor(gpus_to_use=range(torch.cuda.device_count()))
    try:
        raw_outputs = _run_video_session(predictor, file_path, "person")
    finally:
        predictor.shutdown()

    return _build_tracks(raw_outputs, num_frames)


def _run_video_session(predictor, video_path: str, prompt: str, frame_index: int = 0) -> dict:
    """Segment one video with an existing SAM3 predictor; return raw per-frame outputs."""
    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=str(Path(video_path)))
    )
    session_id = response["session_id"]
    predictor.handle_request(request=dict(type="reset_session", session_id=session_id))
    predictor.handle_request(
        request=dict(type="add_prompt", session_id=session_id, frame_index=frame_index, text=prompt)
    )
    outputs_per_frame = {}
    for frame_response in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        outputs_per_frame[frame_response["frame_index"]] = frame_response["outputs"]
    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    return outputs_per_frame


def _build_tracks(raw_outputs: dict, num_frames: int) -> list[list[dict]]:
    """Convert raw SAM3 mask outputs to bounding boxes in standard tracking format."""
    tracks: list[list[dict]] = [[] for _ in range(num_frames)]
    for frame_idx, frame_out in raw_outputs.items():
        obj_ids: list[int] = frame_out["out_obj_ids"].tolist()
        raw_masks = frame_out["out_binary_masks"]

        frame_tracks = []
        for i, obj_id in enumerate(obj_ids):
            mask_np = raw_masks[i]
            if hasattr(mask_np, "cpu"):
                mask_np = mask_np.cpu().numpy()
            bool_mask = mask_np.astype(bool)
            coords = np.argwhere(bool_mask)  # (K, 2) of (row, col) = (y, x)
            if len(coords) == 0:
                continue
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            frame_tracks.append(
                {
                    "track_id": obj_id,
                    "tlbr": np.array([x_min, y_min, x_max, y_max], dtype=float),
                    "tlhw": np.array([x_min, y_min, x_max - x_min, y_max - y_min], dtype=float),
                    "confidence": 1.0,
                }
            )
        if frame_idx < num_frames:
            tracks[frame_idx] = frame_tracks
    return tracks
