import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

def yolov11_bounding_boxes(file_path,outfile=None):
    """
    Run YOLOv11 fine-tuned model with built-in tracking (BYTETrack or StrongSORT).
    Returns frame-by-frame tracking results.
    """
    
    # ------------Load fine-tuned model----------------
    model_path = f"/mnt/lucinda/yolov11/weights/yolov11_xl_fine_tuned.pt"
    model = YOLO(model_path)

    # --------------- Run Tracking --------------------
    tracking_results = model.track(
        source=file_path,
        tracker="bytetrack.yaml",  # or "strongsort.yaml"
        conf=0.4,  # confidence threshold
        device=0,  # check this if you have a GPU
        stream=True
        #save=outfile is not None,  # save output if outfile is specified
        )

    # ------- Convert YOLO output to required format -------
    tracks = []
    for result in tracking_results:
        frame_tracks = []
        if result.boxes is not None:
            for box in result.boxes:
                if box.id is None:
                        print("Warning: No Track id. Skipping box")
                        continue
                track_id = int(box.id.item())
                tlbr = box.xyxy[0].cpu().numpy().astype(np.float32)
                tlhw = box.xywh[0].cpu().numpy().astype(np.float32)
                conf = float(box.conf.item())
                print(f"Track ID: {track_id}")
                print(f"tlbr: {tlbr} with shape {tlbr.shape}")
                print(f"tlhw: {tlhw} with shape {tlhw.shape}")
                print(f"Confidence: {conf}")

                frame_tracks.append({
                    "track_id":track_id,
                    "tlbr":tlbr,
                    "tlhw":tlhw,
                    "confidence":conf
                })
            
        tracks.append(frame_tracks)

    return tracks