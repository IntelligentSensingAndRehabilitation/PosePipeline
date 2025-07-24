import os
import cv2
import numpy as np
from tqdm import tqdm
import datajoint as dj
from pose_pipeline import Video, PersonBbox

package = 'mmpose'

def normalize_scores(scores):
    max_score = np.max(scores)

    if max_score == 0:
        return scores
    
    return scores / max_score

def mmpose_custom_top_down(key, method='HRNet_W48_COCO'):

    from mmpose.apis import init_model as init_pose_estimator
    from mmpose.apis import inference_topdown
    from tqdm import tqdm

    from pose_pipeline import MODEL_DATA_DIR

    if method == 'Hiera_Movi87_1L3pref5':
        pose_cfg = "/home/vscode/workspace/VideoFoundationalModel/video_foundational_model/configs/hiera_movi87.py"
        pose_ckpt = "/home/vscode/workspace/VideoFoundationalModel/video_foundational_model/weights/epoch_5.pth"
        num_keypoints = 87
    
    print(f"processing {key}")
    # check if cuda is available
    bboxes = (PersonBbox & key).fetch1("bbox")
    video =  Video.get_robust_reader(key, return_cap=False) # returning video allows deleting it
    cap = cv2.VideoCapture(video)

    model = init_pose_estimator(pose_cfg, pose_ckpt)

    results = []
    scores = []
    visibility = []

    for bbox in tqdm(bboxes):

        # should match the length of identified person tracks
        ret, frame = cap.read()
        assert ret and frame is not None

        # handle the case where person is not tracked in frame
        if np.any(np.isnan(bbox)):
            results.append(np.zeros((num_keypoints, 2)))
            scores.append(np.zeros(num_keypoints))
            visibility.append(np.zeros(num_keypoints))
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = inference_topdown(model, frame, np.array(bbox)[None],'xywh')[0]

        # print(res)

        keypoints = res.pred_instances.keypoints[0]
        keypoint_confidences = res.pred_instances.keypoint_scores[0]
        keypoint_visibility = res.pred_instances.keypoints_visible[0]

        results.append(keypoints)
        scores.append(keypoint_confidences)
        visibility.append(keypoint_visibility)

    # Convert results to a numpy array
    results = np.asarray(results)

    # Convert scores to a numpy array
    scores = np.asarray(scores)
    # Normalize the score by dividing by the maximum score
    norm_scores = normalize_scores(scores)

    # Convert visibility to a numpy array
    visibility = np.asarray(visibility)

    # Add the normalized scores to the results
    results = np.concatenate([results, norm_scores[..., None]], axis=-1)

    cap.release()
    os.remove(video)

    return results, scores, visibility