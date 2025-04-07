import os
import cv2
import numpy as np
import datajoint as dj
from pose_pipeline import Video
from pose_pipeline.pipeline import HandBbox
from tqdm import tqdm
from mim import download 

package = 'mmpose'

def mmpose_HPE(key, method="RTMPoseHand5"):

    from pose_pipeline import MODEL_DATA_DIR

    from mmpose.apis import inference_topdown, init_model
    from mmpose.evaluation.functional import nms

    if method == "RTMPoseHand5":
        # Define the model config and checkpoint files
        pose_config_id = "rtmpose-m_8xb256-210e_hand5-256x256"
        pose_checkpoint = "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth"

        # define the destination folder
        destination = os.path.join(MODEL_DATA_DIR, f"mmpose/{method}/")

        # download the model and checkpoints
        download(package, [pose_config_id], dest_root=destination)

        # define the model config and checkpoints paths
        pose_model_cfg = os.path.join(destination, f"{pose_config_id}.py")
        pose_model_ckpt = os.path.join(destination, pose_checkpoint)
        
    elif method == "RTMPoseCOCO":
        # Define the model config and checkpoint files
        pose_config_id = "rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256"
        pose_checkpoint = "rtmpose-m_simcc-coco-wholebody-hand_pt-aic-coco_210e-256x256-99477206_20230228.pth"

        # define the destination folder
        destination = os.path.join(MODEL_DATA_DIR, f"mmpose/{method}/")

        # download the model and checkpoints
        download(package, [pose_config_id], dest_root=destination)

        # define the model config and checkpoints paths
        pose_model_cfg = os.path.join(destination, f"{pose_config_id}.py")
        pose_model_ckpt = os.path.join(destination, pose_checkpoint)
    elif method == "freihand":
        # Define the model config and checkpoint files
        pose_config_id = "td-hm_res50_8xb64-100e_freihand2d-224x224"
        pose_checkpoint = "res50_freihand_224x224-ff0799bc_20200914.pth"

        # define the destination folder
        destination = os.path.join(MODEL_DATA_DIR, f"mmpose/{method}/")

        # download the model and checkpoints
        download(package, [pose_config_id], dest_root=destination)

        # define the model config and checkpoints paths
        pose_model_cfg = os.path.join(destination, f"{pose_config_id}.py")
        pose_model_ckpt = os.path.join(destination, pose_checkpoint)

    elif method == "HRNet_dark":
        # Define the model config and checkpoint files
        pose_config_id = "td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256"
        pose_checkpoint = "hrnetv2_w18_rhd2d_256x256_dark-4df3a347_20210330.pth"

        # define the destination folder
        destination = os.path.join(MODEL_DATA_DIR, f"mmpose/{method}/")

        # download the model and checkpoints
        download(package, [pose_config_id], dest_root=destination)

        # define the model config and checkpoints paths
        pose_model_cfg = os.path.join(destination, f"{pose_config_id}.py")
        pose_model_ckpt = os.path.join(destination, pose_checkpoint)
        
    elif method == "HRNet_udp":
        # Define the model config and checkpoint files
        pose_config_id = "td-hm_hrnetv2-w18_udp-8xb64-210e_onehand10k-256x256"
        pose_checkpoint = "hrnetv2_w18_onehand10k_256x256_udp-0d1b515d_20210330.pth"

        # define the destination folder
        destination = os.path.join(MODEL_DATA_DIR, f"mmpose/{method}/")

        # download the model and checkpoints
        download(package, [pose_config_id], dest_root=destination)

        # define the model config and checkpoints paths
        pose_model_cfg = os.path.join(destination, f"{pose_config_id}.py")
        pose_model_ckpt = os.path.join(destination, pose_checkpoint)

    device = "cuda"
    model = init_model(pose_model_cfg, pose_model_ckpt, device=device)

    video = Video.get_robust_reader(
        key, return_cap=False
    )  # returning video allows deleting it
    bboxes = (HandBbox & key).fetch1("bboxes")

    cap = cv2.VideoCapture(video)
    results = []
    for bbox in tqdm(bboxes):
        ret, frame = cap.read()
        assert ret and frame is not None
        # run the frame through the model
        # Pose_results includes the number of detections
        pose_results = inference_topdown(model, frame, bbox)
        # get prediction instances from mmpose results
        num_hands = len(pose_results)
        keypoints_2d = []
        for i in range(num_hands):
            # pred_instances includes the scores as well as keypoint
            pred_instances = pose_results[i].pred_instances
            keypoints = pred_instances.keypoints
            keypoint_scores = pred_instances.keypoint_scores
            # concat scores and keypoints(flatten)
            keypoints_2d.append(
                np.concatenate((keypoints[0, :, :], keypoint_scores.T), axis=-1)
            )
        results.append(np.concatenate(keypoints_2d, axis=0))

    cap.release()
    os.remove(video)

    return np.array(results)


def overlay_hand_keypoints(video, output_file, keypoints, bboxes):
    """Process a video and create overlay of keypoints

    Args:
    video (str): filename for source (from key)
    output_file (str): output filename
    keypoints (list): list of list of keypoints
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
        frame = draw_keypoints(frame, keypoints_2d, threshold=0.2)
        for bbox in bboxes[frame_idx]:
            frame = cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2,
            )  # Green color, 2 pixel thickness

            out.write(frame)
        #remove
        out.release()
        cap.release()


def plot_triangulated_keypoints(key, kp3d = None, only_osim = False,useBlurred = True, crop_bbox = False, camera_id = None):
        from pose_pipeline.pipeline import HandPoseEstimation, HandBbox
        from multi_camera.datajoint.sessions import Recording
        from multi_camera.datajoint.multi_camera_dj import Calibration, MultiCameraRecording, SingleCameraVideo,CalibratedRecording 
        from multi_camera.analysis.camera import project_distortion, get_intrinsic, get_extrinsic, distort_3d

        from pose_pipeline.pipeline import Video, VideoInfo, BlurredVideo, TopDownPerson
        from pose_pipeline.utils.visualization import video_overlay, draw_keypoints
        import cv2

        videos = (HandPoseEstimation * MultiCameraRecording  * SingleCameraVideo & Recording & key & 'estimation_method=0' & 'detection_method=3').proj()
        camera_params= (CalibratedRecording * Calibration & key).fetch1("camera_calibration")
        camera_names = (CalibratedRecording * Calibration & key).fetch1("camera_names")
        video_keys = (videos).fetch("KEY")
        fps = np.unique((VideoInfo & video_keys[0]).fetch1("fps"))
        width = np.unique((VideoInfo & video_keys).fetch("width"))[0]
        height = np.unique((VideoInfo & video_keys).fetch("height"))[0]
        if kp3d is None:
            only_2D = True
        else:
            only_2D = False
            keypoints_2d_MJX = np.array([project_distortion(camera_params, i, kp3d) for i in range(camera_params["mtx"].shape[0])])
        num_cameras = range(len(camera_names))
        if camera_id is not None:
            num_cameras = [camera_id]
        results = []
        for ci in num_cameras:
            cam_idx = ci
            
            i = cam_idx
            # get camera parameters
            K = np.array(get_intrinsic(camera_params, i))

            # don't use real extrinsic since we apply distortion which does this
            R = np.eye(3)
            T = np.zeros((3,))
            cameras = {"K": [K], "R": [R], "T": [T]}


            background = np.ones((height, width, 3), dtype=np.uint8) * 127
            if useBlurred:
                vid_file = (BlurredVideo & video_keys[i]).fetch1("output_video")
            else:
                vid_file = (Video & video_keys[i]).fetch1("video")
            vid = cv2.VideoCapture(vid_file)

            kp2d_camera = np.asarray((HandPoseEstimation & video_keys[ci]).fetch1("keypoints_2d"))
            kp2d_camera = kp2d_camera.reshape(kp2d_camera.shape[0], -1, kp2d_camera.shape[-1])  
            moviinds = np.array((3,2,39,41,57,43,44,8,10,26,12,13))
        
            movikeys = video_keys[ci].copy()
            movikeys['top_down_method']=12

            joints2dMovi = (TopDownPerson & movikeys).fetch1('keypoints')[:,moviinds,:]  
            kp2d_camera = np.concatenate((joints2dMovi, kp2d_camera), axis=1)
            
            
            bboxes = np.asarray((HandBbox & video_keys[ci]).fetch1('bboxes'))
            bboxes = bboxes[:,0,:]

            bbox = np.min(bboxes,axis=0)
            bbox[-2:]= np.max(bboxes,axis=0)[-2:]
            # bbox[:2] -= bbox[:2]/2
            # bbox[-2:] += bbox[-2:]/2
            
            def render_overlay(frame, idx, frame_idx):
                raw_frame = frame.copy()

                if not only_osim or only_2D:
                    # #Red
                    # color = (30, 30, 200)
                    # raw_frame = draw_keypoints(raw_frame, np.array(keypoints_2d_triangulated[idx, frame_idx]), radius=5, threshold=0.10, border_color=color, color=color)

                    #Green
                    color = (30, 200, 30)
                    raw_frame = draw_keypoints(raw_frame, np.array(kp2d_camera[frame_idx,:]), radius=10, threshold=0.10, border_color=color, color=color)
                    
                if not only_2D or only_osim:
                    #Blue
                    color = (200, 30, 30)
                    raw_frame = draw_keypoints(raw_frame, np.array(keypoints_2d_MJX[idx, frame_idx]), radius=10, threshold=0.10, border_color=color, color=color)
                    
                raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

                image = raw_frame
                target_size= (280, int(280 * 1920 / 1080))
                dilate= 1.4 
                src = np.asarray([[0,0], [width,height], [0, height]])
                if crop_bbox:
                    # three points on corner of bounding box
                    src = np.asarray([[bbox[0], bbox[1]], [bbox[2] ,bbox[3]], [bbox[0], bbox[3]]])
                dst = np.array([[0, 0], [target_size[0], target_size[1]], [0, target_size[1]]])  # .astype(np.float32)
                trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
                image = cv2.warpAffine(image, trans, target_size, flags=cv2.INTER_LINEAR)
                return image
            
            def make_frames():
                list_frames = []
                vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                for frame_num in tqdm(range(kp2d_camera.shape[0])):
                    _, frame = vid.read()
                    frame = render_overlay(frame, cam_idx, frame_num)
                    list_frames.append(frame)
                return list_frames
            
            results.append(make_frames())
            os.remove(vid_file)
            vid.release()
        return results
        

def plot_3d_reprojected_keypoints(key, kp3d, only_osim = False, useBlurred = True):
    results = plot_triangulated_keypoints(key, kp3d, only_osim = only_osim, useBlurred= useBlurred)
    def images_to_grid(images, n_cols=4):
        n_rows = int(np.ceil(len(images) / n_cols))
        grid = np.zeros((n_rows * images[0].shape[0], n_cols * images[0].shape[1], 3), dtype=np.uint8)
        for i, img in enumerate(images):
            row = i // n_cols
            col = i % n_cols
            grid[row * img.shape[0] : (row + 1) * img.shape[0], col * img.shape[1] : (col + 1) * img.shape[1], :] = img
        return grid
    def save_collated_video(grid):
        from pose_pipeline.pipeline import HandPoseEstimationMethodLookup,HandPoseEstimation
        method = (HandPoseEstimationMethodLookup & key).fetch1('estimation_method_name')
        prefix= ' '.join((HandPoseEstimation & key).fetch('filename')[0].split('_')[:-2])

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(f'./{prefix}_{method}.mp4', fourcc, 29, (grid[0].shape[1], grid[0].shape[0]))
        for frame in tqdm(grid, desc="Writing"):
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
    # collate the results into a grid
    grid = [images_to_grid(r) for r in zip(*results)]
    save_collated_video(grid)
    return results, grid

