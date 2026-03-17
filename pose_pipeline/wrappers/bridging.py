import gc
import os

import cv2
import numpy as np

from pose_pipeline import Video

# supported formats are
# 'smpl_24', 'h36m_17', 'h36m_25', 'mpi_inf_3dhp_17', 'mpi_inf_3dhp_28', 'coco_19', 'sailvos_26', 'gpa_34', 'aspset_17',
# 'bml_movi_87', 'mads_19', 'berkeley_mhad_43', 'total_capture_21', 'jta_22', 'ikea_asm_17', 'human4d_32', 'smplx_42',
# 'ghum_35', 'lsp_14', '3dpeople_29', 'umpm_15', 'kinectv2_25', 'smpl+head_30', ''


def make_coco_25(model):
    # foot keypoints are available in the model, but not listed in the indices
    all_joints = model.per_skeleton_joint_names[""]

    def f(x):
        x = x.decode("utf-8").split("_")[0]
        return x.encode("utf-8")

    coco_idx = [i for i, x in enumerate(all_joints) if "_coco" in x.decode("utf-8")]

    # make sure the new joints are at the end
    new = np.setdiff1d(coco_idx, model.per_skeleton_indices["coco_19"])
    updated = np.concatenate([model.per_skeleton_indices["coco_19"], new])
    model.per_skeleton_indices["coco_25"] = updated

    model.per_skeleton_joint_names["coco_25"] = [
        f(x) for x in model.per_skeleton_joint_names[""][updated]
    ]
    model.per_skeleton_joint_edges["coco_25"] = model.per_skeleton_joint_edges[
        "coco_19"
    ]

    return model


def get_model():
    if get_model.model is None:
        import tensorflow_hub as hub

        from pose_pipeline import tensorflow_memory_limit

        tensorflow_memory_limit()

        METRABS_URLS = [
            "https://bit.ly/metrabs_l",
            "https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_eff2l_y4_384px_800k_28ds.tar.gz",
        ]

        print("Loading MeTRAbs Model...")
        model_cache = os.environ.get("TFHUB_CACHE_DIR")
        print(f"Model cached in: {model_cache}")
        for url in METRABS_URLS:
            try:
                model = hub.load(url)
                break
            except Exception as e:
                print(f"Failed to load from {url}: {e}")
        else:
            raise RuntimeError("Failed to load MeTRAbs model from all URLs")
        print("MeTRAbs Model Loaded")
        model.per_skeleton_joint_names = {
            k: v.numpy() for k, v in model.per_skeleton_joint_names.items()
        }
        model.per_skeleton_indices = {
            k: v.numpy() for k, v in model.per_skeleton_indices.items()
        }

        model = make_coco_25(model)

        get_model.model = model

    return get_model.model


get_model.model = None


def get_joint_names(skeleton, model=None):

    if model is None:
        model = get_model()

    return model.per_skeleton_joint_names[skeleton]


def get_skeleton_edges(skeleton, model=None):

    if model is None:
        model = get_model()

    return model.per_skeleton_joint_edges[skeleton]


def filter_skeleton(keypoints, skeleton, model=None):

    if model is None:
        model = get_model()
    idx = model.per_skeleton_indices[skeleton]

    keypoints = np.array([k[..., idx, :] for k in keypoints])
    return keypoints


def scale_align(poses: np.ndarray) -> np.ndarray:
    square_scales = np.mean(np.square(poses), axis=(-2, -1), keepdims=True)
    mean_square_scale = np.mean(square_scales, axis=-3, keepdims=True)
    return poses * np.sqrt(mean_square_scale / square_scales)


def point_stdev(poses: np.ndarray, item_axis: int, coord_axis: int) -> np.ndarray:
    coordwise_variance = np.var(poses, axis=item_axis, keepdims=True)
    average_stdev = np.sqrt(np.sum(coordwise_variance, axis=coord_axis, keepdims=True))
    return np.squeeze(average_stdev, (item_axis, coord_axis))


def augmentation_noise(poses3d: np.ndarray) -> np.ndarray:
    return point_stdev(scale_align(poses3d), item_axis=1, coord_axis=-1)


def noise_to_conf(x, half_val=200, sharpness=50):
    x = -(x - half_val) / sharpness
    return 1 / (1 + np.exp(-x))


def bridging_formats_bottom_up(key, model=None, skeleton=""):

    if model is None:
        model = get_model()

    from tqdm import tqdm

    video = Video.get_robust_reader(key, return_cap=False)
    cap = cv2.VideoCapture(video)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    boxes = []
    keypoints2d = []
    keypoints3d = []
    keypoint_noises = []
    for frame_idx in tqdm(range(video_length)):
        ret, frame = cap.read()
        assert ret and frame is not None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pred = model.detect_poses(
            frame,
            skeleton=skeleton,
            num_aug=10,
            average_aug=False,
            detector_flip_aug=True,
            detector_threshold=0.1,
        )

        poses3d_np = pred["poses3d"].numpy()
        boxes.append(pred["boxes"].numpy())
        keypoints2d.append(np.mean(pred["poses2d"].numpy(), axis=1))
        keypoints3d.append(np.mean(poses3d_np, axis=1))
        keypoint_noises.append(augmentation_noise(poses3d_np))

        del pred, frame, poses3d_np
        if frame_idx % 100 == 0:
            gc.collect()

    cap.release()
    os.remove(video)

    return {
        "boxes": boxes,
        "keypoints2d": keypoints2d,
        "keypoints3d": keypoints3d,
        "keypoint_noise": keypoint_noises,
    }


# Bridging with focused keypoint detection using external bounding boxes
def bridging_formats_with_external_bbox(
    key: dict,
    external_bboxes: np.ndarray,
    bbox_present: np.ndarray,
    model: object | None = None,
    skeleton: str = "",
) -> dict[str, np.ndarray | list]:
    """Run MeTRAbs pose estimation using externally provided bounding boxes for each frame.

    Args:
        key: DataJoint key for the video.
        external_bboxes: np.ndarray, shape (num_frames, 4), each bbox as [x, y, w, h]
        bbox_present: np.ndarray, shape (num_frames,), boolean array indicating if bbox is present for each frame
        model: Optionally provide a loaded MeTRAbs model.
        skeleton: Skeleton type for the model.

    Returns:
        dict with keys: boxes, keypoints2d, keypoints3d, keypoint_noise
    """
    import tensorflow as tf
    from tqdm import tqdm

    if model is None:
        model = get_model()

    video = Video.get_robust_reader(key, return_cap=False)
    cap = cv2.VideoCapture(video)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_joints = model.per_skeleton_indices[skeleton].shape[0]
    boxes = []
    keypoints2d = []
    keypoints3d = []
    keypoint_noises = []

    for frame_idx in tqdm(range(video_length)):
        ret, frame = cap.read()
        assert ret and frame is not None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not bbox_present[frame_idx]:
            boxes.append(np.zeros((0, 4)))
            keypoints2d.append(np.zeros((1, n_joints, 2)))
            keypoints3d.append(np.zeros((1, n_joints, 3)))
            keypoint_noises.append(np.zeros((1, n_joints)))
            continue

        bbox = tf.convert_to_tensor([external_bboxes[frame_idx]], dtype=tf.float32)
        pred = model.estimate_poses(
            frame, bbox, skeleton=skeleton, num_aug=10, average_aug=False
        )

        poses3d_np = pred["poses3d"].numpy()
        boxes.append(bbox.numpy())
        keypoints2d.append(np.mean(pred["poses2d"].numpy(), axis=1))
        keypoints3d.append(np.mean(poses3d_np, axis=1))
        keypoint_noises.append(augmentation_noise(poses3d_np))

        del pred, frame, poses3d_np
        if frame_idx % 100 == 0:
            gc.collect()

    keypoints2d = np.squeeze(np.array(keypoints2d), axis=1)
    keypoints3d = np.squeeze(np.array(keypoints3d), axis=1)
    keypoint_noises = np.squeeze(np.array(keypoint_noises), axis=1)

    cap.release()
    os.remove(video)

    return {
        "boxes": boxes,
        "keypoints2d": keypoints2d,
        "keypoints3d": keypoints3d,
        "keypoint_noise": keypoint_noises,
    }


def get_overlay_callback(boxes, poses2d, joint_edges=None):
    def overlay_callback(image, idx):
        image = image.copy()
        bbox = boxes[idx]  # boxes is frames x 5
        p2d = poses2d[idx]  # poses2d is frames x 2
        small = int(5e-3 * np.max(image.shape))

        for bbox, p2d in zip(bbox, p2d):
            cv2.rectangle(
                image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3])),
                (255, 255, 255),
                small,
            )

            if joint_edges is not None:
                for i_start, i_end in joint_edges:
                    cv2.line(
                        image,
                        (int(p2d[i_start, 0]), int(p2d[i_start, 1])),
                        (int(p2d[i_end, 0]), int(p2d[i_end, 1])),
                        (0, 200, 100),
                        thickness=4,
                    )

            for x, y in p2d:
                cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), thickness=3)

        return image

    return overlay_callback


normalized_joint_name_dictionary = {
    "coco_25": [
        "Sternum",  # "Neck",
        "Nose",
        "Pelvis",
        "Left Shoulder",
        "Left Elbow",
        "Left Wrist",
        "Left Hip",
        "Left Knee",
        "Left Ankle",
        "Right Shoulder",
        "Right Elbow",
        "Right Wrist",
        "Right Hip",
        "Right Knee",
        "Right Ankle",
        "Left Eye",
        "Left Ear",
        "Right Eye",
        "Right Ear",
        "Left Big Toe",  # caled lfoo in the code
        "Left Little Toe",
        "Left Heel",
        "Right Big Toe",
        "Right Little Toe",
        "Right Heel",
    ],
    "bml_movi_87": [
        "backneck",
        "upperback",
        "clavicle",
        "sternum",
        "umbilicus",
        "lfronthead",
        "lbackhead",
        "lback",
        "lshom",
        "lupperarm",
        "lelbm",
        "lforearm",
        "lwrithumbside",
        "lwripinkieside",
        "lfin",
        "lasis",
        "lpsis",
        "lfrontthigh",
        "lthigh",
        "lknem",
        "lankm",
        "Left Heel",  # "lhee",
        "lfifthmetatarsal",
        "Left Big Toe",  # "ltoe",
        "lcheek",
        "lbreast",
        "lelbinner",
        "lwaist",
        "lthumb",
        "lfrontinnerthigh",
        "linnerknee",
        "lshin",
        "lfirstmetatarsal",
        "lfourthtoe",
        "lscapula",
        "lbum",
        "rfronthead",
        "rbackhead",
        "rback",
        "rshom",
        "rupperarm",
        "relbm",
        "rforearm",
        "rwrithumbside",
        "rwripinkieside",
        "rfin",
        "rasis",
        "rpsis",
        "rfrontthigh",
        "rthigh",
        "rknem",
        "rankm",
        "Right Heel",  # "rhee",
        "rfifthmetatarsal",
        "Right Big Toe",  # "rtoe",
        "rcheek",
        "rbreast",
        "relbinner",
        "rwaist",
        "rthumb",
        "rfrontinnerthigh",
        "rinnerknee",
        "rshin",
        "rfirstmetatarsal",
        "rfourthtoe",
        "rscapula",
        "rbum",
        "Head",  # "head",
        "mhip",
        "Pelvis",  # "pelv",
        "Sternum",  # "thor",
        "Left Ankle",  # "lank",
        "Left Elbow",  # "lelb",
        "Left Hip",  # "lhip",
        "Left Hand",  # "lhan",
        "Left Knee",  # "lkne",
        "Left Shoulder",  # "lsho",
        "Left Wrist",  # "lwri",
        "Left Foot",  # "lfoo",
        "Right Ankle",  # "rank",
        "Right Elbow",  # "relb",
        "Right Hip",  # "rhip",
        "Right Hand",  # "rhan",
        "Right Knee",  # "rkne",
        "Right Shoulder",  # "rsho",
        "Right Wrist",  # "rwri",
        "Right Foot",  # "rfoo",
    ],
}
