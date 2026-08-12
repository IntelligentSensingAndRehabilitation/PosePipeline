import gc
import os

import cv2
import numpy as np

from pose_pipeline import Video

# torch is imported lazily inside the 3 functions that need it (_load_pytorch_metrabs and the two
# bridging_formats_* runners), so the pure-numpy helpers (filter_skeleton, noise_to_conf, etc.)
# can be imported without pulling in torch.

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


# PyTorch MeTRAbs (SRALab-CBM fork of isarandi/metrabs). Same eff2l model as the former
# TF SavedModel; the PyTorch weights are converted from that TF checkpoint.
METRABS_PT_MODEL_NAME = "metrabs_eff2l_384px_800k_28ds_pytorch"
METRABS_PT_URLS = [
    # Org-mirrored release asset (stable; create via `gh release create` — see README).
    "https://github.com/IntelligentSensingAndRehabilitation/metrabs/releases/download/"
    "weights-eff2l-v0.2/metrabs_eff2l_384px_800k_28ds_pytorch.tar.gz",
    # Upstream mirrors (fallback).
    "https://bit.ly/metrabs_l_pt",
    "https://omnomnom.vision.rwth-aachen.de/data/metrabs/metrabs_eff2l_384px_800k_28ds_pytorch.tar.gz",
]


def _ensure_metrabs_model(model_dir):
    """Auto-download + extract the PyTorch MeTRAbs model archive if not already present.

    Mirrors the old TFHub auto-download so users don't need a manual setup step.
    """
    if os.path.exists(os.path.join(model_dir, "ckpt.pt")):
        return
    import tarfile
    import tempfile
    import urllib.request

    parent = os.path.dirname(model_dir)
    os.makedirs(parent, exist_ok=True)
    for url in METRABS_PT_URLS:
        try:
            print(f"Downloading MeTRAbs (PyTorch) model from {url} ...")
            with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
                urllib.request.urlretrieve(url, tmp.name)
                with tarfile.open(tmp.name, mode="r:gz") as tar:
                    tar.extractall(parent)
            return
        except Exception as e:  # noqa: BLE001 - try the next mirror
            print(f"Failed to download from {url}: {e}")
    raise RuntimeError("Failed to download MeTRAbs PyTorch model from all URLs")


def _load_pytorch_metrabs(model_dir):
    """Build the crop model + multi-person estimator from a downloaded model directory."""
    import torch

    import simplepyutils as spu

    import metrabs_pytorch.backbones.efficientnet as effnet_pt
    import metrabs_pytorch.models.metrabs as metrabs_pt
    from metrabs_pytorch.multiperson import multiperson_model
    from metrabs_pytorch.util import get_config
    from posepile.joint_info import JointInfo

    get_config(os.path.join(model_dir, "config.yaml"))
    cfg = get_config()
    ji = np.load(os.path.join(model_dir, "joint_info.npz"))
    joint_info = JointInfo(ji["joint_names"], ji["joint_edges"])
    backbone = torch.nn.Sequential(
        effnet_pt.PreprocLayer(),
        getattr(effnet_pt, f"efficientnet_v2_{cfg.efficientnet_size}")().features,
    )
    crop_model = metrabs_pt.Metrabs(backbone, joint_info)
    crop_model.eval()
    crop_model((torch.zeros((1, 3, cfg.proc_side, cfg.proc_side)), torch.eye(3)[np.newaxis]))
    crop_model.load_state_dict(torch.load(os.path.join(model_dir, "ckpt.pt")))
    skeleton_infos = spu.load_pickle(os.path.join(model_dir, "skeleton_infos.pkl"))
    joint_transform = np.load(os.path.join(model_dir, "joint_transform_matrix.npy"))
    with torch.device("cuda"):
        model = multiperson_model.Pose3dEstimator(
            crop_model.cuda(), skeleton_infos, joint_transform
        ).cuda()

    # Expose skeleton metadata in the same numpy/bytes format the TF SavedModel used, so
    # make_coco_25 / filter_skeleton / get_joint_names / downstream keep working unchanged.
    model.per_skeleton_joint_names = {
        k: np.array([n.encode("utf-8") for n in v["names"]]) for k, v in skeleton_infos.items()
    }
    model.per_skeleton_indices = {
        k: np.array(v["indices"], dtype=np.int64) for k, v in skeleton_infos.items()
    }
    model.per_skeleton_joint_edges = {
        k: np.array(v["edges"], dtype=np.int64) for k, v in skeleton_infos.items()
    }
    return model


def get_model():
    if get_model.model is None:
        from pose_pipeline import MODEL_DATA_DIR

        # posepile.paths reads DATA_ROOT at import time; inference doesn't use datasets, so an
        # (empty) directory is enough. Set it before importing any metrabs_pytorch/posepile code.
        data_root = os.environ.setdefault("DATA_ROOT", os.path.join(MODEL_DATA_DIR, "posepile_data_root"))
        os.makedirs(data_root, exist_ok=True)

        model_dir = os.path.join(MODEL_DATA_DIR, METRABS_PT_MODEL_NAME)
        _ensure_metrabs_model(model_dir)

        print("Loading MeTRAbs (PyTorch) model...")
        model = _load_pytorch_metrabs(model_dir)
        model = make_coco_25(model)
        print("MeTRAbs (PyTorch) model loaded")

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

    import torch

    if model is None:
        model = get_model()

    from tqdm import tqdm

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
        frame_t = torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1))).cuda()

        try:
            with torch.inference_mode(), torch.device("cuda"):
                pred = model.detect_poses(
                    frame_t,
                    skeleton=skeleton,
                    num_aug=10,
                    average_aug=False,
                    detector_flip_aug=True,
                    detector_threshold=0.1,
                )
            poses3d_np = pred["poses3d"].cpu().numpy()
            boxes.append(pred["boxes"].cpu().numpy())
            keypoints2d.append(np.mean(pred["poses2d"].cpu().numpy(), axis=1))
            keypoints3d.append(np.mean(poses3d_np, axis=1))
            keypoint_noises.append(augmentation_noise(poses3d_np))
            del pred, poses3d_np
        except RuntimeError as e:
            # No person detected in this frame -> detector returns no boxes (empty torch.cat).
            if "non-empty list" not in str(e):
                raise
            boxes.append(np.zeros((0, 5)))
            keypoints2d.append(np.zeros((0, n_joints, 2)))
            keypoints3d.append(np.zeros((0, n_joints, 3)))
            keypoint_noises.append(np.zeros((0, n_joints)))

        del frame, frame_t
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
    import torch

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

        bbox_np = np.asarray([external_bboxes[frame_idx]], dtype=np.float32)  # (1, 4)
        bbox_t = torch.from_numpy(bbox_np).cuda()
        frame_t = torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1))).cuda()
        with torch.inference_mode(), torch.device("cuda"):
            pred = model.estimate_poses(
                frame_t, bbox_t, skeleton=skeleton, num_aug=10, average_aug=False
            )

        poses3d_np = pred["poses3d"].cpu().numpy()
        boxes.append(bbox_np)
        keypoints2d.append(np.mean(pred["poses2d"].cpu().numpy(), axis=1))
        keypoints3d.append(np.mean(poses3d_np, axis=1))
        keypoint_noises.append(augmentation_noise(poses3d_np))

        del pred, frame, frame_t, poses3d_np
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
