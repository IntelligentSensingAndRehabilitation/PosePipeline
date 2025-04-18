import os
import numpy as np
from pose_pipeline import MODEL_DATA_DIR
from pose_pipeline.utils.bounding_box import get_person_dataloader
from pose_pipeline.utils.bounding_box import convert_crop_coords_to_orig_img, convert_crop_cam_to_orig_img
from pose_pipeline.env import add_path
from pose_pipeline import VideoInfo
import torch


def process_meva(key):

    crop_size = 224

    from pose_pipeline import MODEL_DATA_DIR
    from pose_pipeline.utils.bounding_box import get_person_dataloader

    config_file = os.path.join(MODEL_DATA_DIR, "meva/train_meva_2.yml")
    pretrained_model = os.path.join(MODEL_DATA_DIR, "meva/model_best.pth.tar")

    with add_path(os.environ["MEVA_PATH"]):
        frame_ids, dataloader, bbox = get_person_dataloader(key, crop_size=crop_size)

        from meva.lib.meva_model import MEVA, MEVA_demo
        from meva.utils.video_config import update_cfg

        device = "cuda"

        cfg = update_cfg(config_file)
        model = MEVA_demo(
            n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            seqlen=cfg.DATASET.SEQLEN,
            hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
            add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
            bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
            use_residual=cfg.MODEL.TGRU.RESIDUAL,
            cfg=cfg.VAE_CFG,
        ).to(device)

        ckpt = torch.load(pretrained_model)
        ckpt = ckpt["gen_state_dict"]
        model.load_state_dict(ckpt)
        model.eval()

        with torch.no_grad():
            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []

            for batch in dataloader:

                batch_image = batch.unsqueeze(0)
                batch_image = batch_image.to(device)

                batch_size, seqlen = batch_image.shape[:2]
                output = model(batch_image)[-1]

                pred_cam.append(output["theta"][:, :, :3].reshape(batch_size * seqlen, -1).cpu().detach().numpy())
                pred_verts.append(output["verts"].reshape(batch_size * seqlen, -1, 3).cpu().detach().numpy())
                pred_pose.append(output["theta"][:, :, 3:75].reshape(batch_size * seqlen, -1).cpu().detach().numpy())
                pred_betas.append(output["theta"][:, :, 75:].reshape(batch_size * seqlen, -1).cpu().detach().numpy())
                pred_joints3d.append(output["kp_3d"].reshape(batch_size * seqlen, -1, 3).cpu().detach().numpy())
                norm_joints2d.append(output["kp_2d"].reshape(batch_size * seqlen, -1, 2).cpu().detach().numpy())

    key["cams"] = np.concatenate(pred_cam, axis=0)
    key["verts"] = np.concatenate(pred_verts, axis=0)
    key["poses"] = np.concatenate(pred_pose, axis=0)
    key["betas"] = np.concatenate(pred_betas, axis=0)
    key["joints3d"] = np.concatenate(pred_joints3d, axis=0)
    key["joints2d"] = np.concatenate(norm_joints2d, axis=0)

    height, width = (VideoInfo & key).fetch1("height", "width")
    key["cams"] = convert_crop_cam_to_orig_img(key["cams"], bbox, width, height)
    key["joints2d"] = convert_crop_coords_to_orig_img(bbox, key["joints2d"], crop_size)

    return key
