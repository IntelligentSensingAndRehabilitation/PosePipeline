# This script confirms that the mmlab packages are installed correctly.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# base path is relative to the location of this script
base_path = os.path.dirname(__file__)


def test_mmengine():
    import mmengine

    assert hasattr(mmengine, '__version__'), "mmengine is not installed correctly"
    assert isinstance(mmengine.__version__, str), "mmengine version is not a string"
    assert len(mmengine.__version__) > 0, "mmengine version string is empty"

    print(f"mmengine version: {mmengine.__version__}")

def test_mmcv():
    # Copied the test from mmcv github
    # https://github.com/open-mmlab/mmcv/blob/main/.dev_scripts/check_installation.py

    import numpy as np
    import torch

    from mmcv.ops import box_iou_rotated
    from mmcv.utils import collect_env

    def check_installation():
        """Check whether mmcv has been installed successfully."""
        np_boxes1 = np.asarray(
            [[1.0, 1.0, 3.0, 4.0, 0.5], [2.0, 2.0, 3.0, 4.0, 0.6],
            [7.0, 7.0, 8.0, 8.0, 0.4]],
            dtype=np.float32)
        np_boxes2 = np.asarray(
            [[0.0, 2.0, 2.0, 5.0, 0.3], [2.0, 1.0, 3.0, 3.0, 0.5],
            [5.0, 5.0, 6.0, 7.0, 0.4]],
            dtype=np.float32)
        boxes1 = torch.from_numpy(np_boxes1)
        boxes2 = torch.from_numpy(np_boxes2)

        # test mmcv with CPU ops
        box_iou_rotated(boxes1, boxes2)
        print('CPU ops were compiled successfully.')

        # test mmcv with both CPU and CUDA ops
        if torch.cuda.is_available():
            boxes1 = boxes1.cuda()
            boxes2 = boxes2.cuda()
            box_iou_rotated(boxes1, boxes2)
            print('CUDA ops were compiled successfully.')
        else:
            print('No CUDA runtime is found, skipping the checking of CUDA ops.')

    print('Start checking the installation of mmcv ...')
    check_installation()
    print('mmcv has been installed successfully.\n')

    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    print('Environment information:')
    print(dash_line + env_info + '\n' + dash_line)

def test_mmdetection():
    from mim import download 
    import mmdet.apis
    from mmdet.utils import register_all_modules
    from mmdet.apis import init_detector, inference_detector

    package = 'mmdet'

    config_file = 'rtmdet_tiny_8xb32-300e_coco'
    checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

    # Base path is relative to the location of this script
    input_image_path = os.path.join(base_path, "det_demo.jpg")

    # Define the destination folder relative to the location of this script
    destination = os.path.join(base_path, 'mmdetection/')

    # Download the model and checkpoints
    download(package, [config_file], dest_root=destination)

    # Define the model config and checkpoints paths
    model_config = os.path.join(destination, f"{config_file}.py")
    detector_checkpoint = os.path.join(destination, checkpoint_file)
    # Register all modules from mmdet
    register_all_modules()

    # Initialize the model
    model = mmdet.apis.init_detector(model_config, detector_checkpoint, device='cuda:0')
    # Perform inference
    print(inference_detector(model, input_image_path))

def test_mmdetection_demo():

    from mmengine.logging import print_log
    from mmdet.apis import DetInferencer

    # Base path is relative to the location of this script
    cfg_path = os.path.join(base_path, "mmdetection/rtmdet_tiny_8xb32-300e_coco.py")
    ckpt_path = os.path.join(base_path, "mmdetection/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth")

    inferencer = DetInferencer(
        model=cfg_path,
        weights=ckpt_path,
        device='cuda:0',
        palette="none",
    )

    input_image_path = os.path.join(base_path, "det_demo.jpg")
    output_image_path = os.path.join(base_path, "mmdet_output")

    # Run inference on the single image
    inferencer(
        inputs=input_image_path,
        batch_size=1,
        pred_score_thr=0.3,
        out_dir=output_image_path,
        show=False,
        no_save_vis=False,
        no_save_pred=False,
        print_result=True,
    )

def test_mmpose():
    from mim import download
    from mmpose.apis import inference_topdown, init_model
    from mmpose.utils import register_all_modules

    # Register all modules from mmpose
    register_all_modules()

    package = 'mmpose'

    # base path is relative to the location of this script
    input_image_path = os.path.join(base_path, "det_demo.jpg")

    # Define the model config and checkpoint files
    pose_config_id = "td-hm_hrnet-w48_8xb32-210e_coco-256x192"
    pose_checkpoint = "td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth"

    # define the destination folder
    destination = os.path.join(base_path, f"mmpose/")

    # download the model and checkpoints
    download(package, [pose_config_id], dest_root=destination)

    # define the model config and checkpoints paths
    pose_cfg = os.path.join(destination, f"{pose_config_id}.py")
    pose_ckpt = os.path.join(destination, pose_checkpoint)

    # Initialize the model
    pose_model = init_model(pose_cfg, pose_ckpt, device='cuda:0')

    print(inference_topdown(pose_model, input_image_path))

def test_mmpose_demo():
    from mmengine.logging import print_log
    from mmcv.image import imread
    from mmpose.apis import init_model, inference_topdown
    from mmpose.registry import VISUALIZERS
    from mmpose.structures import merge_data_samples

    # Base path is relative to the location of this script
    cfg_path = os.path.join(base_path, "mmpose/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py")
    ckpt_path = os.path.join(base_path, "mmpose/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth")

    model = init_model(
        cfg_path,
        ckpt_path,
        device='cuda:0',
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    )

    input_image_path = os.path.join(base_path, "pose_demo.jpg")
    output_image_path = os.path.join(base_path, "mmpose_output.jpg")

    # init visualizer
    model.cfg.visualizer.radius = 3
    model.cfg.visualizer.alpha = 0.8
    model.cfg.visualizer.line_width = 1

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(
        model.dataset_meta, skeleton_style='mmpose')
    
    # inference a single image
    batch_results = inference_topdown(model, input_image_path)
    results = merge_data_samples(batch_results)

    # show the results
    img = imread(input_image_path, channel_order='rgb')
    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        draw_bbox=True,
        kpt_thr=0.3,
        draw_heatmap=True,
        show_kpt_idx=True,
        skeleton_style='mmpose',
        show=False,
        out_file=output_image_path)