# This script confirms that the mmlab packages are installed correctly.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# base path is relative to the location of this script
base_path = os.path.dirname(__file__)

def test_mmengine():
    import mmengine

    assert hasattr(mmengine, '__version__'), "mmengine is not installed correctly"
    assert isinstance(mmengine.__version__, str), "mmengine version is not a string"
    assert len(mmengine.__version__) > 0, "mmengine version string is empty"

    print(f"mmengine version: {mmengine.__version__}")

def test_mmpretrain():
    from mmengine.fileio import dump
    from rich import print_json

    from mmpretrain.apis import ImageClassificationInferencer

    input_image_path = os.path.join(base_path, "pretrain_demo.jpg")
    # Base path is relative to the location of this script
    output_image_path = os.path.join(base_path, 'mmpretrain/')

    model = "resnet18_8xb32_in1k"

    # build the model from a config file and a checkpoint file
    try:
        pretrained = True
        inferencer = ImageClassificationInferencer(
            model, pretrained=pretrained)
    except ValueError:
        raise ValueError(
            f'Unavailable model "{model}", you can specify a model '
            'name or a config file or find a model name from '
            'https://mmpretrain.readthedocs.io/en/latest/modelzoo_statistics.html#all-checkpoints'  # noqa: E501
        )
    result = inferencer(input_image_path, show=False, show_dir=output_image_path)[0]
    # show the results
    result.pop('pred_scores')  # pred_scores is too verbose for a demo.
    print_json(dump(result, file_format='json', indent=4))

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

def test_mmdetection_demo():
    from mim import download
    from mmdet.apis import DetInferencer
    from mmdet.utils import register_all_modules

    # Register all modules from mmdet
    register_all_modules()

    package = 'mmdet'

    config_file = 'rtmdet_tiny_8xb32-300e_coco'
    checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

    # Base path is relative to the location of this script
    destination = os.path.join(base_path, 'mmdetection/')

    # Download the model and checkpoints
    download(package, [config_file], dest_root=destination)

    # Define the model config and checkpoints paths
    cfg_path = os.path.join(destination, f"{config_file}.py")
    ckpt_path = os.path.join(destination, checkpoint_file)

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

def test_mmpose_demo():
    from mim import download
    from mmcv.image import imread
    from mmpose.apis import init_model, inference_topdown
    from mmpose.registry import VISUALIZERS
    from mmpose.structures import merge_data_samples
    from mmpose.utils import register_all_modules

    # Register all modules from mmpose
    register_all_modules()

    package = 'mmpose'

    # Define the model config and checkpoint files
    pose_config_id = "td-hm_hrnet-w48_8xb32-210e_coco-256x192"
    pose_checkpoint = "td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth"

    # define the destination folder
    destination = os.path.join(base_path, f"mmpose/")

    # download the model and checkpoints
    download(package, [pose_config_id], dest_root=destination)

    # define the model config and checkpoints paths
    cfg_path = os.path.join(destination, f"{pose_config_id}.py")
    ckpt_path = os.path.join(destination, pose_checkpoint)

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