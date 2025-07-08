# This script confirms that the mmlab packages are installed correctly.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

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

# def test_mmdetection():
#     from mim import download 
#     import mmdet.apis
#     from mmdet.utils import register_all_modules

#     os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

#     package = 'mmdet'

#     config_file = 'rtmdet_tiny_8xb32-300e_coco'
#     checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

#     # Define the destination folder
#     destination = os.path.join(os.getcwd(), 'mmdetection/')

#     # Download the model and checkpoints
#     download(package, [config_file], dest_root=destination)

#     # Define the model config and checkpoints paths
#     model_config = os.path.join(destination, f"{config_file}.py")
#     detector_checkpoint = os.path.join(destination, checkpoint_file)
#     # Register all modules from mmdet
#     register_all_modules()

#     # Initialize the model
#     model = mmdet.apis.init_detector(model_config, detector_checkpoint, device='cuda:0')
#     # Perform inference
#     result = mmdet.apis.inference_mot(model, 'demo/demo.jpg')

#     # model = init_detector(config_file, checkpoint_file, device='cpu')
#     # inference_detector(model, 'demo/demo.jpg')

#     # model = init_detector(config_file, checkpoint_file, device='cuda:0')
#     # inference_detector(model, 'demo/demo.jpg')

def test_mmdetection():
    from mim import download 
    import mmdet.apis
    from mmdet.utils import register_all_modules
    from mmdet.apis import init_detector, inference_detector

    package = 'mmdet'

    config_file = 'rtmdet_tiny_8xb32-300e_coco'
    checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

    # Base path is relative to the location of this script
    base_path = os.path.dirname(__file__)
    input_image_path = os.path.join(base_path, "demo.jpg")

    # Define the destination folder relative to the location of this script
    destination = os.path.join(os.path.dirname(__file__), 'mmdetection/')

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
    base_path = os.path.dirname(__file__)

    # cfg_path = "mmdetection/rtmdet_tiny_8xb32-300e_coco.py"
    cfg_path = os.path.join(base_path, "mmdetection/rtmdet_tiny_8xb32-300e_coco.py")
    # ckpt_path = "mmdetection/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
    ckpt_path = os.path.join(base_path, "mmdetection/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth")

    inferencer = DetInferencer(
        model=cfg_path,
        weights=ckpt_path,
        device='cuda:0',
        palette="none",
    )

    input_image_path = os.path.join(base_path, "demo.jpg")
    output_image_path = os.path.join(base_path, "test_output")

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