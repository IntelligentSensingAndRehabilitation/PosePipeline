# This script confirms that the mmlab packages are installed correctly.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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