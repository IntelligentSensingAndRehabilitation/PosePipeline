# [PosePipe: Open-Source Human Pose Estimation Pipeline for Clinical Research](https://arxiv.org/abs/2203.08792)

![Entity Relationship Diagram](https://github.com/IntelligentSensingAndRehabilitation/PosePipeline/blob/main/doc/erd.png?raw=True)

PosePipe is a human pose estimation (HPE) pipeline designed to facilitate movement analysis from videos.  
It uses [DataJoint](https://github.com/datajoint) to manage relationships between algorithms, videos, and intermediate outputs.

Key features:
- Modular wrappers for numerous state-of-the-art HPE algorithms
- Structured video and data management via DataJoint
- Output visualizations to easily compare and analyze results
- Designed for clinical research movement analysis pipelines

---

## Quick Start

1. **Install PosePipe**

```bash
pip install pose_pipeline
```

Detailed [installation instructions](https://github.com/IntelligentSensingAndRehabilitation/PosePipeline/blob/main/INSTALL.md)
are provided to launch a DataJoint MySQL database and install OpenMMLab packages.

2. **Test the pipeline**

Use the [Getting Started Notebook](https://github.com/IntelligentSensingAndRehabilitation/PosePipeline/blob/main/doc/Getting_Started.ipynb) to start running your videos through the pose estimation framework.

## Recent Updates and Supported Algorithms 

- **Upgraded mmcv to v2.x**
- **Tracking Algorithms (from mmdetection):**
  - [DeepSORT](https://github.com/open-mmlab/mmdetection/tree/main/configs/deepsort)
  - [QDTrack](https://github.com/open-mmlab/mmdetection/tree/main/configs/qdtrack)
- **Top Down 2D Body Keypoint Detection Algorithms (from mmpose):**
  - [RTMPose + RTMPose on Coco-Wholebody](https://github.com/open-mmlab/mmpose/tree/dev-1.x/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py)
  - [RTMPose + RTMW on Cocktail14](https://github.com/open-mmlab/mmpose/tree/dev-1.x/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-l_8xb320-270e_cocktail14-384x288.py)
  - [Topdown Heatmap + ViTPose on Coco](https://github.com/open-mmlab/mmpose/tree/dev-1.x/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py)
- **Top Down 2D Hand Keypoint Detection Algorithms (from mmpose):**
  - [RTMPose + RTMPose on Hand5](https://github.com/open-mmlab/mmpose/blob/main/configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py)
  - [RTMPose + RTMPose + Coco + Wholebody + Hand on Coco_wholebody_hand](https://github.com/open-mmlab/mmpose/blob/main/configs/hand_2d_keypoint/rtmpose/coco_wholebody_hand/rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256.py)
  - [Topdown Heatmap + Resnet on Freihand2d](https://github.com/open-mmlab/mmpose/blob/main/configs/hand_2d_keypoint/topdown_heatmap/freihand2d/td-hm_res50_8xb64-100e_freihand2d-224x224.py)
  - [Topdown Heatmap + HRNetv2 + Dark on Rhd2d](https://github.com/open-mmlab/mmpose/blob/main/configs/hand_2d_keypoint/topdown_heatmap/rhd2d/td-hm_hrnetv2-w18_dark-8xb64-210e_rhd2d-256x256.py)
  - [Topdown Heatmap + HRNetv2 + UDP on OneHand10k](https://github.com/open-mmlab/mmpose/blob/main/configs/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_hrnetv2-w18_udp-8xb64-210e_onehand10k-256x256.py)
- **Bottom Up Algorithms:**
  - [MeTRAbs](https://github.com/isarandi/metrabs)

## Developer Setup

VSCode is recommended for development.

Include the following in your `.vscode/settings.json` to enable consistent `black` formatting:

```json
{
  "python.formatting.blackArgs": [
    "--line-length=120",
    "--include='*py'",
    "--exclude='*ipynb'",
    "--extend-exclude='.env'",
    "--extend-exclude='3rdparty/*'"
  ],
  "editor.rulers": [120]
}
```

---

## Project Info

- **License:** GPL-3.0
- **Source Code:** [GitHub Repo](https://github.com/IntelligentSensingAndRehabilitation/PosePipeline/tree/main)
- **PyPI:** [https://pypi.org/project/posepipe](https://pypi.org/project/posepipe)
- **Issues/Contributions:** Please use [Issues](https://github.com/IntelligentSensingAndRehabilitation/PosePipeline/issues) for bug reports and feature requests

---

## Citation

If you use this tool for research, please cite:

```
@misc{posepipe2024,
  author       = {R James Cotton},
  title        = {PosePipe: Open-Source Human Pose Estimation Pipeline for Clinical Research},
  year         = {2024},
  howpublished = {\url{https://github.com/IntelligentSensingAndRehabilitation/PosePipeline}}
}
```