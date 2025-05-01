# [PosePipe: Open-Source Human Pose Estimation Pipeline for Clinical Research](https://arxiv.org/abs/2203.08792)

![Entity Relationship Diagram](https://github.com/IntelligentSensingAndRehabilitation/PosePipeline/blob/main/doc/erd.png)

PosePipe is a human pose estimation (HPE) pipeline designed to facilitate home movement analysis from videos.  
It uses [DataJoint](https://github.com/datajoint) to manage relationships between algorithms, videos, and intermediate outputs.

Key features:
- Modular wrappers for numerous state-of-the-art HPE algorithms
- Structured video and data management via DataJoint
- Output visualizations to easily compare and analyze results
- Designed for clinical research and home movement analysis pipelines

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

---

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