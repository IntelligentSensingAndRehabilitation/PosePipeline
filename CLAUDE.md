# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PosePipe is a DataJoint-based human pose estimation pipeline for clinical movement analysis from videos. It wraps 20+ state-of-the-art algorithms (tracking, 2D/3D pose estimation, SMPL body models, hand pose) behind a unified DataJoint schema with ~49 tables. Part of the ISR lab's isr-containers monorepo, installed as an editable package.

## Commands

```bash
# Install
pip install -e .
pip install -e ".[dev]"          # with test/dev dependencies
bash scripts/mmlab_install_mim.sh  # required for OpenMMlab algorithms

# Tests (require DataJoint DB + GPU, run inside Docker)
pytest tests/
pytest tests/test_models.py         # model loading (MeTRAbs, MMPose)
pytest tests/test_mmlab_packages.py # MMlab package availability

# Formatting
black --line-length=120 --extend-exclude='3rdparty/*' .
```

## Architecture

### DataJoint Pipeline Flow

The pipeline is a directed graph of DataJoint tables defined in `pose_pipeline/pipeline.py`. Data flows through these layers:

1. **Video Layer**: `Video` (manual) → `VideoInfo` (computed metadata: fps, dimensions, frame count)
2. **Tracking Layer**: `TrackingBboxMethod` → `TrackingBbox` → `PersonBboxValid` (manual annotation) → `PersonBbox` (cleaned/interpolated boxes)
3. **2D Pose Layer** (two paths):
   - **Top-Down**: `TopDownMethod` → `TopDownPerson` — runs pose estimation on cropped person boxes
   - **Bottom-Up**: `BottomUpMethod` → `BottomUpPeople` → `BottomUpPerson` — detects all people then matches to tracks
4. **3D Pose Layer** (two paths):
   - **Lifting**: `LiftingMethod` → `LiftingPerson` — 2D→3D via GastNet/VideoPose3D/PoseAug
   - **SMPL**: `SMPLMethod` → `SMPLPerson` — body model fitting via VIBE/MEVA/ProHMR/PARE/PIXIE/HybrIK
5. **Hand Pose Layer**: `HandBboxMethod` → `HandBbox` → `HandPoseEstimationMethod` → `HandPoseEstimation`
6. **Bridging**: `BottomUpBridging` uses MeTRAbs to convert between skeleton formats (COCO_25, bml_movi_87, smpl+head_30, smplx_42, etc.)

Each layer follows the **Lookup → Manual → Computed** pattern: a Lookup table defines available methods, a Manual table selects which method to use for a video, and a Computed table runs the algorithm via `.populate()`.

### Key Modules

- **`pose_pipeline/pipeline.py`** (~2500 lines) — All DataJoint table definitions and their `.make()` methods. This is the core of the package.
- **`pose_pipeline/wrappers/`** — Algorithm-specific wrappers (mmpose, mmdet, bridging/MeTRAbs, sapiens, vibe, pare, hybrik, etc.). Each wrapper is lazily imported to minimize startup dependencies.
- **`pose_pipeline/utils/standard_pipelines.py`** — High-level convenience functions (`tracking_pipeline`, `top_down_pipeline`, `lifting_pipeline`, `smpl_pipeline`, `hand_estimation_pipeline`) that chain multiple table `.populate()` calls.
- **`pose_pipeline/utils/`** — Visualization, keypoint matching (IOU-based), bounding box utilities, SMPL joint name definitions.
- **`pose_pipeline/env.py`** — `add_path()` context manager, `set_environmental_variables()`, GPU memory limiters for PyTorch/TensorFlow.

### DataJoint Conventions

- Results are stored as NumPy arrays in `longblob` fields
- Video files use `@localattach` store protocol (configured via `dj.config['stores']`)
- Schema name: `pose_pipeline` (supports database prefix via `dj.config['custom']`)
- Tables support `reserve_jobs=True` for distributed processing
- Video visualization tables (e.g., `TopDownPersonVideo`, `SMPLPersonVideo`) generate overlay videos as computed outputs

### Method Names

Algorithm selection uses string-based method names matched through Lookup tables. Examples:
- Tracking: `MMDet_deepsort`, `DeepSortYOLOv4`, `MMDet_qdtrack`
- Top-Down: `MMpose`, `RTMPose`, `ViTPose`, `Bridging_bml_movi_87`, `Sapiens_1b_goliath`
- Bottom-Up: `OpenPose_BODY25B`, `Bridging_COCO_25`, `Bridging_OpenPose`
- Lifting: `GastNet`, `VideoPose3D`, `Bridging_bml_movi_87`
- SMPL: `VIBE`, `PIXIE`, `PARE`, `HybrIK`, `ProHMR`

### Integration with ISR Ecosystem

- Consumed by **PipelineOrchestrator** for YAML-based workflow automation
- **BodyModels** provides SMPL/SMPLx skeleton definitions used downstream
- **MultiCameraTracking** handles multi-camera setups that feed into this pipeline
- Shares DataJoint MySQL database with other ISR schemas
