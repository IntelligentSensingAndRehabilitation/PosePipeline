# MHR (Momentum Human Rig) Utilities for PosePipeline
# Used with SAM-3D-Body model outputs

import numpy as np

# MHR 70 joint names (inference output)
MHR70_JOINT_NAMES = [
    "nose",                     # 0
    "left-eye",                 # 1
    "right-eye",                # 2
    "left-ear",                 # 3
    "right-ear",                # 4
    "left-shoulder",            # 5
    "right-shoulder",           # 6
    "left-elbow",               # 7
    "right-elbow",              # 8
    "left-hip",                 # 9
    "right-hip",                # 10
    "left-knee",                # 11
    "right-knee",               # 12
    "left-ankle",               # 13
    "right-ankle",              # 14
    "left-big-toe-tip",         # 15
    "left-small-toe-tip",       # 16
    "left-heel",                # 17
    "right-big-toe-tip",        # 18
    "right-small-toe-tip",      # 19
    "right-heel",               # 20
    "right-thumb-tip",          # 21
    "right-thumb-first-joint",  # 22
    "right-thumb-second-joint", # 23
    "right-thumb-third-joint",  # 24
    "right-index-tip",          # 25
    "right-index-first-joint",  # 26
    "right-index-second-joint", # 27
    "right-index-third-joint",  # 28
    "right-middle-tip",         # 29
    "right-middle-first-joint", # 30
    "right-middle-second-joint",# 31
    "right-middle-third-joint", # 32
    "right-ring-tip",           # 33
    "right-ring-first-joint",   # 34
    "right-ring-second-joint",  # 35
    "right-ring-third-joint",   # 36
    "right-pinky-tip",          # 37
    "right-pinky-first-joint",  # 38
    "right-pinky-second-joint", # 39
    "right-pinky-third-joint",  # 40
    "right-wrist",              # 41
    "left-thumb-tip",           # 42
    "left-thumb-first-joint",   # 43
    "left-thumb-second-joint",  # 44
    "left-thumb-third-joint",   # 45
    "left-index-tip",           # 46
    "left-index-first-joint",   # 47
    "left-index-second-joint",  # 48
    "left-index-third-joint",   # 49
    "left-middle-tip",          # 50
    "left-middle-first-joint",  # 51
    "left-middle-second-joint", # 52
    "left-middle-third-joint",  # 53
    "left-ring-tip",            # 54
    "left-ring-first-joint",    # 55
    "left-ring-second-joint",   # 56
    "left-ring-third-joint",    # 57
    "left-pinky-tip",           # 58
    "left-pinky-first-joint",   # 59
    "left-pinky-second-joint",  # 60
    "left-pinky-third-joint",   # 61
    "left-wrist",               # 62
    "left-olecranon",           # 63
    "right-olecranon",          # 64
    "left-cubital-fossa",       # 65
    "right-cubital-fossa",      # 66
    "left-acromion",            # 67
    "right-acromion",           # 68
    "neck",                     # 69
]

# Joint index lookup
MHR70_JOINT_INDEX = {name: idx for idx, name in enumerate(MHR70_JOINT_NAMES)}

# Semantic joint groups
MHR70_BODY_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 69]  # Face + torso + limbs
MHR70_LEFT_FOOT_JOINTS = [15, 16, 17]  # big toe, small toe, heel
MHR70_RIGHT_FOOT_JOINTS = [18, 19, 20]  # big toe, small toe, heel
MHR70_RIGHT_HAND_JOINTS = list(range(21, 42))  # thumb to pinky + wrist
MHR70_LEFT_HAND_JOINTS = list(range(42, 63))   # thumb to pinky + wrist
MHR70_EXTRA_JOINTS = [63, 64, 65, 66, 67, 68]  # olecranon, cubital fossa, acromion

# COCO 17 keypoint mapping from MHR70
# COCO format: nose, left_eye, right_eye, left_ear, right_ear,
#              left_shoulder, right_shoulder, left_elbow, right_elbow,
#              left_wrist, right_wrist, left_hip, right_hip,
#              left_knee, right_knee, left_ankle, right_ankle
MHR70_TO_COCO17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]


def get_joint_index(name):
    """Get joint index by name."""
    return MHR70_JOINT_INDEX.get(name)


def get_joint_name(index):
    """Get joint name by index."""
    if 0 <= index < len(MHR70_JOINT_NAMES):
        return MHR70_JOINT_NAMES[index]
    return None


def extract_body_keypoints(keypoints):
    """
    Extract main body keypoints from MHR70 keypoints.

    Args:
        keypoints: Array of shape [..., 70, 2] or [..., 70, 3]

    Returns:
        Body keypoints subset
    """
    return keypoints[..., MHR70_BODY_JOINTS, :]


def extract_hand_keypoints(keypoints, hand='left'):
    """
    Extract hand keypoints from MHR70 keypoints.

    Args:
        keypoints: Array of shape [..., 70, 2] or [..., 70, 3]
        hand: 'left' or 'right'

    Returns:
        Hand keypoints (21 joints)
    """
    if hand == 'left':
        return keypoints[..., MHR70_LEFT_HAND_JOINTS, :]
    else:
        return keypoints[..., MHR70_RIGHT_HAND_JOINTS, :]


def extract_foot_keypoints(keypoints, foot='left'):
    """
    Extract foot keypoints from MHR70 keypoints.

    Args:
        keypoints: Array of shape [..., 70, 2] or [..., 70, 3]
        foot: 'left' or 'right'

    Returns:
        Foot keypoints (3 joints: big toe, small toe, heel)
    """
    if foot == 'left':
        return keypoints[..., MHR70_LEFT_FOOT_JOINTS, :]
    else:
        return keypoints[..., MHR70_RIGHT_FOOT_JOINTS, :]


def mhr70_to_coco17(keypoints):
    """
    Convert MHR70 keypoints to COCO 17 format.

    Args:
        keypoints: Array of shape [..., 70, D] where D is 2 or 3

    Returns:
        COCO17 keypoints of shape [..., 17, D]
    """
    return keypoints[..., MHR70_TO_COCO17, :]


def compute_pelvis(keypoints_3d):
    """
    Compute pelvis position as midpoint of hips.

    Args:
        keypoints_3d: Array of shape [..., 70, 3]

    Returns:
        Pelvis position of shape [..., 3]
    """
    left_hip = keypoints_3d[..., MHR70_JOINT_INDEX['left-hip'], :]
    right_hip = keypoints_3d[..., MHR70_JOINT_INDEX['right-hip'], :]
    return (left_hip + right_hip) / 2


def compute_neck(keypoints_3d):
    """
    Get neck position from MHR70 keypoints.

    Args:
        keypoints_3d: Array of shape [..., 70, 3]

    Returns:
        Neck position of shape [..., 3]
    """
    return keypoints_3d[..., MHR70_JOINT_INDEX['neck'], :]


def compute_spine_length(keypoints_3d):
    """
    Compute spine length (pelvis to neck distance).

    Args:
        keypoints_3d: Array of shape [..., 70, 3]

    Returns:
        Spine length of shape [...]
    """
    pelvis = compute_pelvis(keypoints_3d)
    neck = compute_neck(keypoints_3d)
    return np.linalg.norm(neck - pelvis, axis=-1)


def get_limb_pairs():
    """
    Get pairs of joint indices for drawing skeleton.

    Returns:
        List of (joint1_idx, joint2_idx) tuples
    """
    return [
        # Head
        (0, 1), (0, 2), (1, 3), (2, 4),
        # Torso
        (5, 6), (5, 9), (6, 10), (9, 10),
        # Arms
        (5, 7), (7, 62), (6, 8), (8, 41),
        # Legs
        (9, 11), (11, 13), (10, 12), (12, 14),
        # Feet
        (13, 15), (13, 17), (14, 18), (14, 20),
        # Neck
        (5, 69), (6, 69), (0, 69),
    ]
