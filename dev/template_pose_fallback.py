"""Template avatar fallback pose data generator.

This module generates fallback pose data for the Template avatar when
pose_basis_template.json is not available.
"""

import math
from typing import Dict, Any


def _create_identity_delta_matrix() -> list:
    """Create a 4x4 identity matrix as nested lists."""
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _create_z_rotation_delta_matrix(degrees: float) -> list:
    """Create a delta_matrix with Z-axis rotation.
    
    This represents a finger spread rotation around the local Z axis.
    
    Args:
        degrees: Rotation angle in degrees (positive = spread outward)
    
    Returns:
        4x4 matrix as nested lists
    """
    radians = math.radians(degrees)
    cos_r = math.cos(radians)
    sin_r = math.sin(radians)
    
    return [
        [cos_r, -sin_r, 0.0, 0.0],
        [sin_r, cos_r, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _create_bone_pose_data(delta_matrix: list) -> Dict[str, Any]:
    """Create a pose data entry for a single bone."""
    return {
        "location": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "delta_matrix": delta_matrix,
    }


def generate_template_fallback_pose() -> Dict[str, Dict[str, Any]]:
    """Generate fallback pose data for Template avatar.
    
    This function generates mathematically derived pose data for the
    Template avatar's finger bones. The values are based on standard
    finger spread angles:
    - Index finger: 5° spread (toward thumb)
    - Ring finger: 5° spread (away from thumb)
    - Little finger: 10° spread (away from thumb)
    
    Left hand fingers spread in negative Z direction.
    Right hand fingers spread in positive Z direction (mirrored).
    
    Returns:
        Dictionary containing pose data for each finger bone
    """
    pose_data = {}
    
    # Identity matrix for most bones
    identity = _create_identity_delta_matrix()
    
    # Standard humanoid bones with identity transform
    standard_bones = [
        "Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
        "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
        "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
        "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
        "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
        # Thumb bones (no spread)
        "LeftThumbProximal", "LeftThumbIntermediate", "LeftThumbDistal",
        "RightThumbProximal", "RightThumbIntermediate", "RightThumbDistal",
        # Middle finger (center, no spread)
        "LeftMiddleProximal", "LeftMiddleIntermediate", "LeftMiddleDistal",
        "RightMiddleProximal", "RightMiddleIntermediate", "RightMiddleDistal",
    ]
    
    for bone in standard_bones:
        pose_data[bone] = _create_bone_pose_data(identity)
    
    # Finger spread angles (in degrees)
    # These are generic pose values, not proprietary data
    INDEX_SPREAD = 5.0   # Index finger spreads toward thumb
    RING_SPREAD = 5.0    # Ring finger spreads away from thumb
    LITTLE_SPREAD = 10.0  # Little finger spreads more away from thumb
    
    # Left hand - negative Z rotation for spread toward thumb (index)
    # positive Z rotation for spread away from thumb (ring, little)
    left_index_matrix = _create_z_rotation_delta_matrix(-INDEX_SPREAD)
    left_ring_matrix = _create_z_rotation_delta_matrix(RING_SPREAD)
    left_little_matrix = _create_z_rotation_delta_matrix(LITTLE_SPREAD)
    
    # Right hand - mirrored (opposite sign)
    right_index_matrix = _create_z_rotation_delta_matrix(INDEX_SPREAD)
    right_ring_matrix = _create_z_rotation_delta_matrix(-RING_SPREAD)
    right_little_matrix = _create_z_rotation_delta_matrix(-LITTLE_SPREAD)
    
    # Left Index finger
    for bone in ["LeftIndexProximal", "LeftIndexIntermediate", "LeftIndexDistal"]:
        pose_data[bone] = _create_bone_pose_data(left_index_matrix)
    
    # Left Ring finger
    for bone in ["LeftRingProximal", "LeftRingIntermediate", "LeftRingDistal"]:
        pose_data[bone] = _create_bone_pose_data(left_ring_matrix)
    
    # Left Little finger
    for bone in ["LeftLittleProximal", "LeftLittleIntermediate", "LeftLittleDistal"]:
        pose_data[bone] = _create_bone_pose_data(left_little_matrix)
    
    # Right Index finger
    for bone in ["RightIndexProximal", "RightIndexIntermediate", "RightIndexDistal"]:
        pose_data[bone] = _create_bone_pose_data(right_index_matrix)
    
    # Right Ring finger
    for bone in ["RightRingProximal", "RightRingIntermediate", "RightRingDistal"]:
        pose_data[bone] = _create_bone_pose_data(right_ring_matrix)
    
    # Right Little finger
    for bone in ["RightLittleProximal", "RightLittleIntermediate", "RightLittleDistal"]:
        pose_data[bone] = _create_bone_pose_data(right_little_matrix)
    
    return pose_data


# Cached fallback data
_TEMPLATE_FALLBACK_POSE = None


def get_template_fallback_pose() -> Dict[str, Dict[str, Any]]:
    """Get cached fallback pose data for Template avatar.
    
    Returns:
        Dictionary containing pose data for Template avatar bones
    """
    global _TEMPLATE_FALLBACK_POSE
    if _TEMPLATE_FALLBACK_POSE is None:
        _TEMPLATE_FALLBACK_POSE = generate_template_fallback_pose()
    return _TEMPLATE_FALLBACK_POSE
