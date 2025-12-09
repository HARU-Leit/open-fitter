"""Template avatar data fallback generator.

This module generates fallback avatar data for the Template avatar when
avatar_data_template.json is not available.

The bone name patterns are derived from HeuristicBoneMapper.cs in modular-avatar:
https://github.com/bdunderscore/modular-avatar
Copyright (c) 2022 bd_ (licensed under MIT License)

Which in turn incorporates patterns from:
- https://github.com/HhotateA/AvatarModifyTools
  Copyright (c) 2021 @HhotateA_xR (MIT License)
- https://github.com/Azukimochi/BoneRenamer
  Copyright (c) 2023 Azukimochi (MIT License)
"""

from typing import Dict, List, Any, Optional


# Standard humanoid bone name patterns
# First entry is the canonical humanoid name, rest are common aliases
# Based on HeuristicBoneMapper.cs (MIT License)
BONE_NAME_PATTERNS = {
    "Hips": ["Hips", "Hip", "pelvis"],
    "LeftUpperLeg": ["LeftUpperLeg", "UpperLeg_Left", "UpperLeg_L", "Leg_Left", "Leg_L", "ULeg_L", "Left leg", "LeftUpLeg", "UpLeg.L", "Thigh_L", "Upper_Leg_L"],
    "RightUpperLeg": ["RightUpperLeg", "UpperLeg_Right", "UpperLeg_R", "Leg_Right", "Leg_R", "ULeg_R", "Right leg", "RightUpLeg", "UpLeg.R", "Thigh_R", "Upper_Leg_R"],
    "LeftLowerLeg": ["LeftLowerLeg", "LowerLeg_Left", "LowerLeg_L", "Knee_Left", "Knee_L", "LLeg_L", "Left knee", "LeftLeg", "leg_L", "shin.L", "Lower_Leg_L"],
    "RightLowerLeg": ["RightLowerLeg", "LowerLeg_Right", "LowerLeg_R", "Knee_Right", "Knee_R", "LLeg_R", "Right knee", "RightLeg", "leg_R", "shin.R", "Lower_Leg_R"],
    "LeftFoot": ["LeftFoot", "Foot_Left", "Foot_L", "Ankle_L", "Foot.L.001", "Left ankle", "heel.L"],
    "RightFoot": ["RightFoot", "Foot_Right", "Foot_R", "Ankle_R", "Foot.R.001", "Right ankle", "heel.R"],
    "Spine": ["Spine", "spine01"],
    "Chest": ["Chest", "Bust", "spine02", "upper_chest"],
    "Neck": ["Neck"],
    "Head": ["Head"],
    "LeftShoulder": ["LeftShoulder", "Shoulder_Left", "Shoulder_L"],
    "RightShoulder": ["RightShoulder", "Shoulder_Right", "Shoulder_R"],
    "LeftUpperArm": ["LeftUpperArm", "UpperArm_Left", "UpperArm_L", "Arm_Left", "Arm_L", "UArm_L", "Left arm", "UpperLeftArm", "Upper_Arm_L"],
    "RightUpperArm": ["RightUpperArm", "UpperArm_Right", "UpperArm_R", "Arm_Right", "Arm_R", "UArm_R", "Right arm", "UpperRightArm", "Upper_Arm_R"],
    "LeftLowerArm": ["LeftLowerArm", "LowerArm_Left", "LowerArm_L", "LArm_L", "Left elbow", "LeftForeArm", "Elbow_L", "forearm_L", "ForArm_L", "Lower_Arm_L"],
    "RightLowerArm": ["RightLowerArm", "LowerArm_Right", "LowerArm_R", "LArm_R", "Right elbow", "RightForeArm", "Elbow_R", "forearm_R", "ForArm_R", "Lower_Arm_R"],
    "LeftHand": ["LeftHand", "Hand_Left", "Hand_L", "Left wrist", "Wrist_L"],
    "RightHand": ["RightHand", "Hand_Right", "Hand_R", "Right wrist", "Wrist_R"],
    "LeftToes": ["LeftToes", "Toes_Left", "Toe_Left", "ToeIK_L", "Toes_L", "Toe_L", "Foot.L.002", "Left Toe", "LeftToeBase"],
    "RightToes": ["RightToes", "Toes_Right", "Toe_Right", "ToeIK_R", "Toes_R", "Toe_R", "Foot.R.002", "Right Toe", "RightToeBase"],
    "LeftEye": ["LeftEye", "Eye_Left", "Eye_L"],
    "RightEye": ["RightEye", "Eye_Right", "Eye_R"],
    "Jaw": ["Jaw"],
    # Thumb
    "LeftThumbProximal": ["LeftThumbProximal", "ProximalThumb_Left", "ProximalThumb_L", "Thumb1_L", "ThumbFinger1_L", "LeftHandThumb1", "Thumb Proximal.L", "Thunb1_L", "finger01_01_L", "Thumb_Proximal_L"],
    "LeftThumbIntermediate": ["LeftThumbIntermediate", "IntermediateThumb_Left", "IntermediateThumb_L", "Thumb2_L", "ThumbFinger2_L", "LeftHandThumb2", "Thumb Intermediate.L", "Thunb2_L", "finger01_02_L", "Thumb_Intermediate_L"],
    "LeftThumbDistal": ["LeftThumbDistal", "DistalThumb_Left", "DistalThumb_L", "Thumb3_L", "ThumbFinger3_L", "LeftHandThumb3", "Thumb Distal.L", "Thunb3_L", "finger01_03_L", "Thumb_Distal_L"],
    "RightThumbProximal": ["RightThumbProximal", "ProximalThumb_Right", "ProximalThumb_R", "Thumb1_R", "ThumbFinger1_R", "RightHandThumb1", "Thumb Proximal.R", "Thunb1_R", "finger01_01_R", "Thumb_Proximal_R"],
    "RightThumbIntermediate": ["RightThumbIntermediate", "IntermediateThumb_Right", "IntermediateThumb_R", "Thumb2_R", "ThumbFinger2_R", "RightHandThumb2", "Thumb Intermediate.R", "Thunb2_R", "finger01_02_R", "Thumb_Intermediate_R"],
    "RightThumbDistal": ["RightThumbDistal", "DistalThumb_Right", "DistalThumb_R", "Thumb3_R", "ThumbFinger3_R", "RightHandThumb3", "Thumb Distal.R", "Thunb3_R", "finger01_03_R", "Thumb_Distal_R"],
    # Index
    "LeftIndexProximal": ["LeftIndexProximal", "ProximalIndex_Left", "ProximalIndex_L", "Index1_L", "IndexFinger1_L", "LeftHandIndex1", "Index Proximal.L", "finger02_01_L", "f_index.01.L", "Index_Proximal_L"],
    "LeftIndexIntermediate": ["LeftIndexIntermediate", "IntermediateIndex_Left", "IntermediateIndex_L", "Index2_L", "IndexFinger2_L", "LeftHandIndex2", "Index Intermediate.L", "finger02_02_L", "f_index.02.L", "Index_Intermediate_L"],
    "LeftIndexDistal": ["LeftIndexDistal", "DistalIndex_Left", "DistalIndex_L", "Index3_L", "IndexFinger3_L", "LeftHandIndex3", "Index Distal.L", "finger02_03_L", "f_index.03.L", "Index_Distal_L"],
    "RightIndexProximal": ["RightIndexProximal", "ProximalIndex_Right", "ProximalIndex_R", "Index1_R", "IndexFinger1_R", "RightHandIndex1", "Index Proximal.R", "finger02_01_R", "f_index.01.R", "Index_Proximal_R"],
    "RightIndexIntermediate": ["RightIndexIntermediate", "IntermediateIndex_Right", "IntermediateIndex_R", "Index2_R", "IndexFinger2_R", "RightHandIndex2", "Index Intermediate.R", "finger02_02_R", "f_index.02.R", "Index_Intermediate_R"],
    "RightIndexDistal": ["RightIndexDistal", "DistalIndex_Right", "DistalIndex_R", "Index3_R", "IndexFinger3_R", "RightHandIndex3", "Index Distal.R", "finger02_03_R", "f_index.03.R", "Index_Distal_R"],
    # Middle
    "LeftMiddleProximal": ["LeftMiddleProximal", "ProximalMiddle_Left", "ProximalMiddle_L", "Middle1_L", "MiddleFinger1_L", "LeftHandMiddle1", "Middle Proximal.L", "finger03_01_L", "f_middle.01.L", "Middle_Proximal_L"],
    "LeftMiddleIntermediate": ["LeftMiddleIntermediate", "IntermediateMiddle_Left", "IntermediateMiddle_L", "Middle2_L", "MiddleFinger2_L", "LeftHandMiddle2", "Middle Intermediate.L", "finger03_02_L", "f_middle.02.L", "Middle_Intermediate_L"],
    "LeftMiddleDistal": ["LeftMiddleDistal", "DistalMiddle_Left", "DistalMiddle_L", "Middle3_L", "MiddleFinger3_L", "LeftHandMiddle3", "Middle Distal.L", "finger03_03_L", "f_middle.03.L", "Middle_Distal_L"],
    "RightMiddleProximal": ["RightMiddleProximal", "ProximalMiddle_Right", "ProximalMiddle_R", "Middle1_R", "MiddleFinger1_R", "RightHandMiddle1", "Middle Proximal.R", "finger03_01_R", "f_middle.01.R", "Middle_Proximal_R"],
    "RightMiddleIntermediate": ["RightMiddleIntermediate", "IntermediateMiddle_Right", "IntermediateMiddle_R", "Middle2_R", "MiddleFinger2_R", "RightHandMiddle2", "Middle Intermediate.R", "finger03_02_R", "f_middle.02.R", "Middle_Intermediate_R"],
    "RightMiddleDistal": ["RightMiddleDistal", "DistalMiddle_Right", "DistalMiddle_R", "Middle3_R", "MiddleFinger3_R", "RightHandMiddle3", "Middle Distal.R", "finger03_03_R", "f_middle.03.R", "Middle_Distal_R"],
    # Ring
    "LeftRingProximal": ["LeftRingProximal", "ProximalRing_Left", "ProximalRing_L", "Ring1_L", "RingFinger1_L", "LeftHandRing1", "Ring Proximal.L", "finger04_01_L", "f_ring.01.L", "Ring_Proximal_L"],
    "LeftRingIntermediate": ["LeftRingIntermediate", "IntermediateRing_Left", "IntermediateRing_L", "Ring2_L", "RingFinger2_L", "LeftHandRing2", "Ring Intermediate.L", "finger04_02_L", "f_ring.02.L", "Ring_Intermediate_L"],
    "LeftRingDistal": ["LeftRingDistal", "DistalRing_Left", "DistalRing_L", "Ring3_L", "RingFinger3_L", "LeftHandRing3", "Ring Distal.L", "finger04_03_L", "f_ring.03.L", "Ring_Distal_L"],
    "RightRingProximal": ["RightRingProximal", "ProximalRing_Right", "ProximalRing_R", "Ring1_R", "RingFinger1_R", "RightHandRing1", "Ring Proximal.R", "finger04_01_R", "f_ring.01.R", "Ring_Proximal_R"],
    "RightRingIntermediate": ["RightRingIntermediate", "IntermediateRing_Right", "IntermediateRing_R", "Ring2_R", "RingFinger2_R", "RightHandRing2", "Ring Intermediate.R", "finger04_02_R", "f_ring.02.R", "Ring_Intermediate_R"],
    "RightRingDistal": ["RightRingDistal", "DistalRing_Right", "DistalRing_R", "Ring3_R", "RingFinger3_R", "RightHandRing3", "Ring Distal.R", "finger04_03_R", "f_ring.03.R", "Ring_Distal_R"],
    # Little
    "LeftLittleProximal": ["LeftLittleProximal", "ProximalLittle_Left", "ProximalLittle_L", "Little1_L", "LittleFinger1_L", "LeftHandPinky1", "Little Proximal.L", "finger05_01_L", "f_pinky.01.L", "Little_Proximal_L"],
    "LeftLittleIntermediate": ["LeftLittleIntermediate", "IntermediateLittle_Left", "IntermediateLittle_L", "Little2_L", "LittleFinger2_L", "LeftHandPinky2", "Little Intermediate.L", "finger05_02_L", "f_pinky.02.L", "Little_Intermediate_L"],
    "LeftLittleDistal": ["LeftLittleDistal", "DistalLittle_Left", "DistalLittle_L", "Little3_L", "LittleFinger3_L", "LeftHandPinky3", "Little Distal.L", "finger05_03_L", "f_pinky.03.L", "Little_Distal_L"],
    "RightLittleProximal": ["RightLittleProximal", "ProximalLittle_Right", "ProximalLittle_R", "Little1_R", "LittleFinger1_R", "RightHandPinky1", "Little Proximal.R", "finger05_01_R", "f_pinky.01.R", "Little_Proximal_R"],
    "RightLittleIntermediate": ["RightLittleIntermediate", "IntermediateLittle_Right", "IntermediateLittle_R", "Little2_R", "LittleFinger2_R", "RightHandPinky2", "Little Intermediate.R", "finger05_02_R", "f_pinky.02.R", "Little_Intermediate_R"],
    "RightLittleDistal": ["RightLittleDistal", "DistalLittle_Right", "DistalLittle_R", "Little3_R", "LittleFinger3_R", "RightHandPinky3", "Little Distal.R", "finger05_03_R", "f_pinky.03.R", "Little_Distal_R"],
    # UpperChest (often missing)
    "UpperChest": ["UpperChest", "UChest"],
}

# Standard humanoid bone hierarchy (parent -> children)
HUMANOID_HIERARCHY = {
    "Hips": ["Spine", "LeftUpperLeg", "RightUpperLeg"],
    "Spine": ["Chest"],
    "Chest": ["UpperChest", "Neck", "LeftShoulder", "RightShoulder"],
    "UpperChest": [],  # Optional bone, children handled by Chest if missing
    "Neck": ["Head"],
    "Head": ["LeftEye", "RightEye", "Jaw"],
    "LeftShoulder": ["LeftUpperArm"],
    "RightShoulder": ["RightUpperArm"],
    "LeftUpperArm": ["LeftLowerArm"],
    "RightUpperArm": ["RightLowerArm"],
    "LeftLowerArm": ["LeftHand"],
    "RightLowerArm": ["RightHand"],
    "LeftHand": ["LeftThumbProximal", "LeftIndexProximal", "LeftMiddleProximal", "LeftRingProximal", "LeftLittleProximal"],
    "RightHand": ["RightThumbProximal", "RightIndexProximal", "RightMiddleProximal", "RightRingProximal", "RightLittleProximal"],
    "LeftUpperLeg": ["LeftLowerLeg"],
    "RightUpperLeg": ["RightLowerLeg"],
    "LeftLowerLeg": ["LeftFoot"],
    "RightLowerLeg": ["RightFoot"],
    "LeftFoot": ["LeftToes"],
    "RightFoot": ["RightToes"],
    "LeftToes": [],
    "RightToes": [],
    # Fingers
    "LeftThumbProximal": ["LeftThumbIntermediate"],
    "LeftThumbIntermediate": ["LeftThumbDistal"],
    "LeftThumbDistal": [],
    "LeftIndexProximal": ["LeftIndexIntermediate"],
    "LeftIndexIntermediate": ["LeftIndexDistal"],
    "LeftIndexDistal": [],
    "LeftMiddleProximal": ["LeftMiddleIntermediate"],
    "LeftMiddleIntermediate": ["LeftMiddleDistal"],
    "LeftMiddleDistal": [],
    "LeftRingProximal": ["LeftRingIntermediate"],
    "LeftRingIntermediate": ["LeftRingDistal"],
    "LeftRingDistal": [],
    "LeftLittleProximal": ["LeftLittleIntermediate"],
    "LeftLittleIntermediate": ["LeftLittleDistal"],
    "LeftLittleDistal": [],
    "RightThumbProximal": ["RightThumbIntermediate"],
    "RightThumbIntermediate": ["RightThumbDistal"],
    "RightThumbDistal": [],
    "RightIndexProximal": ["RightIndexIntermediate"],
    "RightIndexIntermediate": ["RightIndexDistal"],
    "RightIndexDistal": [],
    "RightMiddleProximal": ["RightMiddleIntermediate"],
    "RightMiddleIntermediate": ["RightMiddleDistal"],
    "RightMiddleDistal": [],
    "RightRingProximal": ["RightRingIntermediate"],
    "RightRingIntermediate": ["RightRingDistal"],
    "RightRingDistal": [],
    "RightLittleProximal": ["RightLittleIntermediate"],
    "RightLittleIntermediate": ["RightLittleDistal"],
    "RightLittleDistal": [],
    # Optional bones
    "LeftEye": [],
    "RightEye": [],
    "Jaw": [],
}

# Core humanoid bones (required for most operations)
CORE_HUMANOID_BONES = [
    "Hips", "Spine", "Chest", "Neck", "Head",
    "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
    "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
    # Fingers
    "LeftThumbProximal", "LeftThumbIntermediate", "LeftThumbDistal",
    "LeftIndexProximal", "LeftIndexIntermediate", "LeftIndexDistal",
    "LeftMiddleProximal", "LeftMiddleIntermediate", "LeftMiddleDistal",
    "LeftRingProximal", "LeftRingIntermediate", "LeftRingDistal",
    "LeftLittleProximal", "LeftLittleIntermediate", "LeftLittleDistal",
    "RightThumbProximal", "RightThumbIntermediate", "RightThumbDistal",
    "RightIndexProximal", "RightIndexIntermediate", "RightIndexDistal",
    "RightMiddleProximal", "RightMiddleIntermediate", "RightMiddleDistal",
    "RightRingProximal", "RightRingIntermediate", "RightRingDistal",
    "RightLittleProximal", "RightLittleIntermediate", "RightLittleDistal",
]


def normalize_bone_name(name: str) -> str:
    """Normalize a bone name for comparison.
    
    Based on HeuristicBoneMapper.NormalizeName()
    """
    import re
    name = name.lower()
    name = re.sub(r'^bone_|[0-9 ._]', '', name)
    return name


def find_matching_bone_name(bone_name: str, patterns: Dict[str, List[str]]) -> Optional[str]:
    """Find the humanoid bone name that matches a given bone name.
    
    Args:
        bone_name: The bone name to look up
        patterns: Dictionary mapping humanoid names to pattern lists
        
    Returns:
        The humanoid bone name if found, None otherwise
    """
    normalized = normalize_bone_name(bone_name)
    
    for humanoid_name, pattern_list in patterns.items():
        for pattern in pattern_list:
            if normalize_bone_name(pattern) == normalized:
                return humanoid_name
    
    return None


def get_preferred_bone_name(humanoid_bone: str, style: str = "underscore") -> str:
    """Get the preferred bone name for a humanoid bone.
    
    Args:
        humanoid_bone: The humanoid bone name (e.g., "LeftUpperLeg")
        style: Naming style - "underscore" (Upper_Leg_L), "camel" (LeftUpperLeg)
        
    Returns:
        The preferred bone name in the requested style
    """
    if style == "camel":
        return humanoid_bone
    
    # Special cases for Template avatar naming conventions
    SPECIAL_MAPPINGS = {
        "LeftToes": "Toe_L",
        "RightToes": "Toe_R",
    }
    if humanoid_bone in SPECIAL_MAPPINGS:
        return SPECIAL_MAPPINGS[humanoid_bone]
    
    # Convert to underscore style (common in many avatars)
    # LeftUpperLeg -> Upper_Leg_L
    import re
    
    # Handle left/right prefix
    if humanoid_bone.startswith("Left"):
        suffix = "_L"
        core = humanoid_bone[4:]
    elif humanoid_bone.startswith("Right"):
        suffix = "_R"
        core = humanoid_bone[5:]
    else:
        suffix = ""
        core = humanoid_bone
    
    # Insert underscores before capital letters
    # UpperLeg -> Upper_Leg
    result = re.sub(r'([a-z])([A-Z])', r'\1_\2', core)
    
    return result + suffix


def generate_humanoid_bones_mapping(style: str = "underscore") -> List[Dict[str, str]]:
    """Generate humanoid bones mapping list.
    
    Args:
        style: Bone naming style
        
    Returns:
        List of {humanoidBoneName, boneName} dictionaries
    """
    result = []
    for humanoid_bone in CORE_HUMANOID_BONES:
        result.append({
            "humanoidBoneName": humanoid_bone,
            "boneName": get_preferred_bone_name(humanoid_bone, style)
        })
    
    # Add UpperChest as optional (Template uses dummy bone)
    result.append({
        "humanoidBoneName": "UpperChest",
        "boneName": "UpperChest(Dummy)"
    })
    
    return result


def _build_hierarchy_node(bone_name: str, humanoid_to_actual: Dict[str, str]) -> Dict[str, Any]:
    """Build a hierarchy node recursively.
    
    Args:
        bone_name: Humanoid bone name
        humanoid_to_actual: Mapping from humanoid name to actual bone name
        
    Returns:
        Hierarchy node dictionary
    """
    actual_name = humanoid_to_actual.get(bone_name, bone_name)
    children = []
    
    if bone_name in HUMANOID_HIERARCHY:
        for child in HUMANOID_HIERARCHY[bone_name]:
            if child in humanoid_to_actual or child in HUMANOID_HIERARCHY:
                children.append(_build_hierarchy_node(child, humanoid_to_actual))
    
    return {
        "name": actual_name,
        "children": children
    }


def generate_bone_hierarchy(style: str = "underscore") -> Dict[str, Any]:
    """Generate bone hierarchy structure.
    
    Args:
        style: Bone naming style
        
    Returns:
        Bone hierarchy dictionary
    """
    # Build mapping from humanoid name to actual name
    humanoid_to_actual = {}
    for humanoid_bone in CORE_HUMANOID_BONES:
        humanoid_to_actual[humanoid_bone] = get_preferred_bone_name(humanoid_bone, style)
    humanoid_to_actual["UpperChest"] = "UpperChest(Dummy)"
    
    # Build hierarchy starting from root (usually armature contains Hips)
    root_name = "Template"  # Root object name
    armature_name = "Armature.Template"
    
    hips_hierarchy = _build_hierarchy_node("Hips", humanoid_to_actual)
    
    return {
        "name": root_name,
        "children": [
            {
                "name": armature_name,
                "children": [hips_hierarchy]
            }
        ]
    }


def generate_template_avatar_data() -> Dict[str, Any]:
    """Generate complete Template avatar data.
    
    Returns:
        Dictionary containing all avatar data fields
    """
    return {
        "name": "Template",
        "defaultFBXPath": None,  # No FBX file needed for fallback
        "meshName": "Body.Template",
        "basePose": "__TEMPLATE_FALLBACK__",
        "basePoseA": "__TEMPLATE_FALLBACK__",
        "blendshapes": [],  # No blendshapes in fallback mode
        "humanoidBones": generate_humanoid_bones_mapping("underscore"),
        "boneHierarchy": generate_bone_hierarchy("underscore"),
        "auxiliaryBones": [],
        "commonSwaySettings": {
            "startDistance": 0.025,
            "endDistance": 0.050
        },
        "swayBones": [],
        "boneComponents": [],
        "shrinkBlendShapes": [],
        "blendShapeGroups": [],
        "blendShapeFields": [],
        "invertedBlendShapeFields": []
    }


# Cached fallback data
_TEMPLATE_AVATAR_DATA = None


def get_template_avatar_data() -> Dict[str, Any]:
    """Get cached Template avatar data.
    
    Returns:
        Dictionary containing Template avatar data
    """
    global _TEMPLATE_AVATAR_DATA
    if _TEMPLATE_AVATAR_DATA is None:
        _TEMPLATE_AVATAR_DATA = generate_template_avatar_data()
    return _TEMPLATE_AVATAR_DATA


def is_template_avatar_data_path(path: str) -> bool:
    """Check if a path refers to Template avatar data.
    
    Args:
        path: File path to check
        
    Returns:
        True if this is a Template avatar data path
    """
    if not path:
        return False
    
    import os
    basename = os.path.basename(path).lower()
    return "avatar_data_template" in basename or basename == "template"


# For testing
if __name__ == "__main__":
    import json
    
    data = generate_template_avatar_data()
    print(json.dumps(data, indent=2))
