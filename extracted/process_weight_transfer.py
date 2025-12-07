import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import math
import time
from collections import deque

import bmesh
import bpy
import mathutils
import numpy as np
from algo_utils.get_humanoid_and_auxiliary_bone_groups import (
    get_humanoid_and_auxiliary_bone_groups,
)
from blender_utils.build_bone_maps import build_bone_maps
from blender_utils.reset_bone_weights import reset_bone_weights
from io_utils.restore_shape_key_state import restore_shape_key_state
from io_utils.restore_weights import restore_weights
from io_utils.save_shape_key_state import save_shape_key_state
from io_utils.store_weights import store_weights
from stages.compute_non_humanoid_masks import compute_non_humanoid_masks
from stages.merge_added_groups import merge_added_groups
from stages.run_distance_normal_smoothing import run_distance_normal_smoothing
from stages.apply_distance_falloff_blend import apply_distance_falloff_blend
from stages.restore_head_weights import restore_head_weights
from stages.apply_metadata_fallback import apply_metadata_fallback
from stages.compare_side_and_bone_weights import compare_side_and_bone_weights
from stages.detect_finger_vertices import detect_finger_vertices
from stages.create_closing_filter_mask import create_closing_filter_mask
from stages.prepare_groups_and_weights import prepare_groups_and_weights
from stages.attempt_weight_transfer import attempt_weight_transfer
from stages.transfer_side_weights import transfer_side_weights
from stages.smooth_and_cleanup import smooth_and_cleanup
from stages.store_intermediate_results import store_intermediate_results
from stages.blend_results import blend_results
from stages.adjust_hands_and_propagate import adjust_hands_and_propagate


class WeightTransferContext:
    """Stateful context to orchestrate weight transfer without changing external IO."""

    def __init__(self, target_obj, armature, base_avatar_data, clothing_avatar_data, field_path, clothing_armature, cloth_metadata=None):
        self.target_obj = target_obj
        self.armature = armature
        self.base_avatar_data = base_avatar_data
        self.clothing_avatar_data = clothing_avatar_data
        self.field_path = field_path
        self.clothing_armature = clothing_armature
        self.cloth_metadata = cloth_metadata
        self.start_time = time.time()

        self.humanoid_to_bone = {}
        self.bone_to_humanoid = {}
        self.auxiliary_bones = {}
        self.auxiliary_bones_to_humanoid = {}
        self.finger_humanoid_bones = [
            "LeftIndexProximal", "LeftIndexIntermediate", "LeftIndexDistal",
            "LeftMiddleProximal", "LeftMiddleIntermediate", "LeftMiddleDistal",
            "LeftRingProximal", "LeftRingIntermediate", "LeftRingDistal",
            "LeftLittleProximal", "LeftLittleIntermediate", "LeftLittleDistal",
            "RightIndexProximal", "RightIndexIntermediate", "RightIndexDistal",
            "RightMiddleProximal", "RightMiddleIntermediate", "RightMiddleDistal",
            "RightRingProximal", "RightRingIntermediate", "RightRingDistal",
            "RightLittleProximal", "RightLittleIntermediate", "RightLittleDistal",
            "LeftHand", "RightHand",
        ]
        self.left_foot_finger_humanoid_bones = [
            "LeftFootThumbProximal",
            "LeftFootThumbIntermediate",
            "LeftFootThumbDistal",
            "LeftFootIndexProximal",
            "LeftFootIndexIntermediate",
            "LeftFootIndexDistal",
            "LeftFootMiddleProximal",
            "LeftFootMiddleIntermediate",
            "LeftFootMiddleDistal",
            "LeftFootRingProximal",
            "LeftFootRingIntermediate",
            "LeftFootRingDistal",
            "LeftFootLittleProximal",
            "LeftFootLittleIntermediate",
            "LeftFootLittleDistal",
        ]
        self.right_foot_finger_humanoid_bones = [
            "RightFootThumbProximal",
            "RightFootThumbIntermediate",
            "RightFootThumbDistal",
            "RightFootIndexProximal",
            "RightFootIndexIntermediate",
            "RightFootIndexDistal",
            "RightFootMiddleProximal",
            "RightFootMiddleIntermediate",
            "RightFootMiddleDistal",
            "RightFootRingProximal",
            "RightFootRingIntermediate",
            "RightFootRingDistal",
            "RightFootLittleProximal",
            "RightFootLittleIntermediate",
            "RightFootLittleDistal",
        ]

        self.finger_bone_names = set()
        self.finger_vertices = set()
        self.closing_filter_mask_weights = None
        self.original_groups = set()
        self.bone_groups = set()
        self.all_deform_groups = set()
        self.original_non_humanoid_groups = set()
        self.original_humanoid_weights = {}
        self.original_non_humanoid_weights = {}
        self.all_weights = {}
        self.new_groups = set()
        self.added_groups = set()
        self.non_humanoid_parts_mask = None
        self.non_humanoid_total_weights = None
        self.non_humanoid_difference_mask = None
        self.distance_falloff_group = None
        self.distance_falloff_group2 = None
        self.non_humanoid_difference_group = None
        self.weights_a = {}
        self.weights_b = {}

    def _build_bone_maps(self):
        """ヒューマノイドボーンと補助ボーンのマッピングを構築する。"""
        (
            self.humanoid_to_bone,
            self.bone_to_humanoid,
            self.auxiliary_bones,
            self.auxiliary_bones_to_humanoid,
        ) = build_bone_maps(self.base_avatar_data)

    def detect_finger_vertices(self):
        detect_finger_vertices(self)

    def create_closing_filter_mask(self):
        create_closing_filter_mask(self)

    def attempt_weight_transfer(self, source_obj, vertex_group, max_distance_try=0.2, max_distance_tried=0.0):
        return attempt_weight_transfer(self, source_obj, vertex_group, max_distance_try, max_distance_tried)



    def prepare_groups_and_weights(self):
        prepare_groups_and_weights(self)

    def transfer_side_weights(self):
        return transfer_side_weights(self)

    def _process_mf_group(self, group_name, temp_shape_name, rotation_deg, humanoid_label_left, humanoid_label_right):
        target_group = self.target_obj.vertex_groups.get(group_name)
        should_process = False
        if target_group:
            for vert in self.target_obj.data.vertices:
                for g in vert.groups:
                    if g.group == target_group.index and g.weight > 0.001:
                        should_process = True
                        break
                if should_process:
                    break

        if not should_process:
            print(f"  {group_name}グループが存在しないか、有効なウェイトがないため処理をスキップ")
            return

        if not (self.armature and self.armature.type == "ARMATURE"):
            print(f"  {group_name}グループが存在しないか、アーマチュアが存在しないため処理をスキップ")
            return

        print(f"  {group_name}グループが存在し、有効なウェイトを持つため処理を実行")
        base_humanoid_weights = store_weights(self.target_obj, self.bone_groups)
        reset_bone_weights(self.target_obj, self.bone_groups)
        restore_weights(self.target_obj, self.all_weights)

        print(f"  {humanoid_label_left}と{humanoid_label_right}ボーンにY軸回転を適用")
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode="POSE")

        left_bone = None
        right_bone = None
        for bone_map in self.base_avatar_data.get("humanoidBones", []):
            if bone_map.get("humanoidBoneName") == humanoid_label_left:
                left_bone = bone_map.get("boneName")
            elif bone_map.get("humanoidBoneName") == humanoid_label_right:
                right_bone = bone_map.get("boneName")

        if left_bone and left_bone in self.armature.pose.bones:
            bone = self.armature.pose.bones[left_bone]
            current_world_matrix = self.armature.matrix_world @ bone.matrix
            head_world_transformed = self.armature.matrix_world @ bone.head
            offset_matrix = mathutils.Matrix.Translation(head_world_transformed * -1.0)
            rotation_matrix = mathutils.Matrix.Rotation(math.radians(rotation_deg * -1), 4, "Y")
            bone.matrix = self.armature.matrix_world.inverted() @ offset_matrix.inverted() @ rotation_matrix @ offset_matrix @ current_world_matrix

        if right_bone and right_bone in self.armature.pose.bones:
            bone = self.armature.pose.bones[right_bone]
            current_world_matrix = self.armature.matrix_world @ bone.matrix
            head_world_transformed = self.armature.matrix_world @ bone.head
            offset_matrix = mathutils.Matrix.Translation(head_world_transformed * -1.0)
            rotation_matrix = mathutils.Matrix.Rotation(math.radians(rotation_deg), 4, "Y")
            bone.matrix = self.armature.matrix_world.inverted() @ offset_matrix.inverted() @ rotation_matrix @ offset_matrix @ current_world_matrix

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.context.view_layer.objects.active = self.target_obj
        bpy.context.view_layer.update()

        shape_key_state = save_shape_key_state(self.target_obj)
        for key_block in self.target_obj.data.shape_keys.key_blocks:
            key_block.value = 0.0

        if self.target_obj.data.shape_keys and temp_shape_name in self.target_obj.data.shape_keys.key_blocks:
            temp_shape_key = self.target_obj.data.shape_keys.key_blocks[temp_shape_name]
            temp_shape_key.value = 1.0
        else:
            temp_shape_key = None

        reset_bone_weights(self.target_obj, self.bone_groups)
        print("  ウェイト転送開始")
        self.attempt_weight_transfer(bpy.data.objects["Body.BaseAvatar"], "BothSideWeights")

        restore_shape_key_state(self.target_obj, shape_key_state)
        if temp_shape_key:
            temp_shape_key.value = 0.0

        print(f"  {humanoid_label_left}と{humanoid_label_right}ボーンにY軸逆回転を適用")
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode="POSE")

        if left_bone and left_bone in self.armature.pose.bones:
            bone = self.armature.pose.bones[left_bone]
            current_world_matrix = self.armature.matrix_world @ bone.matrix
            head_world_transformed = self.armature.matrix_world @ bone.head
            offset_matrix = mathutils.Matrix.Translation(head_world_transformed * -1.0)
            rotation_matrix = mathutils.Matrix.Rotation(math.radians(rotation_deg), 4, "Y")
            bone.matrix = self.armature.matrix_world.inverted() @ offset_matrix.inverted() @ rotation_matrix @ offset_matrix @ current_world_matrix

        if right_bone and right_bone in self.armature.pose.bones:
            bone = self.armature.pose.bones[right_bone]
            current_world_matrix = self.armature.matrix_world @ bone.matrix
            head_world_transformed = self.armature.matrix_world @ bone.head
            offset_matrix = mathutils.Matrix.Translation(head_world_transformed * -1.0)
            rotation_matrix = mathutils.Matrix.Rotation(math.radians(rotation_deg * -1), 4, "Y")
            bone.matrix = self.armature.matrix_world.inverted() @ offset_matrix.inverted() @ rotation_matrix @ offset_matrix @ current_world_matrix

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.context.view_layer.objects.active = self.target_obj
        bpy.context.view_layer.update()

        target_group = self.target_obj.vertex_groups.get(group_name)
        if target_group and base_humanoid_weights:
            print("  ウェイト合成処理開始")
            for vert in self.target_obj.data.vertices:
                vert_idx = vert.index
                target_weight = 0.0
                for g in vert.groups:
                    if g.group == target_group.index:
                        target_weight = g.weight
                        break
                current_factor = target_weight
                base_factor = 1.0 - target_weight
                for group_name in self.bone_groups:
                    if group_name in self.target_obj.vertex_groups:
                        group = self.target_obj.vertex_groups[group_name]
                        current_weight = 0.0
                        for g in vert.groups:
                            if g.group == group.index:
                                current_weight = g.weight
                                break
                        base_weight = 0.0
                        if vert_idx in base_humanoid_weights and group_name in base_humanoid_weights[vert_idx]:
                            base_weight = base_humanoid_weights[vert_idx][group_name]
                        blended_weight = current_weight * current_factor + base_weight * base_factor
                        if blended_weight > 0.0001:
                            group.add([vert_idx], blended_weight, "REPLACE")
                            base_humanoid_weights[vert_idx][group_name] = blended_weight
                        else:
                            try:
                                group.remove([vert_idx])
                                base_humanoid_weights[vert_idx][group_name] = 0.0
                            except RuntimeError:
                                pass
            print("  ウェイト合成処理完了")

    def run_armpit_process(self):
        self._process_mf_group("MF_Armpit", "WT_shape_forA.MFTemp", 45, "LeftUpperArm", "RightUpperArm")

    def run_crotch_process(self):
        self._process_mf_group("MF_crotch", "WT_shape_forCrotch.MFTemp", 70, "LeftUpperLeg", "RightUpperLeg")

    def smooth_and_cleanup(self):
        smooth_and_cleanup(self)

    def compute_non_humanoid_masks(self):
        compute_non_humanoid_masks(self)

    def merge_added_groups(self):
        merge_added_groups(self)

    def store_intermediate_results(self):
        store_intermediate_results(self)

    def blend_results(self):
        blend_results(self)

    def adjust_hands_and_propagate(self):
        adjust_hands_and_propagate(self)

    def compare_side_and_bone_weights(self):
        compare_side_and_bone_weights(self)

    def run_distance_normal_smoothing(self):
        run_distance_normal_smoothing(self)

    def apply_distance_falloff_blend(self):
        apply_distance_falloff_blend(self)

    def restore_head_weights(self):
        restore_head_weights(self)

    def apply_metadata_fallback(self):
        apply_metadata_fallback(self)

    def run(self):
        print(f"処理開始: {self.target_obj.name}")
        self._build_bone_maps()
        self.detect_finger_vertices()
        self.create_closing_filter_mask()
        self.prepare_groups_and_weights()
        if not self.transfer_side_weights():
            return
        self.run_armpit_process()
        self.run_crotch_process()
        self.smooth_and_cleanup()
        self.compute_non_humanoid_masks()
        self.merge_added_groups()
        self.store_intermediate_results()
        self.blend_results()
        self.adjust_hands_and_propagate()
        self.compare_side_and_bone_weights()
        self.run_distance_normal_smoothing()
        self.apply_distance_falloff_blend()
        self.restore_head_weights()
        self.apply_metadata_fallback()
        total_time = time.time() - self.start_time
        print(f"処理完了: {self.target_obj.name} - 合計時間: {total_time:.2f}秒")


def process_weight_transfer(target_obj, armature, base_avatar_data, clothing_avatar_data, field_path, clothing_armature, cloth_metadata=None):
    """Orchestrator that delegates weight transfer to a stateful context."""
    context = WeightTransferContext(
        target_obj=target_obj,
        armature=armature,
        base_avatar_data=base_avatar_data,
        clothing_avatar_data=clothing_avatar_data,
        field_path=field_path,
        clothing_armature=clothing_armature,
        cloth_metadata=cloth_metadata,
    )
    context.run()
