import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CURR_DIR)
_GRANDPARENT_DIR = os.path.dirname(_PARENT_DIR)
for _p in (_PARENT_DIR, _GRANDPARENT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from add_clothing_pose_from_json import add_clothing_pose_from_json
from algo_utils.create_hinge_bone_group import create_hinge_bone_group
from algo_utils.remove_empty_vertex_groups import remove_empty_vertex_groups
from apply_blendshape_deformation_fields import apply_blendshape_deformation_fields
from blender_utils.create_deformation_mask import create_deformation_mask
from blender_utils.create_overlapping_vertices_attributes import (
    create_overlapping_vertices_attributes,
)
from blender_utils.merge_auxiliary_to_humanoid_weights import (
    merge_auxiliary_to_humanoid_weights,
)
from blender_utils.process_bone_weight_consolidation import (
    process_bone_weight_consolidation,
)
from blender_utils.propagate_bone_weights import propagate_bone_weights
from blender_utils.reset_shape_keys import reset_shape_keys
from blender_utils.subdivide_breast_faces import subdivide_breast_faces
from blender_utils.subdivide_long_edges import subdivide_long_edges
from blender_utils.triangulate_mesh import triangulate_mesh
from io_utils.restore_vertex_weights import restore_vertex_weights
from io_utils.save_vertex_weights import save_vertex_weights
from math_utils.normalize_vertex_weights import normalize_vertex_weights
from process_mesh_with_connected_components_inline import (
    process_mesh_with_connected_components_inline,
)


class MeshPreparationStage:
    """Executes mesh preparation (cycle 1) responsibilities."""

    def __init__(self, processor):
        self.processor = processor

    def run(self):
        def _run(self):
            time = self.time_module

            print("Status: BlendShape用 Deformation Field適用中")
            print(
                f"Progress: {(self.pair_index + 0.33) / self.total_pairs * 0.9:.3f}"
            )
            self.blend_shape_labels = (
                self.config_pair['blend_shapes'].split(',')
                if self.config_pair['blend_shapes']
                else None
            )
            if self.blend_shape_labels:
                for obj in self.clothing_meshes:
                    reset_shape_keys(obj)
                    remove_empty_vertex_groups(obj)
                    normalize_vertex_weights(obj)
                    apply_blendshape_deformation_fields(
                        obj,
                        self.config_pair['field_data'],
                        self.blend_shape_labels,
                        self.clothing_avatar_data,
                        self.config_pair['blend_shape_values'],
                    )
            blendshape_time = time.time()
            print(
                f"BlendShape用 Deformation Field適用: {blendshape_time - self.base_weights_time:.2f}秒"
            )

            print("Status: ポーズ適用中")
            print(
                f"Progress: {(self.pair_index + 0.35) / self.total_pairs * 0.9:.3f}"
            )
            add_clothing_pose_from_json(
                self.clothing_armature,
                self.config_pair['pose_data'],
                self.config_pair['init_pose'],
                self.config_pair['clothing_avatar_data'],
                self.config_pair['base_avatar_data'],
            )
            pose_time = time.time()
            print(f"ポーズ適用: {pose_time - blendshape_time:.2f}秒")

            print("Status: 重複頂点属性設定中")
            print(
                f"Progress: {(self.pair_index + 0.4) / self.total_pairs * 0.9:.3f}"
            )
            create_overlapping_vertices_attributes(
                self.clothing_meshes, self.base_avatar_data
            )
            vertices_attributes_time = time.time()
            print(
                f"重複頂点属性設定: {vertices_attributes_time - pose_time:.2f}秒"
            )

            for obj in self.clothing_meshes:
                create_hinge_bone_group(
                    obj, self.clothing_armature, self.clothing_avatar_data
                )

            print("Status: メッシュ変形処理中")
            print(
                f"Progress: {(self.pair_index + 0.45) / self.total_pairs * 0.9:.3f}"
            )
            self.propagated_groups_map = {}
            field_distance_groups = {}
            cycle1_start = time.time()
            for obj in self.clothing_meshes:
                obj_start = time.time()
                print("cycle1 " + obj.name)

                reset_shape_keys(obj)
                remove_empty_vertex_groups(obj)
                normalize_vertex_weights(obj)
                merge_auxiliary_to_humanoid_weights(
                    obj, self.clothing_avatar_data
                )

                temp_group_name = propagate_bone_weights(obj)
                if temp_group_name:
                    self.propagated_groups_map[obj.name] = temp_group_name

                cleanup_weights_time_start = time.time()
                for vert in obj.data.vertices:
                    groups_to_remove = []
                    for g in vert.groups:
                        if g.weight < 0.0005:
                            groups_to_remove.append(g.group)
                    for group_idx in groups_to_remove:
                        try:
                            obj.vertex_groups[group_idx].remove([vert.index])
                        except RuntimeError:
                            continue
                cleanup_weights_time = time.time() - cleanup_weights_time_start
                print(f"  微小ウェイト除外: {cleanup_weights_time:.2f}秒")

                create_deformation_mask(obj, self.clothing_avatar_data)

                if (
                    self.pair_index == 0
                    and self.use_subdivision
                    and obj.name not in self.cloth_metadata
                ):
                    subdivide_long_edges(obj)
                    subdivide_breast_faces(obj, self.clothing_avatar_data)

                if (
                    self.use_triangulation
                    and not self.use_subdivision
                    and obj.name not in self.cloth_metadata
                    and self.pair_index == self.total_pairs - 1
                ):
                    triangulate_mesh(obj)

                original_weights = save_vertex_weights(obj)

                process_bone_weight_consolidation(
                    obj, self.clothing_avatar_data
                )

                process_mesh_with_connected_components_inline(
                    obj,
                    self.config_pair['field_data'],
                    self.blend_shape_labels,
                    self.clothing_avatar_data,
                    self.base_avatar_data,
                    self.clothing_armature,
                    self.cloth_metadata,
                    subdivision=self.use_subdivision,
                    skip_blend_shape_generation=self.config_pair[
                        'skip_blend_shape_generation'
                    ],
                    config_data=self.config_pair['config_data'],
                )

                restore_vertex_weights(obj, original_weights)

                if obj.data.shape_keys:
                    generated_shape_keys = []
                    for shape_key in obj.data.shape_keys.key_blocks:
                        if shape_key.name.endswith("_generated"):
                            generated_shape_keys.append(shape_key.name)

                    for generated_name in generated_shape_keys:
                        base_name = generated_name[:-10]
                        generated_key = obj.data.shape_keys.key_blocks.get(
                            generated_name
                        )
                        base_key = obj.data.shape_keys.key_blocks.get(base_name)

                        if generated_key and base_key:
                            for i, point in enumerate(generated_key.data):
                                base_key.data[i].co = point.co
                            print(
                                f"Merged {generated_name} into {base_name} for {obj.name}"
                            )
                            obj.shape_key_remove(generated_key)
                            print(
                                f"Removed generated shape key: {generated_name} from {obj.name}"
                            )

                print(f"  {obj.name}の処理: {time.time() - obj_start:.2f}秒")

            cycle1_end = time.time()
            self.cycle1_end_time = cycle1_end
            print(f"サイクル1全体: {cycle1_end - cycle1_start:.2f}秒")

            for obj in self.clothing_meshes:
                if obj.data.shape_keys:
                    for key_block in obj.data.shape_keys.key_blocks:
                        print(
                            f"Shape key: {key_block.name} / {key_block.value} found on {obj.name}"
                        )

        _run(self.processor)
