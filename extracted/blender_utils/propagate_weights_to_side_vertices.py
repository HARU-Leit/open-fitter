import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bmesh


def propagate_weights_to_side_vertices(
    target_obj,
    bone_groups,
    original_humanoid_weights,
    clothing_armature,
    max_iterations=100
):
    """
    側面ウェイトを持つがボーンウェイトを持たない頂点にウェイトを伝播する。
    
    Args:
        target_obj: 対象のBlenderオブジェクト
        bone_groups: ボーングループ名のセット
        original_humanoid_weights: 元のヒューマノイドウェイト辞書
        clothing_armature: 衣装アーマチュア（省略可）
        max_iterations: 最大反復回数
    """
    bm = bmesh.new()
    bm.from_mesh(target_obj.data)
    bm.verts.ensure_lookup_table()

    left_group = target_obj.vertex_groups.get("LeftSideWeights")
    right_group = target_obj.vertex_groups.get("RightSideWeights")

    all_deform_groups = set(bone_groups)
    if clothing_armature:
        all_deform_groups.update(bone.name for bone in clothing_armature.data.bones)

    def get_side_weight(vert_idx, group):
        if not group:
            return 0.0
        try:
            for g in target_obj.data.vertices[vert_idx].groups:
                if g.group == group.index:
                    return g.weight
        except Exception:
            return 0.0
        return 0.0

    def has_bone_weights(vert_idx):
        for g in target_obj.data.vertices[vert_idx].groups:
            if target_obj.vertex_groups[g.group].name in all_deform_groups:
                return True
        return False

    vertices_to_process = set()
    for vert in target_obj.data.vertices:
        if (get_side_weight(vert.index, left_group) > 0 or get_side_weight(vert.index, right_group) > 0) and not has_bone_weights(vert.index):
            vertices_to_process.add(vert.index)

    if not vertices_to_process:
        bm.free()
        return

    print(f"Found {len(vertices_to_process)} vertices without bone weights but with side weights")

    iteration = 0
    while vertices_to_process and iteration < max_iterations:
        propagated_this_iteration = set()
        for vert_idx in vertices_to_process:
            vert = bm.verts[vert_idx]
            neighbors_with_weights = []
            for edge in vert.link_edges:
                other = edge.other_vert(vert)
                if has_bone_weights(other.index):
                    distance = (vert.co - other.co).length
                    neighbors_with_weights.append((other.index, distance))
            if neighbors_with_weights:
                closest_vert_idx = min(neighbors_with_weights, key=lambda x: x[1])[0]
                for group in target_obj.vertex_groups:
                    if group.name in all_deform_groups:
                        weight = 0.0
                        for g in target_obj.data.vertices[closest_vert_idx].groups:
                            if g.group == group.index:
                                weight = g.weight
                                break
                        if weight > 0:
                            group.add([vert_idx], weight, "REPLACE")
                propagated_this_iteration.add(vert_idx)

        if not propagated_this_iteration:
            break

        print(f"Iteration {iteration + 1}: Propagated weights to {len(propagated_this_iteration)} vertices")
        vertices_to_process -= propagated_this_iteration
        iteration += 1

    if vertices_to_process:
        print(f"Restoring original weights for {len(vertices_to_process)} remaining vertices")
        for vert_idx in vertices_to_process:
            if vert_idx in original_humanoid_weights:
                for group in target_obj.vertex_groups:
                    if group.name in all_deform_groups:
                        try:
                            group.remove([vert_idx])
                        except RuntimeError:
                            continue
                for group_name, weight in original_humanoid_weights[vert_idx].items():
                    if group_name in target_obj.vertex_groups:
                        target_obj.vertex_groups[group_name].add([vert_idx], weight, "REPLACE")

    bm.free()
