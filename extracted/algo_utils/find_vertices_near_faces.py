import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import math

import bmesh
import bpy
from math_utils.barycentric_coords_from_point import barycentric_coords_from_point
from mathutils.bvhtree import BVHTree


def _capture_selection_state():
    return {
        "active": bpy.context.active_object,
        "selected": list(bpy.context.selected_objects),
        "mode": bpy.context.mode,
    }


def _restore_selection_state(state):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in state["selected"]:
        obj.select_set(True)
    if state["active"]:
        bpy.context.view_layer.objects.active = state["active"]
    if state["mode"] and state["mode"].startswith('EDIT'):
        bpy.ops.object.mode_set(mode='EDIT')


def _find_vertex_group(mesh_obj, vertex_group_name):
    for vg in mesh_obj.vertex_groups:
        if vg.name == vertex_group_name:
            return vg
    return None


def _duplicate_and_triangulate_base(base_mesh):
    bpy.ops.object.select_all(action='DESELECT')
    base_mesh.select_set(True)
    bpy.context.view_layer.objects.active = base_mesh
    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.duplicate()
    temp_base_mesh = bpy.context.active_object
    temp_base_mesh.name = f"{base_mesh.name}_temp_triangulated"

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.object.mode_set(mode='OBJECT')

    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_base_mesh = temp_base_mesh.evaluated_get(depsgraph)
    return temp_base_mesh, depsgraph, evaluated_base_mesh.data, evaluated_base_mesh.matrix_world


def _collect_vertices_in_group(mesh_data, group_index):
    vertices_in_group = set()
    for vertex_idx, vertex in enumerate(mesh_data.vertices):
        for group_elem in vertex.groups:
            if group_elem.group == group_index and group_elem.weight > 0.001:
                vertices_in_group.add(vertex_idx)
                break
    return vertices_in_group


def _select_target_faces(mesh_data, vertices_in_group, use_all_faces):
    if use_all_faces:
        return [face.index for face in mesh_data.polygons]

    target_faces = []
    for face in mesh_data.polygons:
        if all(vertex_idx in vertices_in_group for vertex_idx in face.vertices):
            target_faces.append(face.index)
    return target_faces


def _build_bvh(base_mesh_data, base_world_matrix, target_face_indices):
    temp_bm = bmesh.new()
    temp_bm.from_mesh(base_mesh_data)
    temp_bm.faces.ensure_lookup_table()
    temp_bm.verts.ensure_lookup_table()

    vertices = []
    faces = []

    for vert in temp_bm.verts:
        vertices.append(base_world_matrix @ vert.co)

    for face_idx in target_face_indices:
        face = temp_bm.faces[face_idx]
        faces.append([v.index for v in face.verts])

    bvh = BVHTree.FromPolygons(vertices, faces) if faces else None
    return bvh, vertices, faces, temp_bm


def _interpolate_weights(
    target_mesh_data,
    target_world_matrix,
    max_distance,
    max_angle_degrees,
    bvh,
    faces,
    vertices,
    temp_base_vertex_group,
    base_mesh_data,
):
    vertex_interpolated_weights = {}
    found_vertices = []

    if not bvh:
        return vertex_interpolated_weights, found_vertices

    for vertex_idx, vertex in enumerate(target_mesh_data.vertices):
        world_vertex_pos = target_world_matrix @ vertex.co
        nearest_point, normal, face_idx, distance = bvh.find_nearest(world_vertex_pos)

        if max_angle_degrees is not None and normal is not None and nearest_point is not None:
            v = (world_vertex_pos - nearest_point).normalized()
            angle = math.degrees(math.acos(min(1.0, max(-1.0, v.dot(normal)))))
            if angle > max_angle_degrees:
                vertex_interpolated_weights[vertex_idx] = 0.0
                continue

        if nearest_point is None or face_idx is None or distance > max_distance:
            vertex_interpolated_weights[vertex_idx] = 0.0
            continue

        found_vertices.append(vertex_idx)
        face_vertex_indices = faces[face_idx]
        face_vertices = [vertices[vi] for vi in face_vertex_indices]
        bary_coords = barycentric_coords_from_point(
            nearest_point, face_vertices[0], face_vertices[1], face_vertices[2]
        )

        weights = []
        for vi in face_vertex_indices:
            base_vert = base_mesh_data.vertices[vi]
            vert_weight = 0.0
            for group_elem in base_vert.groups:
                if group_elem.group == temp_base_vertex_group.index:
                    vert_weight = group_elem.weight
                    break
            weights.append(vert_weight)

        interpolated_weight = (
            bary_coords[0] * weights[0]
            + bary_coords[1] * weights[1]
            + bary_coords[2] * weights[2]
        )
        vertex_interpolated_weights[vertex_idx] = max(0.0, min(1.0, interpolated_weight))

    return vertex_interpolated_weights, found_vertices


def _ensure_target_vertex_group(target_mesh, vertex_group_name):
    if vertex_group_name in target_mesh.vertex_groups:
        target_mesh.vertex_groups.remove(target_mesh.vertex_groups[vertex_group_name])
    return target_mesh.vertex_groups.new(name=vertex_group_name)


def _apply_weights_to_target(target_mesh_data, target_vertex_group, vertex_interpolated_weights):
    for vertex_idx in range(len(target_mesh_data.vertices)):
        weight = vertex_interpolated_weights.get(vertex_idx, 0.0)
        target_vertex_group.add([vertex_idx], weight, 'REPLACE')


def _set_active_vertex_group(target_mesh, vertex_group_name):
    for i, group in enumerate(target_mesh.vertex_groups):
        target_mesh.vertex_groups.active_index = i
        if group.name == vertex_group_name:
            break


def _build_and_interpolate(
    base_mesh_data,
    base_world_matrix,
    target_face_indices,
    target_mesh_data,
    target_world_matrix,
    max_distance,
    max_angle_degrees,
    temp_base_vertex_group,
):
    import time

    print("BVHTreeを使用して高速検索を実行中...")
    start_time = time.time()
    bvh, vertices, faces, temp_bm = _build_bvh(base_mesh_data, base_world_matrix, target_face_indices)

    if not bvh:
        print("警告: 対象となる面が見つかりません")
        temp_bm.free()
        return None, None

    vertex_interpolated_weights, found_vertices = _interpolate_weights(
        target_mesh_data,
        target_world_matrix,
        max_distance,
        max_angle_degrees,
        bvh,
        faces,
        vertices,
        temp_base_vertex_group,
        base_mesh_data,
    )

    temp_bm.free()
    end_time = time.time()
    print(f"BVHTree検索完了: {end_time - start_time:.3f}秒")
    return vertex_interpolated_weights, found_vertices


def find_vertices_near_faces(base_mesh, target_mesh, vertex_group_name, max_distance=1.0, max_angle_degrees=None, use_all_faces=False,  smooth_repeat=3):
    """
    ベースメッシュの特定の頂点グループに属する面から指定距離内にあるターゲットメッシュの頂点を見つける、法線の方向を考慮する
    
    Args:
        base_mesh: ベースメッシュオブジェクト（面を構成する頂点が属する頂点グループを持つ）
        target_mesh: ターゲットメッシュオブジェクト（検索対象の頂点を持つ）
        vertex_group_name (str): 検索対象の頂点グループ名（両メッシュで共通）
        max_distance (float): 最大距離
        max_angle_degrees (float): 最大角度 (度)、Noneの場合は法線の方向を考慮しない
        use_all_faces (bool): すべての面を使用するかどうか
        smooth_repeat (int): スムージングの繰り返し回数
    """
    
    if not base_mesh or base_mesh.type != 'MESH':
        print("エラー: ベースメッシュが指定されていないか、メッシュではありません")
        return

    if not target_mesh or target_mesh.type != 'MESH':
        print("エラー: ターゲットメッシュが指定されていないか、メッシュではありません")
        return

    base_vertex_group = _find_vertex_group(base_mesh, vertex_group_name)
    if not base_vertex_group:
        print(f"エラー: ベースメッシュに頂点グループ '{vertex_group_name}' が見つかりません")
        return

    selection_state = _capture_selection_state()
    print(f"ベースメッシュ '{base_mesh.name}' の頂点グループ '{vertex_group_name}' に属する面を分析中...")

    print("ベースメッシュを複製して三角面化中...")
    temp_base_mesh, depsgraph, base_mesh_data, base_world_matrix = _duplicate_and_triangulate_base(base_mesh)
    temp_base_mesh_removed = False

    try:
        temp_base_vertex_group = _find_vertex_group(temp_base_mesh, vertex_group_name)
        if not temp_base_vertex_group:
            print(f"エラー: 複製メッシュに頂点グループ '{vertex_group_name}' が見つかりません")
            return

        base_vertices_in_group = _collect_vertices_in_group(base_mesh_data, temp_base_vertex_group.index)
        print(f"頂点グループに属する頂点数: {len(base_vertices_in_group)}")

        target_face_indices = _select_target_faces(base_mesh_data, base_vertices_in_group, use_all_faces)
        print(f"条件を満たす面数: {len(target_face_indices)} (すべて三角形)")

        if not target_face_indices:
            print("警告: 条件を満たす面が見つかりません")
            return

        target_vertex_group = _ensure_target_vertex_group(target_mesh, vertex_group_name)

        evaluated_target_mesh = target_mesh.evaluated_get(depsgraph)
        target_mesh_data = evaluated_target_mesh.data
        target_world_matrix = evaluated_target_mesh.matrix_world

        vertex_interpolated_weights, found_vertices = _build_and_interpolate(
            base_mesh_data,
            base_world_matrix,
            target_face_indices,
            target_mesh_data,
            target_world_matrix,
            max_distance,
            max_angle_degrees,
            temp_base_vertex_group,
        )

        if vertex_interpolated_weights is None:
            return

        _apply_weights_to_target(target_mesh_data, target_vertex_group, vertex_interpolated_weights)

        bpy.ops.object.select_all(action='DESELECT')
        target_mesh.select_set(True)
        bpy.context.view_layer.objects.active = target_mesh

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')

        _set_active_vertex_group(target_mesh, vertex_group_name)

        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')

        if smooth_repeat > 0:
            bpy.ops.object.vertex_group_smooth(factor=0.5, repeat=smooth_repeat, expand=0.5)
        bpy.ops.object.mode_set(mode='OBJECT')

        print(f"一時メッシュ '{temp_base_mesh.name}' を削除中...")
        bpy.data.objects.remove(temp_base_mesh, do_unlink=True)
        temp_base_mesh_removed = True

        print(f"作成された頂点グループ: {vertex_group_name}")
        print(f"条件を満たした頂点数: {len(found_vertices)}")
        print(f"最大距離: {max_distance}")
    finally:
        if not temp_base_mesh_removed:
            bpy.data.objects.remove(temp_base_mesh, do_unlink=True)
        _restore_selection_state(selection_state)
