import os
import sys

from mathutils.bvhtree import BVHTree
import bpy
import numpy as np
import os
import sys


# Merged from calculate_weight_pattern_similarity.py

def calculate_weight_pattern_similarity(weights1, weights2):
    """
    2つのウェイトパターン間の類似性を計算する
    
    Parameters:
        weights1: 1つ目のウェイトパターン {group_name: weight}
        weights2: 2つ目のウェイトパターン {group_name: weight}
        
    Returns:
        float: 類似度（0.0〜1.0、1.0が完全一致）
    """
    # 両方のパターンに存在するグループを取得
    all_groups = set(weights1.keys()) | set(weights2.keys())
    
    if not all_groups:
        return 0.0
    
    # 各グループのウェイト差の合計を計算
    total_diff = 0.0
    for group in all_groups:
        w1 = weights1.get(group, 0.0)
        w2 = weights2.get(group, 0.0)
        total_diff += abs(w1 - w2)
    
    # 正規化（グループ数で割る）
    normalized_diff = total_diff / len(all_groups)
    
    # 類似度に変換（差が小さいほど類似度が高い）
    similarity = 1.0 - min(normalized_diff, 1.0)
    
    return similarity

# Merged from normalize_vertex_weights.py

def normalize_vertex_weights(obj):
    """
    指定されたメッシュオブジェクトのボーンウェイトを正規化する。
    Args:
        obj: 正規化するメッシュオブジェクト
    """
    if obj.type != 'MESH':
        print(f"Error: {obj.name} is not a mesh object")
        return

    # 頂点グループが存在するか確認
    if not obj.vertex_groups:
        print(f"Warning: {obj.name} has no vertex groups")
        return
        
    # 各頂点が少なくとも1つのグループに属しているか確認
    for vert in obj.data.vertices:
        if not vert.groups:
            print(f"Warning: Vertex {vert.index} in {obj.name} has no weights")
    
    # Armatureモディファイアの確認
    has_armature = any(mod.type == 'ARMATURE' for mod in obj.modifiers)
    if not has_armature:
        print(f"Error: {obj.name} has no Armature modifier")
        return
    
    # すべての選択を解除
    bpy.ops.object.select_all(action='DESELECT')

    # アクティブオブジェクトを設定
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # ウェイトの正規化を実行
    bpy.ops.object.vertex_group_normalize_all(
        group_select_mode='BONE_DEFORM',
        lock_active=False
    )
    print(f"Normalized weights for {obj.name}")

# Merged from normalize_bone_weights.py

def normalize_bone_weights(obj: bpy.types.Object, avatar_data: dict) -> None:
    """
    メッシュのボーン変形に関わる頂点ウェイトを正規化する。
    
    Parameters:
        obj: メッシュオブジェクト
        avatar_data: アバターデータ
    """
    if obj.type != 'MESH':
        return
        
    # 正規化対象のボーングループを取得
    target_groups = set()
    # Humanoidボーンを追加
    for bone_map in avatar_data.get("humanoidBones", []):
        if "boneName" in bone_map:
            target_groups.add(bone_map["boneName"])
    
    # 補助ボーンを追加
    for aux_set in avatar_data.get("auxiliaryBones", []):
        for aux_bone in aux_set.get("auxiliaryBones", []):
            target_groups.add(aux_bone)
    
    # 各頂点について処理
    for vert in obj.data.vertices:
        # ターゲットグループのウェイト合計を計算
        total_weight = 0.0
        weights = {}
        
        for g in vert.groups:
            group_name = obj.vertex_groups[g.group].name
            if group_name in target_groups:
                total_weight += g.weight
                weights[group_name] = g.weight
        
        # ウェイトの正規化
        for group_name, weight in weights.items():
            normalized_weight = weight / total_weight
            obj.vertex_groups[group_name].add([vert.index], normalized_weight, 'REPLACE')

# Merged from calculate_distance_based_weights.py

def calculate_distance_based_weights(source_obj_name, target_obj_name, vertex_group_name="DistanceWeight", min_distance=0.0, max_distance=0.03):
    """
    指定されたオブジェクトの各頂点から別のオブジェクトまでの最近接面距離を計測し、
    距離に基づいて頂点ウェイトを設定する関数
    
    Args:
        source_obj_name (str): ウェイトを設定するオブジェクト名
        target_obj_name (str): 距離計測対象のオブジェクト名
        vertex_group_name (str): 作成する頂点グループ名
        min_distance (float): 最小距離（ウェイト1.0になる距離）
        max_distance (float): 最大距離（ウェイト0.0になる距離）
    """
    
    # オブジェクトを取得
    source_obj = bpy.data.objects.get(source_obj_name)
    target_obj = bpy.data.objects.get(target_obj_name)
    
    if not source_obj:
        print(f"エラー: オブジェクト '{source_obj_name}' が見つかりません")
        return False
    
    if not target_obj:
        print(f"エラー: オブジェクト '{target_obj_name}' が見つかりません")
        return False
    
    # メッシュデータを取得
    source_mesh = source_obj.data
    target_mesh = target_obj.data
    
    # 頂点グループを作成または取得
    if vertex_group_name not in source_obj.vertex_groups:
        vertex_group = source_obj.vertex_groups.new(name=vertex_group_name)
    else:
        vertex_group = source_obj.vertex_groups[vertex_group_name]
    
    # ターゲットオブジェクトのBVHTreeを作成
    print("BVHTreeを構築中...")
    
    # ターゲットメッシュのワールド座標での頂点とポリゴンを取得
    target_verts = []
    target_polys = []
    
    # 評価されたメッシュを取得（モディファイアが適用された状態）
    depsgraph = bpy.context.evaluated_depsgraph_get()
    target_eval = target_obj.evaluated_get(depsgraph)
    target_mesh_eval = target_eval.data
    
    # ワールド座標に変換
    target_matrix = target_obj.matrix_world
    
    for vert in target_mesh_eval.vertices:
        world_co = target_matrix @ vert.co
        target_verts.append(world_co)
    
    for poly in target_mesh_eval.polygons:
        target_polys.append(poly.vertices)
    
    # BVHTreeを構築
    bvh = BVHTree.FromPolygons(target_verts, target_polys)
    
    print("距離計算とウェイト設定中...")
    
    # ソースオブジェクトの各頂点について処理
    source_matrix = source_obj.matrix_world
    source_eval = source_obj.evaluated_get(depsgraph)
    source_mesh_eval = source_eval.data
    
    weights = []
    
    for i, vert in enumerate(source_mesh_eval.vertices):
        # 頂点のワールド座標を取得
        world_co = source_matrix @ vert.co
        
        # 最近接面までの距離を計算
        location, normal, index, distance = bvh.find_nearest(world_co)
        
        if location is None:
            print(f"警告: 頂点 {i} の最近接面が見つかりません")
            distance = max_distance
        
        # 距離に基づいてウェイトを計算
        if distance <= min_distance:
            weight = 1.0
        elif distance >= max_distance:
            weight = 0.0
        else:
            # 線形補間でウェイトを計算（max_distanceに近づくほど0に近づく）
            weight = 1.0 - ((distance - min_distance) / (max_distance - min_distance))
        
        weights.append(weight)
        
        # 頂点グループにウェイトを設定
        vertex_group.add([i], weight, 'REPLACE')
    
    print(f"完了: {len(weights)} 個の頂点にウェイトを設定しました")
    print(f"最小ウェイト: {min(weights):.4f}")
    print(f"最大ウェイト: {max(weights):.4f}")
    print(f"平均ウェイト: {np.mean(weights):.4f}")
    
    return True