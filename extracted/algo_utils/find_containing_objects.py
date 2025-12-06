import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import bmesh
import bpy
from mathutils.bvhtree import BVHTree


class _ContainingContext:
    """State holder for computing clothing containment."""

    def __init__(self, clothing_meshes, threshold):
        self.clothing_meshes = clothing_meshes
        self.threshold = threshold

        # Intermediate state
        self.average_distances = {}  # {(container, contained): average_distance}
        self.best_containers = {}  # {contained: (container, avg_distance)}
        self.containing_objects = {}  # {container: [contained, ...]}
        self.parent_map = {}  # {child: parent}
        self.merged_containing_objects = {}  # {root: [contained, ...]}
        self.roots_in_order = []  # insertion order for roots
        self.final_result = {}  # {root: [contained, ...]}

    # ---- 距離計測と平均距離算出 ----
    def compute_average_distances(self):
        for i, obj1 in enumerate(self.clothing_meshes):
            for j, obj2 in enumerate(self.clothing_meshes):
                if i == j:
                    continue

                depsgraph = bpy.context.evaluated_depsgraph_get()

                eval_obj1 = obj1.evaluated_get(depsgraph)
                eval_mesh1 = eval_obj1.data

                eval_obj2 = obj2.evaluated_get(depsgraph)
                eval_mesh2 = eval_obj2.data

                bm1 = bmesh.new()
                bm1.from_mesh(eval_mesh1)
                bm1.transform(obj1.matrix_world)
                bvh_tree1 = BVHTree.FromBMesh(bm1)

                all_within_threshold = True
                total_distance = 0.0
                vertex_count = 0

                for vert in eval_mesh2.vertices:
                    vert_world = obj2.matrix_world @ vert.co
                    nearest = bvh_tree1.find_nearest(vert_world)

                    if nearest is None:
                        all_within_threshold = False
                        break

                    distance = nearest[3]
                    total_distance += distance
                    vertex_count += 1

                    if distance > self.threshold:
                        all_within_threshold = False
                        break

                if all_within_threshold and vertex_count > 0:
                    average_distance = total_distance / vertex_count
                    self.average_distances[(obj1, obj2)] = average_distance

                bm1.free()

    # ---- 最良コンテナの選択 ----
    def choose_best_containers(self):
        for (container, contained), avg_distance in self.average_distances.items():
            if contained not in self.best_containers or avg_distance < self.best_containers[contained][1]:
                self.best_containers[contained] = (container, avg_distance)

    # ---- 初期包含辞書の構築 ----
    def build_initial_containing_objects(self):
        for contained, (container, _) in self.best_containers.items():
            if container not in self.containing_objects:
                self.containing_objects[container] = []
            self.containing_objects[container].append(contained)
        return self.containing_objects

    # ---- 親子マップの構築 ----
    def build_parent_map(self):
        for container, contained_list in self.containing_objects.items():
            for child in contained_list:
                self.parent_map[child] = container

    # ---- 体積計算 ----
    @staticmethod
    def get_bounding_box_volume(obj):
        try:
            dims = getattr(obj, "dimensions", None)
            if dims is None:
                return 0.0
            return float(dims[0]) * float(dims[1]) * float(dims[2])
        except Exception:
            return 0.0

    # ---- ルート探索（サイクル対応） ----
    def find_root(self, obj):
        visited_list = []
        visited_set = set()
        current = obj

        while current in self.parent_map and current not in visited_set:
            visited_list.append(current)
            visited_set.add(current)
            current = self.parent_map[current]

        if current in visited_set:
            cycle_start = visited_list.index(current)
            cycle_nodes = visited_list[cycle_start:]
            root = max(
                cycle_nodes,
                key=lambda o: (
                    self.get_bounding_box_volume(o),
                    getattr(o, "name", str(id(o)))
                )
            )
        else:
            root = current

        for node in visited_list:
            self.parent_map[node] = root

        return root

    # ---- 子孫収集 ----
    def collect_descendants(self, obj, visited):
        result = []
        for child in self.containing_objects.get(obj, []):
            if child in visited:
                continue
            visited.add(child)
            result.append(child)
            result.extend(self.collect_descendants(child, visited))
        return result

    # ---- 包含階層の統合 ----
    def merge_containing_objects(self):
        for container in self.containing_objects.keys():
            root = self.find_root(container)
            if root not in self.merged_containing_objects:
                self.merged_containing_objects[root] = []
                self.roots_in_order.append(root)

        assigned_objects = set()
        for root in self.roots_in_order:
            visited = {root}
            descendants = self.collect_descendants(root, visited)
            for child in descendants:
                if child in assigned_objects:
                    continue
                self.merged_containing_objects[root].append(child)
                assigned_objects.add(child)

        for contained, (container, _) in self.best_containers.items():
            if contained in assigned_objects:
                continue
            root = self.find_root(container)
            if root not in self.merged_containing_objects:
                self.merged_containing_objects[root] = []
                self.roots_in_order.append(root)
            if contained == root:
                continue
            self.merged_containing_objects[root].append(contained)
            assigned_objects.add(contained)

        self.final_result = {
            root: self.merged_containing_objects[root]
            for root in self.roots_in_order
            if self.merged_containing_objects[root]
        }

    # ---- 重複検出とログ出力 ----
    def detect_duplicates_and_log(self):
        if not self.final_result:
            return

        seen_objects = set()
        duplicate_objects = set()

        for container, contained_list in self.final_result.items():
            if container in seen_objects:
                duplicate_objects.add(container)
            else:
                seen_objects.add(container)

            for obj in contained_list:
                if obj in seen_objects:
                    duplicate_objects.add(obj)
                else:
                    seen_objects.add(obj)

        if duplicate_objects:
            duplicate_names = sorted(
                {getattr(obj, "name", str(id(obj))) for obj in duplicate_objects}
            )
            print(
                "find_containing_objects: 同じオブジェクトが複数回検出されました -> "
                + ", ".join(duplicate_names)
            )

    # ---- オーケストレーション ----
    def run(self):
        self.compute_average_distances()
        self.choose_best_containers()
        self.build_initial_containing_objects()

        if not self.containing_objects:
            return {}

        self.build_parent_map()
        self.merge_containing_objects()
        self.detect_duplicates_and_log()
        return self.final_result


def find_containing_objects(clothing_meshes, threshold=0.02):
    """Find containment pairs between clothing meshes."""

    ctx = _ContainingContext(clothing_meshes, threshold)

    ctx.compute_average_distances()
    ctx.choose_best_containers()
    ctx.build_initial_containing_objects()

    if not ctx.containing_objects:
        return {}

    ctx.build_parent_map()
    ctx.merge_containing_objects()
    ctx.detect_duplicates_and_log()

    return ctx.final_result
