import json
import csv
import torch
import numpy as np
import os.path as osp
from plyfile import PlyData
from src.utils.color import to_float_rgb


__all__ = ['read_one_scan', 'read_one_test_scan']


########################################################################
#                             Votenet Utils                            #
#              https://github.com/facebookresearch/votenet             #
########################################################################

def represents_int(s):
    """ if string s represents an int. """
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename, label_from="raw_category", label_to="nyu40id"):
    assert osp.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping


def read_mesh_vertices(filename, rgb=True, normal=True):
    """read XYZ RGB for each vertex.
    Note: RGB values are in 0-255
    """
    assert osp.isfile(filename)

    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=[num_verts, 9], dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
        if rgb:
            vertices[:, 3] = plydata["vertex"].data["red"]
            vertices[:, 4] = plydata["vertex"].data["green"]
            vertices[:, 5] = plydata["vertex"].data["blue"]

    if normal:
        import open3d
        mesh = open3d.io.read_triangle_mesh(filename)
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        vertices[:, 6:9] = np.asarray(mesh.vertex_normals)

    return vertices


def read_aggregation(filename):
    assert osp.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data["segGroups"])
        for i in range(num_objects):
            object_id = data["segGroups"][i]["objectId"] + 1  # instance ids should be 1-indexed
            label = data["segGroups"][i]["label"]
            segs = data["segGroups"][i]["segments"]
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_axis_align_matrix(filename):
    lines = open(filename).readlines()
    axis_align_matrix = None
    for line in lines:
        if "axisAlignment" in line:
            axis_align_matrix = torch.Tensor(
                [float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")]).reshape((4, 4))
            break
    return axis_align_matrix


def read_segmentation(filename):
    assert osp.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data["segIndices"])
        for i in range(num_verts):
            seg_id = data["segIndices"][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def export(mesh_file, agg_file, seg_file, meta_file, label_map_file, output_file=None):
    """points are XYZ RGB (RGB in 0-255),
    semantic label as nyu40 ids,
    instance label as 1-#instance,
    box as (cx,cy,cz,dx,dy,dz,semantic_label)
    """
    label_map = read_label_mapping(label_map_file, label_from="raw_category", label_to="nyu40id")
    mesh_vertices = read_mesh_vertices(mesh_file, rgb=True, normal=True)

    # Load scene axis alignment matrix
    axis_align_matrix = read_axis_align_matrix(meta_file).numpy()
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    mesh_vertices[:, 0:3] = pts[:, 0:3]

    # Load semantic and instance labels
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]
    instance_bboxes = np.zeros((num_instances, 7))
    for obj_id in object_id_to_segs:
        label_id = object_id_to_label_id[obj_id]
        obj_pc = mesh_vertices[instance_ids == obj_id, 0:3]
        if len(obj_pc) == 0:
            continue
        # Compute axis aligned box
        # An axis aligned bounding box is parameterized by
        # (cx,cy,cz) and (dx,dy,dz) and label id
        # where (cx,cy,cz) is the center point of the box,
        # dx is the x-axis length of the box.
        xmin = np.min(obj_pc[:, 0])
        ymin = np.min(obj_pc[:, 1])
        zmin = np.min(obj_pc[:, 2])
        xmax = np.max(obj_pc[:, 0])
        ymax = np.max(obj_pc[:, 1])
        zmax = np.max(obj_pc[:, 2])
        bbox = np.array(
            [
                (xmin + xmax) / 2.0,
                (ymin + ymax) / 2.0,
                (zmin + zmax) / 2.0,
                xmax - xmin,
                ymax - ymin,
                zmax - zmin,
                label_id,
            ]
        )
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        instance_bboxes[obj_id - 1, :] = bbox

    return (
        mesh_vertices.astype(np.float32),
        label_ids.astype(np.int64),
        instance_ids.astype(np.int64),
        instance_bboxes.astype(np.float32),
        object_id_to_label_id)


########################################################################
#                          TorchPoints3D Utils                         #
#           https://github.com/torch-points3d/torch-points3d           #
########################################################################

def read_one_scan(scannet_dir, scan_name, label_map_file):
    mesh_file = osp.join(scannet_dir, scan_name, scan_name + "_vh_clean_2.ply")
    agg_file = osp.join(scannet_dir, scan_name, scan_name + ".aggregation.json")
    seg_file = osp.join(scannet_dir, scan_name, scan_name + "_vh_clean_2.0.010000.segs.json")
    meta_file = osp.join(scannet_dir, scan_name, scan_name + ".txt")

    mesh_vertices, semantic_labels, instance_labels, instance_bboxes, instance2semantic = export(
        mesh_file, agg_file, seg_file, meta_file, label_map_file, None)

    # Return values as tensors
    pos = torch.from_numpy(mesh_vertices[:, :3])
    rgb = to_float_rgb(torch.from_numpy(mesh_vertices[:, 3:6]))
    normal = torch.from_numpy(mesh_vertices[:, 6:9])
    y = torch.from_numpy(semantic_labels)
    obj = torch.from_numpy(instance_labels)

    return pos, rgb, normal, y, obj


def read_one_test_scan(scannet_dir, scan_name):
    mesh_file = osp.join(scannet_dir, scan_name, scan_name + "_vh_clean_2.ply")
    mesh_vertices = read_mesh_vertices(mesh_file, rgb=True, normal=True)

    pos = torch.from_numpy(mesh_vertices[:, :3])
    rgb = to_float_rgb(torch.from_numpy(mesh_vertices[:, 3:6]))
    normal = torch.from_numpy(mesh_vertices[:, 6:9])

    return pos, rgb, normal
