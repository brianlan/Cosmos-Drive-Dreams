# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import json

from pathlib import Path
from scipy.spatial.transform import Rotation as R
from termcolor import cprint
from utils.wds_utils import get_sample
import os
from utils.pcd_utils import interpolate_polyline_to_points, triangulate_polygon_3d, filter_by_height_relative_to_ego
from utils.graphics_utils import LineSegment2D, BoundingBox2D, Polygon2D, TriangleList2D

MINIMAP_TO_TYPE = json.load(open(Path(__file__).parent.parent / 'config' /'hdmap_type_config.json'))

MINIMAP_TO_SEMANTIC_LABEL = {
    'lanelines': 5,
    'lanes': 5,
    'poles': 9,
    'road_boundaries': 5,
    'wait_lines': 10,
    'crosswalks': 5,
    'road_markings': 10,
}

def extract_vertices(minimap_data, vertices_list=None):
    if vertices_list is None:
        vertices_list = []

    if isinstance(minimap_data, dict):
        for key, value in minimap_data.items():
            if key == 'vertices':
                vertices_list.append(value)
            else:
                extract_vertices(value, vertices_list)

    elif isinstance(minimap_data, list):
        for item in minimap_data:
            extract_vertices(item, vertices_list)

    return vertices_list

def get_type_from_name(minimap_name):
    """
    Args:
        minimap_name: str, name of the minimap

    Returns:
        minimap_type: str, type of the minimap
    """
    if minimap_name in MINIMAP_TO_TYPE:
        return MINIMAP_TO_TYPE[minimap_name]
    else:
        raise ValueError(f"Invalid minimap name: {minimap_name}")


def cuboid3d_to_polyline(cuboid3d_eight_vertices):
    """
    Convert cuboid3d to polyline

    Args:
        cuboid3d_eight_vertices: np.ndarray, shape (8, 3), dtype=np.float32,
            eight vertices of the cuboid3d

    Returns:
        polyline: np.ndarray, shape (N, 3), dtype=np.float32,
            polyline vertices
    """
    if isinstance(cuboid3d_eight_vertices, list):
        cuboid3d_eight_vertices = np.array(cuboid3d_eight_vertices)

    connected_vertices_indices = [0,1,2,3,0,4,5,6,7,4,5,1,2,6,7,3]
    connected_polyline = np.array(cuboid3d_eight_vertices)[connected_vertices_indices]

    return connected_polyline


def simplify_minimap(minimap_wds_file):
    """
    Args:
        minimap_wds_file: path to the webdataset file containing minimap data.
        Note that cuboid3d are converted to polylines!!

    Returns:
        minimap_data_wo_meta_info: list of list of 3d points
            containing extracted minimap data, it represents a polyline or polygon
            [[[x, y, z], [x, y, z], ...]],
            [[[x, y, z], [x, y, z], ...]], ...]

        -> minimap can be polygons, e.g., crosswalks, road_markings
        -> minimap can be polylines, e.g., lanelines, lanes, road_boundaries, wait_lines, poles
        -> minimap can be cuboid3d, e.g., traffic_signs, traffic_lights
    """

    minimap_raw_data = get_sample(minimap_wds_file)
    minimap_key_name = [key for key in minimap_raw_data.keys() if key.endswith('.json')][0]
    minimap_data = minimap_raw_data[minimap_key_name]
    minimap_data_wo_meta_info = extract_vertices(minimap_data)
    minimap_name = minimap_key_name.split('.')[0]

    # close the polygon
    if get_type_from_name(minimap_name) == 'polygon':
        for single_polygon in minimap_data_wo_meta_info:
            single_polygon.append(single_polygon[0])

    # 8 vertices, we can also make it polyline, just repeat some edges!
    if get_type_from_name(minimap_name) == 'cuboid3d':
        connected_vertices_indices = [0,1,2,3,0,4,5,6,7,4,5,1,2,6,7,3]
        for i, eight_vertices in enumerate(minimap_data_wo_meta_info):
            connected_polyline = cuboid3d_to_polyline(eight_vertices)
            minimap_data_wo_meta_info[i] = connected_polyline

    # for each polyline, if they are list, convert them to np.ndarray
    for i, polyline in enumerate(minimap_data_wo_meta_info):
        if isinstance(polyline, list):
            minimap_data_wo_meta_info[i] = np.array(polyline)

    return minimap_data_wo_meta_info, minimap_name


def create_minimap_projection(
    minimap_name,
    minimap_data_wo_meta_info,
    camera_poses,
    camera_model
):
    """
    Args:
        minimap_name: str, name of the minimap
        minimap_data_wo_meta_info: list of np.ndarray, results from simplify_minimap
        camera_poses: np.ndarray, shape (N, 4, 4), dtype=np.float32, camera poses of N frames
        camera_model: CameraModel, camera model

    Returns:
        minimaps_projection: np.ndarray,
            shape (N, H, W, 3), dtype=np.uint8, projected minimap data across N frames
    """
    MINIMAP_TO_RGB = json.load(open(Path(__file__).parent.parent / 'config' /'hdmap_color_config.json'))['hdmap']

    image_height, image_width = camera_model.height, camera_model.width
    # cprint(f"Processing minimap {minimap_name} with shape {image_height}x{image_width}", 'green')

    minimap_type = get_type_from_name(minimap_name)


    if minimap_type == 'polygon':
        projection_images = camera_model.draw_hull_depth(
            camera_poses,
            minimap_data_wo_meta_info,
            colors=np.array(MINIMAP_TO_RGB[minimap_name]),
        )
    elif minimap_type == 'polyline':
        if minimap_name == 'lanelines' or minimap_name == 'road_boundaries':
            segment_interval = 0.8
        else:
            segment_interval = 0

        projection_images = camera_model.draw_line_depth(
            camera_poses,
            minimap_data_wo_meta_info,
            colors=np.array(MINIMAP_TO_RGB[minimap_name]),
            segment_interval=segment_interval,
        )
    elif minimap_type == 'cuboid3d':
        projection_images = camera_model.draw_hull_depth(
            camera_poses,
            minimap_data_wo_meta_info,
            colors=np.array(MINIMAP_TO_RGB[minimap_name]),
        )
    else:
        raise ValueError(f"Invalid minimap type: {minimap_type}")

    return projection_images


def cuboid3d_update_vertices_remove_others(cuboid3d):
    """
    This is specific to the cuboid3d label of traffic_signs and traffic_lights.

    Args:
        cuboid3d: dict
            {
                'center': {'x': 180.09763, 'y': 83.11682, 'z': -16.458818},
                'orientation': {'x': -0.071179695, 'y': 0.0033680226, 'z': -0.36912337}, # yaw, pitch, roll
                'dimensions': {'x': 0.01, 'y': 0.64360946, 'z': 0.7865788}, # length, width, height
                'vertices': [{}, {}, {}, {}, {}, {}, {}, {}] # usually missing,
            }

    Returns:
        updated cuboid3d: dict
            remove 'center', 'orientation', 'dimensions'
            add 'vertices' with 8 vertices
    """
    max_xyz = np.array([cuboid3d['dimensions']['x'] / 2.0, cuboid3d['dimensions']['y'] / 2.0, cuboid3d['dimensions']['z'] / 2.0])
    min_xyz = -max_xyz

    # just a kind reminder, the order of 8 vertices here is different from build_cuboid_bounding_box() in bbox_utils.py
    # but it does not matter.
    vertices_of_cuboid = np.zeros((8, 3), dtype=np.float32)
    vertices_of_cuboid[0] = np.array([min_xyz[0], min_xyz[1], min_xyz[2]])
    vertices_of_cuboid[1] = np.array([min_xyz[0], max_xyz[1], min_xyz[2]])
    vertices_of_cuboid[2] = np.array([max_xyz[0], max_xyz[1], min_xyz[2]])
    vertices_of_cuboid[3] = np.array([max_xyz[0], min_xyz[1], min_xyz[2]])
    vertices_of_cuboid[4] = np.array([min_xyz[0], min_xyz[1], max_xyz[2]])
    vertices_of_cuboid[5] = np.array([min_xyz[0], max_xyz[1], max_xyz[2]])
    vertices_of_cuboid[6] = np.array([max_xyz[0], max_xyz[1], max_xyz[2]])
    vertices_of_cuboid[7] = np.array([max_xyz[0], min_xyz[1], max_xyz[2]])

    yaw, pitch, roll = cuboid3d['orientation']['x'], cuboid3d['orientation']['y'], cuboid3d['orientation']['z']
    rotation_matrix = R.from_euler('ZYX', [yaw, pitch, roll]).as_matrix()

    # Apply rotation.
    vertices_of_cuboid = (rotation_matrix @ vertices_of_cuboid.T).T

    # Add center translation
    center = np.array([cuboid3d['center']['x'], cuboid3d['center']['y'], cuboid3d['center']['z']])
    vertices_of_cuboid = vertices_of_cuboid + center

    # following protobuf format
    cuboid3d['vertices'] = [{'x': vertex[0], 'y': vertex[1], 'z': vertex[2]} for vertex in vertices_of_cuboid.tolist()]
    cuboid3d.pop('center', None)
    cuboid3d.pop('orientation', None)
    cuboid3d.pop('dimensions', None)

    return cuboid3d


def convert_cuboid3d_recursive(minimap_message):
    """
    Recursively traverse minimap_message and update any cuboid3d entries with vertices.

    Args:
        minimap_message: dict/list, potentially nested structure that may contain cuboid3d entries

    Returns:
        Updated minimap_message with cuboid3d entries converted to include vertices
    """
    if isinstance(minimap_message, dict):
        # If this is a cuboid3d dict, update it
        if 'cuboid3d' in minimap_message:
            minimap_message['cuboid3d'] = cuboid3d_update_vertices_remove_others(minimap_message['cuboid3d'])
        # Recursively process all dict values
        for key in minimap_message:
            minimap_message[key] = convert_cuboid3d_recursive(minimap_message[key])

    elif isinstance(minimap_message, list):
        # Recursively process all list items
        minimap_message = [convert_cuboid3d_recursive(item) for item in minimap_message]

    return minimap_message


def transform_decoded_label(decoded_label, transformation_matrix):
    """
    Apply transformation matrix to decoded label

    Args:
        decoded_label: dict,
            decoded label from decode_static_label, can have several hierarchies,
            but the last one is always numpy array with shape [N, 3]
        transformation_matrix: 4x4 numpy array,
            transformation matrix we want to apply to the numpy array
    Returns:
        dict: transformed decoded_label with the same structure, but numpy array transformed
    """

    def transform_vertices(vertices, transformation_matrix):
        """
        Args:
            vertices: numpy array, shape [N, 3]
            transformation_matrix: 4x4 numpy array
        Returns:
            numpy array, shape [N, 3]
        """
        if vertices.shape == (0,):
            return vertices

        # add 1 to the vertices
        vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        transformed_vertices = np.dot(transformation_matrix, vertices.T).T
        return transformed_vertices[:, :3]

    # recursively apply transformation matrix to the vertices
    def transform(decoded_label, transformation_matrix):
        if isinstance(decoded_label, np.ndarray):
            return transform_vertices(decoded_label, transformation_matrix)
        elif isinstance(decoded_label, dict):
            transformed_label = {}
            for key, value in decoded_label.items():
                transformed_label[key] = transform(value, transformation_matrix)
            return transformed_label
        else:
            raise ValueError(f"Unknown type in decoded_label: {type(decoded_label)}")

    return transform(decoded_label, transformation_matrix)


def prepare_minimap_data_for_world_scenario(minimap_wds_files):
    MINIMAP_TO_RGB = json.load(open(Path(__file__).parent.parent / 'config' /'world_scenario_color_config.json'))['hdmap']

    minimap_name_to_minimap_data = {}
    for minimap_wds_file in minimap_wds_files:
        minimap_data_wo_meta_info, minimap_name = simplify_minimap(minimap_wds_file)
        minimap_name_to_minimap_data[minimap_name] = minimap_data_wo_meta_info

    return minimap_name_to_minimap_data, MINIMAP_TO_RGB


def create_minimap_geometry_objects_from_data(
    minimap_name_to_minimap_data,
    camera_pose,
    camera_model,
    minimap_to_rgb,
    camera_pose_init=None,
):
    """
    Build geometry objects for minimap layers for a single frame.

    Args:
        minimap_name_to_minimap_data: dict[name -> list[np.ndarray]]
        camera_pose: np.ndarray (4,4)
        camera_model: CameraModel
        minimap_to_rgb: dict[str, list[np.ndarray]], mapping from minimap name to RGB values
        camera_pose_init: np.ndarray (4,4), reference camera pose for ego space transformation

    Returns:
        list: geometry objects (LineSegment2D/Polygon2D)
    """

    all_geometry_objects = []
    for minimap_name, minimap_data in minimap_name_to_minimap_data.items():
        minimap_type = get_type_from_name(minimap_name)
        # We create LineSegment2D geometry object for each polyline
        if minimap_type == 'polyline':
            line_segment_list = []
            for polyline in minimap_data:
                # Filter out minimap that is under the ego vehicle
                if camera_pose_init is not None and filter_by_height_relative_to_ego(
                    polyline, camera_model, camera_pose, camera_pose_init
                ):
                    continue
                # Subdivide the polyline so that distortion is observed in camera view
                polyline_subdivided = interpolate_polyline_to_points(polyline, segment_interval=0.2)
                line_segment = np.stack([polyline_subdivided[0:-1], polyline_subdivided[1:]], axis=1) # [N, 2, 3]
                line_segment_list.append(line_segment)
            if len(line_segment_list) == 0:
                continue

            all_line_segments = np.concatenate(line_segment_list, axis=0) # [N', 2, 3]
            xy_and_depth = camera_model.get_xy_and_depth(all_line_segments.reshape(-1, 3), camera_pose).reshape(-1, 2, 3) # [N', 3]

            # filter the line segments with both vertices have depth >= 0
            valid_line_segment_vertices = xy_and_depth[:, :, 2] >= 0
            valid_line_segment_indices = np.all(valid_line_segment_vertices, axis=1)
            valid_xy_and_depth = xy_and_depth[valid_line_segment_indices]
            if len(valid_xy_and_depth) == 0:
                continue

            color_float = np.array(minimap_to_rgb[minimap_name]) / 255.0
            all_geometry_objects.append(
                LineSegment2D(
                    valid_xy_and_depth,
                    base_color=color_float,
                    line_width=5 if minimap_name == 'poles' else 12,
                )
            )

        elif minimap_type == 'polygon' or minimap_type == 'cuboid3d':
            # merge all vertices from polygons and record indices
            polygon_vertices = []
            polygon_vertex_counts = []
            for polygon in minimap_data:
                # Filter out minimap that is under the ego vehicle
                if camera_pose_init is not None and filter_by_height_relative_to_ego(
                    polygon, camera_model, camera_pose, camera_pose_init
                ):
                    continue
                if minimap_name == 'crosswalks':
                    # Subdivide the polygon so that distortion is observed in camera view
                    polygon_subdivided = interpolate_polyline_to_points(polygon, segment_interval=0.8)
                    # Use triangulation for crosswalks to handle concave polygons in camera view
                    triangles_3d = triangulate_polygon_3d(polygon_subdivided)
                    polygon_subdivided = triangles_3d.reshape(-1, 3)
                    if len(polygon_subdivided) == 0:
                        continue
                else:
                    polygon_subdivided = polygon
                polygon_vertices.append(polygon_subdivided)
                polygon_vertex_counts.append(len(polygon_subdivided))
            if len(polygon_vertices) == 0:
                continue
            # get xy and depth for all vertices at once
            all_vertices = np.concatenate(polygon_vertices, axis=0)
            all_xy_and_depth = camera_model.get_xy_and_depth(all_vertices, camera_pose)

            # recover individual polygons using recorded counts, and keep access to original 3D subdivided vertices
            start_idx = 0
            for vertex_count in polygon_vertex_counts:
                polygon_xy_and_depth = all_xy_and_depth[start_idx:start_idx+vertex_count]
                start_idx += vertex_count
                color_float = np.array(minimap_to_rgb[minimap_name]) / 255.0

                if minimap_name == 'crosswalks':
                    triangles_proj = polygon_xy_and_depth.reshape(-1, 3, 3)
                    # Filter out triangles that are completely behind camera
                    invalid_triangles_indices = np.all(triangles_proj[:, :, 2] < 0, axis=1)
                    valid_triangles_indices = ~invalid_triangles_indices
                    if valid_triangles_indices.sum() > 0:
                        all_geometry_objects.append(
                            TriangleList2D(triangles_proj[valid_triangles_indices], base_color=color_float)
                        )
                else:
                    # filter out the polygons that are out of the image
                    if np.all(polygon_xy_and_depth[:, 2] < 0):
                        continue
                    # Use regular polygon rendering for other polygon types
                    all_geometry_objects.append(
                        Polygon2D(polygon_xy_and_depth, base_color=color_float)
                    )
        else:
            raise ValueError(f"Invalid minimap type: {minimap_type}")

    return all_geometry_objects


def prepare_traffic_light_status_data(
    traffic_light_position_wds_file,
    traffic_light_per_frame_status_wds_file,
):
    """
    Preload traffic light position and per-frame status data.

    Args:
        traffic_light_position_wds_file: str, path to the webdataset file containing traffic light position data
        traffic_light_per_frame_status_wds_file: str, path to the webdataset file containing traffic light per-frame status data

    Returns:
        tuple: (position_list, status_dict, tl_status_to_rgb)
            - position_list: list[dict] or None if file doesn't exist
            - status_dict: dict or None if file doesn't exist or unavailable
            - tl_status_to_rgb: dict, always returned
    """
    tl_status_to_rgb = json.load(open(Path(__file__).parent.parent / 'config' /'world_scenario_color_config.json'))['traffic_lights']

    # Check if traffic light position file exists
    if not os.path.exists(traffic_light_position_wds_file):
        return None, None, tl_status_to_rgb

    try:
        traffic_light_position_sample = get_sample(traffic_light_position_wds_file)
        position_list = traffic_light_position_sample['traffic_lights.json']['labels']
    except (KeyError, StopIteration, FileNotFoundError) as e:
        # File exists but is empty or malformed
        return None, None, tl_status_to_rgb

    # Check if traffic light status file exists
    if not os.path.exists(traffic_light_per_frame_status_wds_file):
        return position_list, None, tl_status_to_rgb

    try:
        traffic_light_per_frame_status_sample = get_sample(traffic_light_per_frame_status_wds_file)
        status_dict = traffic_light_per_frame_status_sample['aggregated_states.json']['traffic_light_states']
    except (KeyError, StopIteration, FileNotFoundError) as e:
        # File exists but is empty or malformed
        return position_list, None, tl_status_to_rgb

    return (
        position_list,
        status_dict,
        tl_status_to_rgb,
    )


def create_traffic_light_status_geometry_objects_from_data(
    traffic_light_position_list,
    traffic_light_per_frame_status_dict,
    frame_id,
    camera_pose,
    camera_model,
    tl_status_to_rgb,
):
    """
    Build geometry objects (Polygon2D) for traffic lights for a single frame.

    Args:
        traffic_light_position_list: list[dict], traffic light position data
        traffic_light_per_frame_status_dict: dict[str, dict], traffic light per-frame status data
        frame_id: int, frame id
        camera_pose: np.ndarray, shape (4, 4), dtype=np.float32, camera pose
        camera_model: CameraModel, ftheta or pinhole camera model
        tl_status_to_rgb: dict[str, list[np.ndarray]], mapping from traffic light status to RGB values

    Returns:
        list[Polygon2D]: geometry objects for traffic lights
    """
    if traffic_light_position_list is None:
        return []
    # in RDS-HQ format, if there is only one element, it is actually a label for no traffic light.
    if len(traffic_light_position_list) == 1:
        return []

    polygon_vertices = []
    polygon_colors = []
    for traffic_light_index, traffic_light_data in enumerate(traffic_light_position_list):
        if traffic_light_per_frame_status_dict is None:
            # Use unknown color for traffic light without status
            signal_render_color = np.array(tl_status_to_rgb['unknown']) / 255.0
        else:
            this_frame_status = traffic_light_per_frame_status_dict[str(traffic_light_index)]
            assert {"name": "feature_id", "text": this_frame_status['feature_id']} in \
                traffic_light_data['labelData']['shape3d']['attributes']
            signal_state = this_frame_status['state'][frame_id]
            signal_render_color = np.array(tl_status_to_rgb[signal_state]) / 255.0
        polygon_vertices.append(
            cuboid3d_to_polyline(
                traffic_light_data['labelData']['shape3d']['cuboid3d']['vertices']
            ) # [16, 3]
        )
        polygon_colors.append(signal_render_color)

    # projecting all points together to save time
    all_polygon_vertices = np.concatenate(polygon_vertices, axis=0) # [N * 16, 3]
    all_xy_and_depth = camera_model.get_xy_and_depth(all_polygon_vertices, camera_pose) # [N * 16, 3]
    all_xy_and_depth = all_xy_and_depth.reshape(-1, 16, 3) # [N, 16, 3]

    geometry_objects = []
    for tl_index, xy_and_depth in enumerate(all_xy_and_depth):
        if np.all(xy_and_depth[:, 2] < 0):
            continue
        geometry_objects.append(
            Polygon2D(xy_and_depth, base_color=polygon_colors[tl_index])
        )

    return geometry_objects
