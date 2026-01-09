#!/usr/bin/env python3
"""
Convert Ruqi/pre-fusion pickle data to the RDS-HQ layout expected by Cosmos-Drive-Dreams.

Input  (ruqi):
    - One pickle file with structure:
        {
            <scene_id>: {
                "scene_info": {
                    "calibration": {<cam>: {"extrinsic": (R_cam_to_ego, t_cam_to_ego), "intrinsic": [fx, fy, cx, cy], ...}, ...},
                    "camera_mask": {<cam>: <path>, ...},
                    ...
                },
                "frame_info": {
                    <timestamp_str>: {
                        "ego_pose": {"rotation": R_ego_to_world, "translation": t_world},
                        "camera_image": {<cam>: {"path": <img_path>, "calibration": {...}}, ...},
                        "3d_boxes": [...],
                        "3d_polylines": [...],
                        "lidar_points": {...},
                        ...
                    },
                    ...
                }
            }
        }

Output (RDS-HQ-style folders under --output-root/<scene_id>):
    - pinhole_intrinsic/<scene_id>.tar
    - pose/<scene_id>.tar
    - vehicle_pose/<scene_id>.tar
    - all_object_info/<scene_id>.tar
    - 3d_lanelines/<scene_id>.tar
    - 3d_road_boundaries/<scene_id>.tar
    - 3d_road_markings/<scene_id>.tar
    - 3d_traffic_lights/<scene_id>.tar            (empty placeholder)
    - 3d_traffic_lights_status/<scene_id>.tar     (empty placeholder)
    - 3d_traffic_signs/<scene_id>.tar             (empty placeholder)
    - 3d_wait_lines/<scene_id>.tar                (empty placeholder)
    - 3d_crosswalks/<scene_id>.tar                (empty placeholder)
    - 3d_poles/<scene_id>.tar                     (empty placeholder)

Notes:
    - The source Ruqi clip is 10 FPS; this script duplicates each frame 3× to produce 30 FPS
      poses and bbox tracks to satisfy Cosmos renderers (which assume INPUT_POSE_FPS=30).
    - Object coordinates and map polylines are assumed to be ego-centric; they are lifted
      into world coordinates via each frame's ego_pose.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

# Allow importing toolkit utils without installing as a package
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TOOLKIT_ROOT = PROJECT_ROOT / "cosmos-drive-dreams-toolkits"
import sys

if str(TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_ROOT))

from utils.wds_utils import write_to_tar, encode_dict_to_npz_bytes  # type: ignore

try:
    import pypcd4 as pypcd  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    pypcd = None


def make_se3(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 pose matrix (camera/object -> world)."""
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rotation.astype(np.float32)
    mat[:3, 3] = translation.astype(np.float32)
    return mat


def transform_points(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply SE3 to Nx3 points."""
    return (mat[:3, :3] @ pts.T + mat[:3, 3:4]).T


def derive_image_size(first_frame: dict, data_root: Path) -> Tuple[int, int]:
    """Read the first available image to get (width, height)."""
    cam_entry = next(iter(first_frame["camera_image"].values()))
    img_rel = cam_entry["path"]
    img_path = (data_root / img_rel).resolve()
    with Image.open(img_path) as im:
        w, h = im.size
    return w, h


def build_intrinsics_sample(
    clip_id: str, scene_info: dict, resolution: Tuple[int, int]
) -> dict:
    """Create pinhole_intrinsic sample dict."""
    w, h = resolution
    sample = {"__key__": clip_id}
    for cam, calib in scene_info["calibration"].items():
        fx, fy, cx, cy = map(float, calib["intrinsic"])
        sample[f"pinhole_intrinsic.{cam}.npy"] = np.array(
            [fx, fy, cx, cy, w, h], dtype=np.float32
        )
    return sample


def box_class_to_object_type(box_cls: str) -> str:
    """Map Ruqi box class to Cosmos-friendly object type."""
    mapping = {
        "car": "Vehicle",
        "suv": "Vehicle",
        "truck": "Truck",
        "construction_vehicle": "Construction",
        "motor": "Motorcycle",
        "human": "Pedestrian",
        "cone": "Cone",
        "anti_collision": "Barrier",
        "barrier": "Barrier",
        "isolation_barrier": "Barrier",
        "isolation_barrel": "Barrier",
        "Fire_Hydrant": "Fire_Hydrant",
        "pole": "Pole",
    }
    return mapping.get(box_cls, "Unknown")


def polyline_class_to_layer(poly_cls: str) -> str | None:
    """Map Ruqi polyline class to RDS-HQ minimap layer."""
    if poly_cls in {"centerline", "lane"}:
        return "lanelines"
    if poly_cls == "curb":
        return "road_boundaries"
    if poly_cls in {"arrow", "stop_line"}:
        return "road_markings"
    return None


def duplicate_frames_to_30fps(num_src_frames: int) -> List[int]:
    """Return list of target frame indices at 30 FPS."""
    target = []
    for i in range(num_src_frames):
        base = i * 3
        target.extend([base, base + 1, base + 2])
    return target


def convert_scene(
    scene_id: str,
    scene: dict,
    output_root: Path,
    clip_id: str,
    data_root: Path,
    keyframe_stride: int = 1,
) -> None:
    frame_keys = sorted(scene["frame_info"].keys())
    frame_index_lookup = {k: i for i, k in enumerate(frame_keys)}
    selected_keys = frame_keys[:: max(1, keyframe_stride)]
    first_frame = scene["frame_info"][frame_keys[0]]
    resolution = derive_image_size(first_frame, data_root)
    anchor_translation = np.asarray(first_frame["ego_pose"]["translation"], dtype=np.float32)

    # Prepare intrinsic tar
    intrinsic_sample = build_intrinsics_sample(clip_id, scene["scene_info"], resolution)
    write_to_tar(intrinsic_sample, output_root / "pinhole_intrinsic" / f"{clip_id}.tar")

    # Precompute camera-to-ego transforms
    cam_to_ego: Dict[str, np.ndarray] = {}
    for cam, calib in scene["scene_info"]["calibration"].items():
        # Ruqi extrinsic appears to be ego->cam; invert to get cam->ego as expected by renderer
        ego_to_cam = make_se3(
            np.asarray(calib["extrinsic"][0], dtype=np.float32),
            np.asarray(calib["extrinsic"][1], dtype=np.float32),
        )
        cam_to_ego[cam] = np.linalg.inv(ego_to_cam)

    pose_sample = {"__key__": clip_id}
    vehicle_pose_sample = {"__key__": clip_id}
    all_object_info_sample = {"__key__": clip_id}

    # Build 30 FPS timeline by duplicating each 10 FPS frame 3×
    target_frame_ids = duplicate_frames_to_30fps(len(frame_keys))

    for src_idx, frame_key in enumerate(frame_keys):
        frame = scene["frame_info"][frame_key]
        ego_pose = make_se3(
            np.asarray(frame["ego_pose"]["rotation"], dtype=np.float32),
            np.asarray(frame["ego_pose"]["translation"], dtype=np.float32) - anchor_translation,
        )

        for offset, tgt_frame_id in enumerate(target_frame_ids[src_idx * 3 : src_idx * 3 + 3]):
            # vehicle pose
            vehicle_pose_sample[
                f"{tgt_frame_id:06d}.vehicle_pose.npy"
            ] = ego_pose.astype(np.float32)

            # camera poses
            for cam, cam_2_ego in cam_to_ego.items():
                cam_to_world = ego_pose @ cam_2_ego
                pose_sample[
                    f"{tgt_frame_id:06d}.pose.{cam}.npy"
                ] = cam_to_world.astype(np.float32)

            # all object info: only keep selected (key) frames to reflect trustworthy annotations
            if frame_key in selected_keys:
                object_info_this_frame: Dict[str, dict] = {}
                for box in frame["3d_boxes"]:
                    track_id = str(box.get("track_id", -1))
                    R_box = np.asarray(box["rotation"], dtype=np.float32)
                    t_box = np.asarray(box["translation"], dtype=np.float32)
                    box_in_ego = make_se3(R_box, t_box)
                    box_in_world = ego_pose @ box_in_ego
                    velocity = np.asarray(box.get("velocity", [0, 0, 0]), dtype=np.float32)
                    object_info_this_frame[track_id] = {
                        "object_to_world": box_in_world.tolist(),
                        "object_lwh": list(box.get("size", [0, 0, 0])),
                        "object_is_moving": bool(np.linalg.norm(velocity[:2]) > 0.1),
                        "object_type": box_class_to_object_type(box.get("class", "")),
                    }

                all_object_info_sample[
                    f"{tgt_frame_id:06d}.all_object_info.json"
                ] = object_info_this_frame

    write_to_tar(pose_sample, output_root / "pose" / f"{clip_id}.tar")
    write_to_tar(vehicle_pose_sample, output_root / "vehicle_pose" / f"{clip_id}.tar")
    write_to_tar(all_object_info_sample, output_root / "all_object_info" / f"{clip_id}.tar")

    # Minimap layers from first frame's polylines
    polylines_by_layer: Dict[str, List[List[float]]] = {}
    first_ego_pose = make_se3(
        np.asarray(first_frame["ego_pose"]["rotation"], dtype=np.float32),
        np.asarray(first_frame["ego_pose"]["translation"], dtype=np.float32) - anchor_translation,
    )
    for pl in first_frame.get("3d_polylines", []):
        layer = polyline_class_to_layer(pl.get("class", ""))
        if layer is None:
            continue
        pts_world = transform_points(first_ego_pose, np.asarray(pl["points"], dtype=np.float32))
        polylines_by_layer.setdefault(layer, []).append(pts_world.tolist())

    def write_minimap(layer: str, polylines: Iterable[List[List[float]]]):
        sample = {"__key__": clip_id, f"{layer}.json": {"labels": []}}
        for poly in polylines:
            sample[f"{layer}.json"]["labels"].append(
                {
                    "labelData": {
                        "shape3d": {
                            "polyline3d": {
                                "vertices": poly,
                                "unit": "m",
                            }
                        }
                    }
                }
            )
        write_to_tar(sample, output_root / f"3d_{layer}" / f"{clip_id}.tar")

    for layer_name, polys in polylines_by_layer.items():
        write_minimap(layer_name, polys)

    # Write empty placeholders for layers we don't have
    empty_layers = [
        "traffic_lights",
        "traffic_lights_status",
        "traffic_signs",
        "wait_lines",
        "crosswalks",
        "poles",
    ]
    for layer in empty_layers:
        sample = {"__key__": clip_id, f"{layer}.json": {"labels": []}}
        write_to_tar(sample, output_root / f"3d_{layer}" / f"{clip_id}.tar")

    # If road_markings were absent above, still emit an empty file to keep renderers happy.
    if "road_markings" not in polylines_by_layer:
        write_minimap("road_markings", [])
    if "road_boundaries" not in polylines_by_layer:
        write_minimap("road_boundaries", [])
    if "lanelines" not in polylines_by_layer:
        write_minimap("lanelines", [])

    # LiDAR (requires pypcd4)
    def read_lidar_xyz_intensity(path: Path):
        if pypcd is not None:
            try:
                pc = pypcd.PointCloud.from_path(path)
                xyz = np.stack([pc["x"], pc["y"], pc["z"]], axis=1).astype(np.float32)
                if "intensity" in pc.fields:
                    intensity = np.asarray(pc["intensity"], dtype=np.float32)
                else:
                    intensity = np.zeros(len(xyz), dtype=np.float32)
                return xyz, intensity
            except Exception:
                pass
        try:
            import open3d as o3d  # type: ignore
            pcd = o3d.io.read_point_cloud(str(path))
            xyz = np.asarray(pcd.points, dtype=np.float32)
            if pcd.has_colors():
                intensity = np.mean(np.asarray(pcd.colors, dtype=np.float32), axis=1)
            else:
                intensity = np.zeros(len(xyz), dtype=np.float32)
            return xyz, intensity
        except Exception:
            return None, None

    if pypcd is None:
        print("Warning: pypcd4 not available; will try open3d. If that fails, LiDAR export will be skipped.")

    lidar_sample = {"__key__": clip_id}
    lidar_frames = []
    for idx, frame_key in enumerate(selected_keys):
        frame = scene["frame_info"][frame_key]
        lidar_section = frame.get("lidar_points") or {}
        if "lidar_top" not in lidar_section:
            continue
        lidar_entry = lidar_section["lidar_top"]
        raw_path = (data_root / lidar_entry["path"]).resolve()
        if not raw_path.exists():
            print(f"Warning: LiDAR file missing: {raw_path}")
            continue

        xyz, intensity = read_lidar_xyz_intensity(raw_path)
        if xyz is None:
            print(f"Warning: failed to read LiDAR {raw_path}")
            continue

        n = len(xyz)
        row = np.zeros(n, dtype=np.uint8)
        col = (np.arange(n, dtype=np.uint16)) % 3600

        lidar_extrinsic = make_se3(
            np.asarray(lidar_entry["extrinsic"]["rotation"], dtype=np.float32),
            np.asarray(lidar_entry["extrinsic"]["translation"], dtype=np.float32),
        )
        ego_pose = make_se3(
            np.asarray(frame["ego_pose"]["rotation"], dtype=np.float32),
            np.asarray(frame["ego_pose"]["translation"], dtype=np.float32) - anchor_translation,
        )
        lidar_to_world = ego_pose @ lidar_extrinsic

        start_ts = int(float(frame_key))

        lidar_sample[f"{idx:06d}.lidar_raw.npz"] = encode_dict_to_npz_bytes(
            {
                "xyz": xyz.astype(np.float32),
                "intensity": intensity.astype(np.float32),
                "row": row,
                "column": col,
                "starting_timestamp": np.array(start_ts, dtype=np.int64),
                "lidar_to_world": lidar_to_world.astype(np.float32),
            }
        )
        lidar_frames.append(idx)

    if lidar_frames:
        write_to_tar(lidar_sample, output_root / "lidar_raw" / f"{clip_id}.tar")


def main():
    parser = argparse.ArgumentParser(description="Convert Ruqi scene to RDS-HQ layout.")
    parser.add_argument(
        "--src-pkl",
        type=Path,
        required=True,
        help="Path to ruqi prefusion pickle (e.g., /ssd5/datasets/ruqi/scenes/ABC1_1735885669/prefusion_ruqi.pkl)",
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        default=None,
        help="Scene id inside the pickle. Default: first key in the pickle.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination root for RDS-HQ layout (scene subfolder will be created here).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root containing the Ruqi scene assets (images/pcd). Default: parent of the pickle's directory.",
    )
    parser.add_argument(
        "--keyframe-stride",
        type=int,
        default=5,
        help="Use every Nth frame for trusted annotations/LiDAR (e.g., 5 if labels are 2Hz and images 10Hz).",
    )
    parser.add_argument(
        "--clip-id",
        type=str,
        default=None,
        help="Clip id used in output filenames. Default: scene id.",
    )
    args = parser.parse_args()

    with open(args.src_pkl, "rb") as f:
        data = pickle.load(f)

    scene_id = args.scene_id or next(iter(data.keys()))
    clip_id = args.clip_id or scene_id
    scene = data[scene_id]

    dst_root = args.output_root / clip_id
    dst_root.mkdir(parents=True, exist_ok=True)

    data_root = args.data_root or args.src_pkl.parent.parent

    convert_scene(scene_id, scene, dst_root, clip_id, data_root, keyframe_stride=args.keyframe_stride)
    print(f"[OK] Converted {scene_id} -> {dst_root}")


if __name__ == "__main__":
    main()
