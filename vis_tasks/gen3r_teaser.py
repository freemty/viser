#!/usr/bin/env python3

"""Gen3R + SpaTracker-style teaser visualizer built on top of viser.
Example usage:
    python vis_tasks/gen3r_teaser.py --dataset-root assets/example_data/gen3r/re10k/113
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh

import viser
import viser.transforms as vtf

try:
    from matplotlib import cm as _mpl_cm
except ImportError:  # pragma: no cover - optional dependency
    _mpl_cm = None


_FALLBACK_SPECTRAL: List[Tuple[float, Tuple[int, int, int]]] = [
    (0.0, (158, 1, 66)),
    (0.1, (213, 62, 79)),
    (0.2, (244, 109, 67)),
    (0.3, (253, 174, 97)),
    (0.4, (254, 224, 139)),
    (0.5, (255, 255, 191)),
    (0.6, (230, 245, 152)),
    (0.7, (171, 221, 164)),
    (0.8, (102, 194, 165)),
    (0.9, (50, 136, 189)),
    (1.0, (94, 79, 162)),
]


def spectral_color(value: float) -> Tuple[int, int, int]:
    """Map [0, 1] -> RGB using matplotlib's Spectral colormap or a fallback."""
    value = float(np.clip(value, 0.0, 1.0))
    if _mpl_cm is not None:
        rgb = _mpl_cm.get_cmap("Spectral_r")(value)[:3]
        return tuple(int(round(255.0 * c)) for c in rgb)

    for (p0, c0), (p1, c1) in zip(_FALLBACK_SPECTRAL[:-1], _FALLBACK_SPECTRAL[1:]):
        if value <= p1:
            denom = (p1 - p0) if p1 > p0 else 1.0
            local_t = (value - p0) / denom
            return tuple(
                int(round(c0[channel] + (c1[channel] - c0[channel]) * local_t))
                for channel in range(3)
            )
    return _FALLBACK_SPECTRAL[-1][1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize Gen3R point clouds with SpaTracker-style camera poses and images."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/gen3r/re10k/113"),
        help="Root directory that contains cameras.txt, frames/, and pcds.ply.",
    )
    parser.add_argument(
        "--cameras-path",
        type=Path,
        default=None,
        help="Custom cameras.txt path. Defaults to <dataset_root>/cameras.txt.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Directory with frame images. Defaults to <dataset_root>/frames.",
    )
    parser.add_argument(
        "--pcd-path",
        type=Path,
        default=None,
        help="PLY point cloud path. Defaults to <dataset_root>/pcds.ply.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=4,
        help="Sample every N-th frame when visualizing camera poses.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional hard cap on the number of frames after downsampling.",
    )
    parser.add_argument(
        "--image-short-edge",
        type=int,
        default=320,
        help="Downsample images so their short edge matches this size (<=0 disables).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=-1,
        help="Randomly subsample the point cloud to this many points (<=0 keeps all).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.05,
        help="Point size (meters) for the point cloud visualization.",
    )
    parser.add_argument(
        "--frustum-scale",
        type=float,
        default=0.5,
        help="Scale factor passed to viser.scene.add_camera_frustum (0.15 mirrors docs).",
    )
    parser.add_argument(
        "--axes-length",
        type=float,
        default=0.15,
        help="Length of the camera frame axes in meters.",
    )
    parser.add_argument(
        "--axes-radius",
        type=float,
        default=0.005,
        help="Radius of the camera frame axes in meters.",
    )
    parser.add_argument(
        "--scene-scale",
        type=float,
        default=15.0,
        help="Global scale multiplier applied to both points and camera translations.",
    )
    parser.add_argument(
        "--camera-pos-scale",
        type=float,
        default=1.0,
        help="Additional multiplier applied only to camera translations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed for point cloud subsampling.",
    )
    return parser.parse_args()


def load_camera_matrices(
    cameras_txt: Path,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    lines = [
        line.strip()
        for line in cameras_txt.read_text().splitlines()
        if line.strip()
    ]
    if len(lines) % 2 != 0:
        raise ValueError(
            f"Expected an even number of lines in {cameras_txt}, got {len(lines)}."
        )
    midpoint = len(lines) // 2
    extr_lines = lines[:midpoint]
    intr_lines = lines[midpoint:]
    extrinsics: List[np.ndarray] = []
    intrinsics: List[np.ndarray] = []

    for idx, line in enumerate(extr_lines):
        values = np.fromstring(line, sep=" ")
        if values.size != 12:
            raise ValueError(
                f"Extrinsic line {idx} in {cameras_txt} does not have 12 floats."
            )
        extrinsics.append(values.reshape(3, 4))

    for idx, line in enumerate(intr_lines):
        values = np.fromstring(line, sep=" ")
        if values.size != 9:
            raise ValueError(
                f"Intrinsic line {idx} in {cameras_txt} does not have 9 floats."
            )
        intrinsics.append(values.reshape(3, 3))

    if len(extrinsics) != len(intrinsics):
        raise ValueError(
            f"Mismatch between extrinsics ({len(extrinsics)}) and intrinsics "
            f"({len(intrinsics)})."
        )
    return extrinsics, intrinsics


def _geometry_to_arrays(
    geometry: trimesh.base.Trimesh,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(geometry, trimesh.Scene):
        vertices: List[np.ndarray] = []
        colors: List[np.ndarray] = []
        for mesh in geometry.geometry.values():
            pts, cols = _geometry_to_arrays(mesh)
            vertices.append(pts)
            colors.append(cols)
        return np.concatenate(vertices, axis=0), np.concatenate(colors, axis=0)

    if isinstance(geometry, trimesh.Trimesh):
        pts = np.asarray(geometry.vertices)
        if (
            geometry.visual is not None
            and geometry.visual.vertex_colors is not None
            and len(geometry.visual.vertex_colors) == len(pts)
        ):
            cols = np.asarray(geometry.visual.vertex_colors)[:, :3]
        else:
            cols = np.full((len(pts), 3), 255, dtype=np.uint8)
        return pts, cols

    if isinstance(geometry, trimesh.points.PointCloud):
        pts = np.asarray(geometry.vertices)
        cols = (
            np.asarray(geometry.colors)[:, :3]
            if geometry.colors is not None
            else np.full((len(pts), 3), 255, dtype=np.uint8)
        )
        return pts, cols

    raise TypeError(f"Unsupported geometry type: {type(geometry)}")


def load_point_cloud(
    pcd_path: Path,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    geometry = trimesh.load(pcd_path, process=False)
    points, colors = _geometry_to_arrays(geometry)
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(points), size=max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
    return points.astype(np.float32), colors.astype(np.uint8)


def load_frame_image(
    frame_path: Path,
    target_short_edge: int,
) -> Tuple[np.ndarray, float]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - convenience error path.
        raise ImportError(
            "This script now depends on Pillow. Install it with `pip install pillow`."
        ) from exc

    with Image.open(frame_path) as img:
        rgb_img = img.convert("RGB")
        width, height = rgb_img.size
        scale = 1.0
        if target_short_edge > 0:
            short_edge = min(height, width)
            if short_edge > target_short_edge:
                scale = target_short_edge / short_edge
                resampling = getattr(Image, "Resampling", Image)
                resample_filter = getattr(resampling, "LANCZOS", Image.BICUBIC)
                new_size = (
                    max(1, int(round(width * scale))),
                    max(1, int(round(height * scale))),
                )
                rgb_img = rgb_img.resize(new_size, resample=resample_filter)
                width, height = rgb_img.size

        rgb = np.asarray(rgb_img, dtype=np.uint8)
    return np.ascontiguousarray(rgb), scale


def extrinsic_to_pose(extrinsic: np.ndarray) -> vtf.SE3:
    rotation_world_from_cam = vtf.SO3.from_matrix(extrinsic[:, :3]).inverse()
    translation_world_from_cam = -(rotation_world_from_cam @ extrinsic[:, 3])
    return vtf.SE3.from_rotation_and_translation(
        rotation_world_from_cam, translation_world_from_cam
    )


def collect_frame_paths(frames_dir: Path) -> List[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg")
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(frames_dir.glob(pattern))
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f"No frame images found under {frames_dir}.")
    return paths


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root
    cameras_path = args.cameras_path or dataset_root / "cameras.txt"
    frames_dir = args.frames_dir or dataset_root / "frames"
    pcd_path = args.pcd_path or dataset_root / "pcds.ply"
    if not pcd_path.exists():
        pcd_path = dataset_root / "pcds_denser_statfilter.ply"

    if args.frame_step <= 0:
        raise ValueError("--frame-step must be >= 1.")
    if not cameras_path.exists():
        raise FileNotFoundError(cameras_path)
    if not frames_dir.exists():
        raise FileNotFoundError(frames_dir)
    if not pcd_path.exists():
        raise FileNotFoundError(pcd_path)

    extrinsics, intrinsics = load_camera_matrices(cameras_path)
    frame_paths = collect_frame_paths(frames_dir)
    if len(frame_paths) != len(extrinsics):
        raise ValueError(
            "Frame count does not match camera parameter count "
            f"({len(frame_paths)} vs {len(extrinsics)})."
        )

    points_raw, colors = load_point_cloud(pcd_path, args.max_points, args.seed)

    server = viser.ViserServer()
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    server.gui.add_markdown(
        "### Gen3R teaser\n"
        "* Click any camera frustum to teleport the viewer.\n"
        "* Adjust sampling via --frame-step/--max-points.\n"
    )

    gui_scene_scale = server.gui.add_slider(
        "Scene scale",
        min=0.5,
        max=30.0,
        step=0.1,
        initial_value=args.scene_scale,
    )
    gui_camera_pos_scale = server.gui.add_slider(
        "Camera pos scale",
        min=0.1,
        max=10.0,
        step=0.1,
        initial_value=args.camera_pos_scale,
    )
    gui_point_size = server.gui.add_slider(
        "Point size",
        min=0.002,
        max=1.0,
        step=0.01,
        initial_value=args.point_size,
    )
    gui_camera_scale = server.gui.add_slider(
        "Camera scale",
        min=0.1,
        max=3,
        step=0.005,
        initial_value=args.frustum_scale,
    )

    point_cloud = server.scene.add_point_cloud(
        name="/gen3r/pcd",
        points=points_raw * gui_scene_scale.value,
        colors=colors,
        point_size=gui_point_size.value,
        point_shape="rounded",
    )

    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []
    frame_positions: List[np.ndarray] = []
    axes_radius_ratio = (
        args.axes_radius / args.axes_length if args.axes_length > 0 else 0.0333
    )

    def update_positions() -> None:
        scene_scale = gui_scene_scale.value
        cam_scale = gui_camera_pos_scale.value
        with server.atomic():
            point_cloud.points = points_raw * scene_scale
            for frame, base_pos in zip(frames, frame_positions):
                frame.position = base_pos * scene_scale * cam_scale

    @gui_point_size.on_update
    def _(_) -> None:
        point_cloud.point_size = gui_point_size.value

    @gui_scene_scale.on_update
    def _(_) -> None:
        update_positions()

    @gui_camera_pos_scale.on_update
    def _(_) -> None:
        update_positions()

    @gui_camera_scale.on_update
    def _(_) -> None:
        radius = gui_camera_scale.value * axes_radius_ratio
        for frame in frames:
            frame.axes_length = gui_camera_scale.value
            frame.axes_radius = radius
        for frustum in frustums:
            frustum.scale = gui_camera_scale.value

    selected_frames: List[Tuple[int, Path, np.ndarray, np.ndarray]] = []
    for idx, (frame_path, extrinsic, intrinsic) in enumerate(
        zip(frame_paths, extrinsics, intrinsics)
    ):
        if idx % args.frame_step != 0:
            continue
        selected_frames.append((idx, frame_path, extrinsic, intrinsic))
        if args.max_frames is not None and len(selected_frames) >= args.max_frames:
            break

    if not selected_frames:
        raise ValueError("No frames selected. Try reducing --frame-step or max filters.")

    total_selected = len(selected_frames)
    denom = max(1, total_selected - 1)

    for order, (idx, frame_path, extrinsic, intrinsic) in enumerate(selected_frames):
        color = spectral_color(order / denom)
        pose = extrinsic_to_pose(extrinsic)
        frame_name = f"/gen3r/frame_{idx:04d}"
        base_translation = pose.translation().copy()
        frame = server.scene.add_frame(
            frame_name,
            wxyz=pose.rotation().wxyz,
            position=base_translation
            * gui_scene_scale.value
            * gui_camera_pos_scale.value,
            axes_length=gui_camera_scale.value,
            axes_radius=gui_camera_scale.value * axes_radius_ratio,
            # origin_color=color,
        )
        frames.append(frame)
        frame_positions.append(base_translation)

        image, scale = load_frame_image(frame_path, args.image_short_edge)
        height, width = image.shape[:2]
        fy = intrinsic[1, 1] * scale
        fov = 2.0 * float(np.arctan2(height / 2.0, fy))
        aspect = width / height

        frustum = server.scene.add_camera_frustum(
            f"{frame_name}/frustum",
            fov=fov,
            aspect=aspect,
            scale=gui_camera_scale.value,
            image=image,
            color=color,
        )
        frustums.append(frustum)

        @frustum.on_click
        def _(event: viser.SceneNodePointerEvent, target_frame=frame) -> None:
            for client in server.get_clients().values():
                client.camera.wxyz = target_frame.wxyz
                client.camera.position = target_frame.position

    update_positions()

    url = f"http://{server.get_host()}:{server.get_port()}"
    print(f"Viser server is live at {url}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping viser server.")


if __name__ == "__main__":
    main()
