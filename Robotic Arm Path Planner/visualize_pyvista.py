"""
PyVista 3D visualizer for the Canadarm2 reference-model RRT project.

Run from the project folder:
    python visualize_pyvista.py

Educational demonstration only. This is not an official Canadarm2 model
and does not provide operational robot safety validation.
"""

import math
from pathlib import Path as FilePath

import numpy as np
import pyvista as pv
import trimesh

import Pathplanner as planner


# --------------------------- Settings ---------------------------

SHOW_PATH_POSES = 5
ROBOT_COLOR = "royalblue"
OBSTACLE_COLOR = "tomato"
CLEARANCE_COLOR = "orange"


# ----------------------- Robot and planning ---------------------

def load_demo():
    urdf_path = planner.obtain_default_urdf()
    model = planner.RobotModel(urdf_path)

    start = (model.lower + model.upper) / 2.0
    goal = start.copy()
    step = math.radians(45.0)

    if goal[0] + step <= model.upper[0]:
        goal[0] += step
    elif goal[0] - step >= model.lower[0]:
        goal[0] -= step
    else:
        raise ValueError("Could not create a demonstration target.")

    obstacle = planner.find_test_obstacle(
        model, start, goal, radius=0.10
    )

    return model, start, goal, obstacle, urdf_path


# -------------------------- Mesh loading ------------------------

def resolve_mesh_path(filename, urdf_path):
    """Resolve common relative and package:// URDF mesh references."""
    filename = str(filename)

    candidates = []
    if filename.startswith("package://"):
        # First try the package path relative to the URDF directory.
        remainder = filename[len("package://"):]
        candidates.append(FilePath(urdf_path).parent / remainder)
        # Some URDFs store meshes under the package's subdirectories.
        parts = remainder.split("/", 1)
        if len(parts) == 2:
            candidates.append(FilePath(urdf_path).parent / parts[1])
    else:
        path = FilePath(filename)
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(FilePath(urdf_path).parent / path)
            candidates.append(path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find mesh '{filename}'. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def visual_to_trimesh(visual, urdf_path):
    """Create a trimesh object from one URDF visual element."""
    geometry = visual.geometry

    if geometry.mesh is not None:
        mesh_path = resolve_mesh_path(
            geometry.mesh.filename, urdf_path
        )
        loaded = trimesh.load(
            str(mesh_path), force="scene", process=False
        )

        if isinstance(loaded, trimesh.Scene):
            meshes = [
                item.copy()
                for item in loaded.geometry.values()
                if isinstance(item, trimesh.Trimesh)
            ]
            if not meshes:
                raise ValueError(f"No triangle mesh in {mesh_path}")
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded.copy()
        else:
            raise ValueError(f"Unsupported mesh file: {mesh_path}")

        scale = getattr(geometry.mesh, "scale", None)
        if scale is not None:
            mesh.apply_scale(np.asarray(scale, dtype=float))
        return mesh

    if geometry.box is not None:
        return trimesh.creation.box(
            extents=np.asarray(geometry.box.size, dtype=float)
        )

    if geometry.cylinder is not None:
        return trimesh.creation.cylinder(
            radius=float(geometry.cylinder.radius),
            height=float(geometry.cylinder.length),
            sections=32,
        )

    if geometry.sphere is not None:
        return trimesh.creation.icosphere(
            subdivisions=2,
            radius=float(geometry.sphere.radius),
        )

    raise ValueError("Unsupported URDF visual geometry type.")


# ------------------------- PyVista helpers ----------------------

def add_robot_pose(plotter, model, q, urdf_path, color, opacity=1.0,
                   label="Robot"):
    """Render URDF visuals at configuration q; return counts and warnings."""
    cfg = {
        name: float(value)
        for name, value in zip(model.joint_names, q)
    }
    link_transforms = model.robot.link_fk(cfg=cfg)

    mesh_count = 0
    warnings = []

    for link in model.robot.links:
        link_transform = link_transforms.get(link)
        if link_transform is None:
            continue

        for visual in link.visuals or []:
            try:
                mesh = visual_to_trimesh(visual, urdf_path)
                mesh.apply_transform(link_transform @ visual.origin)

                poly = pv.wrap(mesh)
                plotter.add_mesh(
                    poly,
                    color=color,
                    opacity=opacity,
                    smooth_shading=True,
                    show_edges=False,
                    label=label if mesh_count == 0 else None,
                )
                mesh_count += 1
            except Exception as exc:
                warnings.append(f"{link.name}: {exc}")

    # Use line segments as a fallback if no visual geometry could be rendered.
    if mesh_count == 0:
        points = []
        for p1, p2 in model.link_segments(q):
            points.extend([p1, p2])
        if points:
            line_points = np.asarray(points, dtype=float)
            lines = []
            for i in range(0, len(line_points), 2):
                lines.extend([2, i, i + 1])
            polyline = pv.PolyData(line_points)
            polyline.lines = np.asarray(lines)
            plotter.add_mesh(
                polyline, color=color, line_width=5,
                point_size=8, render_points_as_spheres=True,
                label=f"{label} (line fallback)"
            )

    return mesh_count, warnings


def add_sphere(plotter, center, radius, color, opacity, label):
    sphere = pv.Sphere(
        radius=float(radius),
        center=np.asarray(center, dtype=float),
        theta_resolution=36,
        phi_resolution=24,
    )
    plotter.add_mesh(
        sphere, color=color, opacity=opacity, label=label
    )


# ------------------------------ Main ----------------------------

def main():
    pv.set_plot_theme("document")

    model, start, goal, obstacle, urdf_path = load_demo()

    print(f"URDF: {urdf_path}")
    print(f"Movable joints: {len(model.joint_names)}")
    print("Joint names:", ", ".join(model.joint_names))

    print("\nPlanning a demonstration path...")
    baseline_safe = planner.direct_path_is_safe(
        model, start, goal, obstacle
    )
    path, tree_nodes, elapsed = planner.plan_path(
        model,
        start,
        goal,
        obstacle=obstacle,
        max_iterations=5000,
    )
    verified = (
        planner.validate_path(model, path, obstacle)
        if path is not None else False
    )

    print(f"Direct path safe: {baseline_safe}")
    print(f"Planning time: {elapsed:.3f} s")
    print(f"RRT nodes: {tree_nodes}")
    print(f"Path found: {path is not None}")
    print(f"Path verified (simplified checks): {verified}")

    if path is None:
        path_to_show = [start]
        print("No path found; displaying the starting configuration.")
    else:
        path_to_show = path
        print(f"Waypoints: {len(path)}")
        print(
            "Joint-space path length: "
            f"{planner.calculate_path_length(path):.4f} rad"
        )

    plotter = pv.Plotter(window_size=(1400, 900))
    plotter.set_background("#f4f7fb")
    plotter.add_axes()
    plotter.show_grid(
        xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)"
    )

    # Draw a small number of arm poses to indicate motion.
    selected = sorted(set(
        np.linspace(
            0,
            len(path_to_show) - 1,
            min(SHOW_PATH_POSES, len(path_to_show)),
            dtype=int,
        ).tolist()
    ))

    pose_colors = [
        "#1677c8", "#6baed6", "#36a89a", "#f0a34a", "#b45ac4"
    ]
    total_meshes = 0
    all_warnings = []

    for pose_num, path_index in enumerate(selected):
        is_endpoint = pose_num in (0, len(selected) - 1)
        count, warnings = add_robot_pose(
            plotter,
            model,
            path_to_show[path_index],
            urdf_path,
            color=pose_colors[pose_num % len(pose_colors)],
            opacity=1.0 if is_endpoint else 0.24,
            label="Robot pose" if pose_num == 0 else f"Path pose {pose_num + 1}",
        )
        total_meshes += count
        all_warnings.extend(warnings)

    # Plot end-effector path.
    if len(path_to_show) > 1:
        terminal_name = next(
            iter(model.terminal_positions(path_to_show[0]))
        )
        positions = np.asarray([
            model.terminal_positions(q)[terminal_name]
            for q in path_to_show
        ])
        if len(positions) >= 2:
            trajectory = pv.lines_from_points(positions)
            plotter.add_mesh(
                trajectory,
                color="#d62728",
                line_width=5,
                label="End-effector trajectory",
            )
            plotter.add_mesh(
                pv.PolyData(positions[[0, -1]]),
                color="#222222",
                point_size=12,
                render_points_as_spheres=True,
                label="Trajectory endpoints",
            )

    if obstacle is not None:
        add_sphere(
            plotter, obstacle.center, obstacle.radius,
            OBSTACLE_COLOR, 0.55, "Obstacle"
        )
        add_sphere(
            plotter,
            obstacle.center,
            obstacle.radius + planner.LINK_CLEARANCE,
            CLEARANCE_COLOR,
            0.12,
            "Assumed clearance boundary",
        )

    plotter.add_legend()
    plotter.add_text(
        "Space Robotic Arm Path Planner\n"
        "URDF visual geometry + RRT path\n"
        "Educational reference model; simplified collision checking",
        position="upper_left",
        font_size=11,
        color="#172033",
    )
    plotter.camera_position = "iso"
    plotter.reset_camera()

    print(f"\nURDF visual meshes rendered across poses: {total_meshes}")
    if all_warnings:
        print("\nMesh/geometry warnings:")
        for warning in all_warnings[:30]:
            print(" -", warning)
    if total_meshes == 0:
        print(
            "\nWARNING: No URDF visual meshes were rendered. "
            "See warnings above; the display may use line fallback."
        )

    print("\nClose the 3D window to exit.")
    plotter.show()


if __name__ == "__main__":
    main()
