"""
Polished PyVista visualizer for the Canadarm2 reference-model RRT project.

Run from the project folder:
    python visualize_pyvista_polished.py

Educational demonstration only. This is not an official Canadarm2 model
and does not provide operational robot safety validation.
"""

import math
import random
from pathlib import Path as FilePath

import numpy as np
import pyvista as pv
import trimesh

import Pathplanner as planner


# --------------------------- Presentation settings ---------------------------

WINDOW_SIZE = (1400, 850)
SHOW_PATH_POSES = 5
MAX_RRT_ITERATIONS = 5000

BACKGROUND = "#050B18"
PANEL = "#0B1729"
TEXT = "#EAF3FF"
MUTED = "#8FA9C8"
CYAN = "#36D9FF"
BLUE = "#368BFF"
GREEN = "#43E0B0"
ORANGE = "#FFB454"
RED = "#FF536B"
PURPLE = "#A58BFF"

LINK_PALETTE = [
    "#D8E5F5", "#A8C7E8", "#7FB5D8", "#65D6E8",
    "#49C8B8", "#78A7FF", "#D5E2F0", "#FFFFFF",
]


# ----------------------- Robot and planning ---------------------






def load_demo():
    """Load the robot and let the user enter a spherical obstacle."""

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

    print("\n" + "=" * 55)
    print("CANADARM2 PATH PLANNER - OBSTACLE SETUP")
    print("=" * 55)
    print("Enter the centre of a spherical obstacle in metres.")
    print("Example: X=0.5, Y=0.2, Z=0.8, radius=0.1")
    print("The coordinates use the robot model's reference frame.\n")

    while True:
        try:
            x = float(input("Obstacle X (m): "))
            y = float(input("Obstacle Y (m): "))
            z = float(input("Obstacle Z (m): "))
            radius = float(input("Obstacle radius (m): "))

            values = np.array([x, y, z, radius], dtype=float)

            if not np.all(np.isfinite(values)):
                print("Please enter finite numbers only.\n")
                continue

            if radius <= 0:
                print("The obstacle radius must be greater than zero.\n")
                continue

            obstacle = planner.SphericalObstacle(
                np.array([x, y, z], dtype=float),
                radius,
            )

            if not planner.configuration_is_safe(model, start, obstacle):
                print(
                    "\nObstacle intersects the robot's start pose. "
                    "Try another position or a smaller radius.\n"
                )
                continue

            if not planner.configuration_is_safe(model, goal, obstacle):
                print(
                    "\nObstacle intersects the robot's goal pose. "
                    "Try another position or a smaller radius.\n"
                )
                continue

            break

        except ValueError as exc:
            print(f"Invalid input: {exc}")
            print("Please enter valid numbers and try again.\n")

    print("\nObstacle created successfully.")
    print(f"Centre: ({x:.3f}, {y:.3f}, {z:.3f}) m")
    print(f"Radius: {radius:.3f} m")
    print(f"Assumed link clearance: {planner.LINK_CLEARANCE:.3f} m")

    return model, start, goal, obstacle, urdf_path






# -------------------------- Mesh loading ------------------------

def resolve_mesh_path(filename, urdf_path):
    filename = str(filename)
    candidates = []

    if filename.startswith("package://"):
        remainder = filename[len("package://"):]
        candidates.append(FilePath(urdf_path).parent / remainder)

        parts = remainder.split("/", 1)
        if len(parts) == 2:
            candidates.append(
                FilePath(urdf_path).parent / parts[1]
            )
    else:
        path = FilePath(filename)

        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend([
                FilePath(urdf_path).parent / path,
                path,
            ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find mesh '{filename}'. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def visual_to_trimesh(visual, urdf_path):
    geometry = visual.geometry

    if geometry.mesh is not None:
        mesh_path = resolve_mesh_path(
            geometry.mesh.filename, urdf_path
        )

        loaded = trimesh.load(
            str(mesh_path),
            force="scene",
            process=False,
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
            sections=48,
        )

    if geometry.sphere is not None:
        return trimesh.creation.icosphere(
            subdivisions=3,
            radius=float(geometry.sphere.radius),
        )

    raise ValueError("Unsupported URDF visual geometry type.")


# ------------------------- Scene design helpers ----------------------

def add_hud(plotter, title, subtitle, status, metrics):
    """Display a clean, compact heads-up display."""

    plotter.add_text(
        title,
        position=(20, 810),
        font_size=13,
        color=TEXT,
        shadow=True,
    )

    plotter.add_text(
        subtitle,
        position=(20, 780),
        font_size=8,
        color=MUTED,
    )

    plotter.add_text(
        status,
        position=(1110, 810),
        font_size=10,
        color=GREEN if "PATH FOUND" in status else ORANGE,
        shadow=True,
    )

    metric_text = "\n".join(
        f"{key}:  {value}"
        for key, value in metrics.items()
    )

    plotter.add_text(
        metric_text,
        position=(20, 25),
        font_size=10,
        color=TEXT,
        shadow=True,
    )

    plotter.add_text(
        "SPACE ROBOTICS  |  RRT MOTION PLANNING",
        position=(1000, 20),
        font_size=8,
        color=MUTED,
    )


def add_starfield(plotter, bounds, count=550, seed=12):
    """Add subtle points for a space-inspired background."""

    rng = np.random.default_rng(seed)

    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    points = np.column_stack([
        rng.uniform(xmin, xmax, count),
        rng.uniform(ymin, ymax, count),
        rng.uniform(zmin, zmax, count),
    ])

    cloud = pv.PolyData(points)

    plotter.add_mesh(
        cloud,
        color="#8FB7E8",
        point_size=2,
        opacity=0.5,
        render_points_as_spheres=True,
    )


def add_floor_grid(plotter, bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    z = zmin + 0.02
    xs = np.linspace(xmin, xmax, 15)
    ys = np.linspace(ymin, ymax, 15)

    points = []
    lines = []

    for x in xs:
        i = len(points)
        points.extend([[x, ymin, z], [x, ymax, z]])
        lines.extend([2, i, i + 1])

    for y in ys:
        i = len(points)
        points.extend([[xmin, y, z], [xmax, y, z]])
        lines.extend([2, i, i + 1])

    grid = pv.PolyData(np.asarray(points))
    grid.lines = np.asarray(lines)

    plotter.add_mesh(
        grid,
        color="#18304A",
        line_width=1,
        opacity=0.5,
    )


def add_glowing_line(plotter, points, color=CYAN, width=5):
    """Layer a soft wide line beneath a crisp center line."""

    if len(points) < 2:
        return

    line = pv.lines_from_points(np.asarray(points))

    plotter.add_mesh(
        line,
        color=color,
        line_width=width * 3,
        opacity=0.12,
    )

    plotter.add_mesh(
        line,
        color=color,
        line_width=width,
        opacity=0.95,
    )


def add_robot_pose(
    plotter,
    model,
    q,
    urdf_path,
    opacity=1.0,
    label="Robot",
):
    cfg = {
        name: float(value)
        for name, value in zip(model.joint_names, q)
    }

    link_transforms = model.robot.link_fk(cfg=cfg)

    mesh_count = 0
    warnings = []

    for link_index, link in enumerate(model.robot.links):
        transform = link_transforms.get(link)

        if transform is None:
            continue

        link_color = LINK_PALETTE[
            link_index % len(LINK_PALETTE)
        ]

        for visual in link.visuals or []:
            try:
                mesh = visual_to_trimesh(visual, urdf_path)

                mesh.apply_transform(
                    transform @ visual.origin
                )

                plotter.add_mesh(
                    pv.wrap(mesh),
                    color=link_color,
                    opacity=opacity,
                    smooth_shading=True,
                    ambient=0.25,
                    diffuse=0.8,
                    specular=0.45,
                    specular_power=25,
                    show_edges=False,
                    label=label if mesh_count == 0 else None,
                )

                mesh_count += 1

            except Exception as exc:
                warnings.append(f"{link.name}: {exc}")

    # Fallback: draw the robot as connected segments if meshes fail.
    if mesh_count == 0:
        points = []

        for p1, p2 in model.link_segments(q):
            points.extend([p1, p2])

        if points:
            pts = np.asarray(points, dtype=float)
            lines = []

            for i in range(0, len(pts), 2):
                lines.extend([2, i, i + 1])

            polyline = pv.PolyData(pts)
            polyline.lines = np.asarray(lines)

            plotter.add_mesh(
                polyline,
                color=CYAN,
                line_width=5,
                point_size=10,
                render_points_as_spheres=True,
                label=f"{label} (line fallback)",
            )

    return mesh_count, warnings


def add_sphere(
    plotter,
    center,
    radius,
    color,
    opacity,
    label,
    style="surface",
):
    sphere = pv.Sphere(
        radius=float(radius),
        center=np.asarray(center, dtype=float),
        theta_resolution=48,
        phi_resolution=32,
    )

    plotter.add_mesh(
        sphere,
        color=color,
        opacity=opacity,
        style=style,
        line_width=2,
        smooth_shading=True,
        label=label,
    )


def build_rrt_tree_geometry(model, tree_edges, max_edges=1500):
    """
    Convert RRT joint-space edges into 3D end-effector positions.

    These lines visualize where the end effector moves between
    explored configurations. They are a 3D projection of the
    joint-space tree, not the robot's complete swept geometry.
    """

    if not tree_edges:
        return None, np.empty((0, 3))

    # Limit the number drawn so the scene remains responsive.
    edges_to_draw = tree_edges[:max_edges]

    position_cache = {}
    points = []
    lines = []

    def end_effector_position(q):
        # Reuse positions for configurations already encountered.
        key = tuple(np.asarray(q, dtype=float))

        if key not in position_cache:
            terminals = model.terminal_positions(q)

            if not terminals:
                return None

            position_cache[key] = np.asarray(
                next(iter(terminals.values())),
                dtype=float,
            )

        return position_cache[key]

    for parent_q, child_q in edges_to_draw:
        parent_pos = end_effector_position(parent_q)
        child_pos = end_effector_position(child_q)

        if parent_pos is None or child_pos is None:
            continue

        parent_index = len(points)
        points.append(parent_pos)

        child_index = len(points)
        points.append(child_pos)

        # PyVista line format: 2 points, then their point indices.
        lines.extend([2, parent_index, child_index])

    if not points:
        return None, np.empty((0, 3))

    points = np.asarray(points, dtype=float)

    tree_mesh = pv.PolyData(points)
    tree_mesh.lines = np.asarray(lines)

    return tree_mesh, points



# ------------------------------ Main ----------------------------

def main():
    pv.set_plot_theme("dark")

    # Make both Python and NumPy random sampling repeatable.
    random.seed(planner.RANDOM_SEED)
    np.random.seed(planner.RANDOM_SEED)

    model, start, goal, obstacle, urdf_path = load_demo()

    print(f"URDF: {urdf_path}")
    print(f"Movable joints: {len(model.joint_names)}")
    print("Joint names:", ", ".join(model.joint_names))
    print("\nPlanning a demonstration path...")

    baseline_safe = planner.direct_path_is_safe(
        model, start, goal, obstacle
    )


    path, tree_nodes, elapsed, tree_edges = planner.plan_path(
        model,
        start,
        goal,
        obstacle=obstacle,
        max_iterations=MAX_RRT_ITERATIONS,
        return_tree=True,
    )

    tree_mesh, tree_points = build_rrt_tree_geometry(
        model, tree_edges
    )


    verified = (
        planner.validate_path(model, path, obstacle)
        if path is not None
        else False
    )

    print(f"Direct path safe: {baseline_safe}")
    print(f"Planning time: {elapsed:.3f} s")
    print(f"RRT nodes: {tree_nodes}")
    print(f"Path found: {path is not None}")
    print(f"Path verified (simplified checks): {verified}")

    path_to_show = path if path is not None else [start]

    if path is not None:
        path_length = planner.calculate_path_length(path)

        print(f"Waypoints: {len(path)}")
        print(f"Joint-space path length: {path_length:.4f} rad")

    else:
        path_length = 0.0
        print("No path found; showing the start pose and target marker.")

    plotter = pv.Plotter(
        window_size=WINDOW_SIZE,
        lighting="three lights",
    )

    plotter.set_background(BACKGROUND)
    plotter.enable_anti_aliasing("ssaa")
    plotter.enable_eye_dome_lighting()

    # Determine scene bounds from the robot and obstacle.
    segment_points = []

    for q in [start, goal]:
        for p1, p2 in model.link_segments(q):
            segment_points.extend([p1, p2])

    if obstacle is not None:
        segment_points.append(
            np.asarray(obstacle.center, dtype=float)
        )
    

    # Include the explored tree in the camera's scene bounds.
    if len(tree_points) > 0:
        segment_points.extend(tree_points)


    points = (
        np.asarray(segment_points, dtype=float)
        if segment_points
        else np.array([[0, 0, 0]])
    )

    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    span = max(float(np.max(maxs - mins)), 1.0)
    center = (mins + maxs) / 2.0
    half = span * 0.9

    bounds = (
        center[0] - half,
        center[0] + half,
        center[1] - half,
        center[1] + half,
        mins[2] - 0.4 * span,
        maxs[2] + 0.35 * span,
    )

    add_starfield(plotter, bounds)
    add_floor_grid(plotter, bounds)


    # Draw the explored RRT branches before the robot and final route.
    if tree_mesh is not None:
        plotter.add_mesh(
            tree_mesh,
            color=BLUE,
            line_width=2,
            opacity=0.55,
            label="RRT explored branches",
        )

        plotter.add_mesh(
            pv.PolyData(tree_points),
            color=PURPLE,
            point_size=4,
            opacity=0.7,
            render_points_as_spheres=True,
            label="RRT explored configurations",
        )


    # Draw ghost poses along the planned trajectory.
    selected = sorted(set(np.linspace(
        0,
        len(path_to_show) - 1,
        min(SHOW_PATH_POSES, len(path_to_show)),
        dtype=int,
    ).tolist()))

    total_meshes = 0
    all_warnings = []

    for pose_num, path_index in enumerate(selected):
        if path is None:
            if pose_num > 0:
                continue
            opacity = 1.0
        else:
            opacity = (
                1.0
                if pose_num == len(selected) - 1
                else 0.17
            )

        count, warnings = add_robot_pose(
            plotter,
            model,
            path_to_show[path_index],
            urdf_path,
            opacity=opacity,
            label=(
                "Robot / planned pose"
                if pose_num == 0
                else f"Trajectory pose {pose_num + 1}"
            ),
        )

        total_meshes += count
        all_warnings.extend(warnings)

    # End-effector trajectory and endpoint markers.
    if path is not None and len(path_to_show) > 1:
        terminal_name = next(
            iter(model.terminal_positions(path_to_show[0]))
        )

        positions = np.asarray([
            model.terminal_positions(q)[terminal_name]
            for q in path_to_show
        ])

        add_glowing_line(
            plotter,
            positions,
            color=CYAN,
            width=5,
        )

        plotter.add_mesh(
            pv.PolyData(positions[0:1]),
            color=GREEN,
            point_size=18,
            render_points_as_spheres=True,
            label="Start",
        )

        plotter.add_mesh(
            pv.PolyData(positions[-1:]),
            color=PURPLE,
            point_size=18,
            render_points_as_spheres=True,
            label="Goal",
        )

    # Show the target end-effector position even if no path is found.
    goal_terminals = model.terminal_positions(goal)

    if goal_terminals:
        goal_pos = np.asarray(
            next(iter(goal_terminals.values())),
            dtype=float,
        )

        plotter.add_mesh(
            pv.PolyData(goal_pos.reshape(1, 3)),
            color=PURPLE,
            point_size=20,
            render_points_as_spheres=True,
            label="Goal end-effector position",
        )

    # Obstacle and assumed clearance boundary.
    if obstacle is not None:
        add_sphere(
            plotter,
            obstacle.center,
            obstacle.radius,
            RED,
            0.78,
            "Obstacle",
        )

        add_sphere(
            plotter,
            obstacle.center,
            obstacle.radius + planner.LINK_CLEARANCE,
            ORANGE,
            0.10,
            "Assumed clearance boundary",
        )

        add_sphere(
            plotter,
            obstacle.center,
            obstacle.radius + planner.LINK_CLEARANCE,
            ORANGE,
            0.9,
            "Clearance wireframe",
            style="wireframe",
        )

    status = (
        "● PATH FOUND"
        if path is not None and verified
        else "● NO VERIFIED PATH"
    )

    add_hud(
        plotter,
        "CANADARM2  /  MOTION PLANNER",
        "REFERENCE MODEL  •  RRT SEARCH  •  SIMPLIFIED COLLISION CHECKING",
        status,
        {
            "Joints": str(len(model.joint_names)),
            "RRT nodes": f"{tree_nodes:,}",
            "Plan time": f"{elapsed:.2f} s",
            "Waypoints": str(len(path) if path is not None else 0),
            "Path length": f"{path_length:.3f} rad",
            "Direct path": "SAFE" if baseline_safe else "BLOCKED",
        },
    )

    plotter.add_axes(
        line_width=2,
        color=TEXT,
        x_color="#FF7785",
        y_color="#55E6A5",
        z_color="#62B9FF",
    )

    plotter.show_bounds(
        bounds=bounds,
        grid="back",
        location="outer",
        all_edges=True,
        color="#29425F",
        xtitle="X (m)",
        ytitle="Y (m)",
        ztitle="Z (m)",
        font_size=8,
    )


    # Frame the scene before zooming.
    plotter.camera_position = "iso"
    plotter.reset_camera(bounds=bounds, render=False)
    plotter.camera.zoom(1.85)
    plotter.render()

    print(
        f"\nURDF visual meshes rendered across poses: {total_meshes}"
    )

    if all_warnings:
        print("\nMesh/geometry warnings (first 15):")

        for warning in all_warnings[:15]:
            print(" -", warning)

    print("\nClose the 3D window to exit.")
    plotter.show()


if __name__ == "__main__":
    main()