
"""
Canadarm2-Inspired Space Robotic Arm Motion Planner
===================================================

Single-file educational demonstration featuring:
- URDF model loading and joint limits
- Forward kinematics
- Joint-space RRT path planning
- Spherical obstacle avoidance
- Sampled path validation
- PyVista 3D rendering of URDF visual meshes
- Trajectory visualization and planning metrics

DISCLAIMER:
This uses a community reference URDF and simplified collision checks.
It is not an official Canadarm2 model or an operational safety system.
"""

import argparse
import fractions
import math
import random
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

# Compatibility for older urdfpy dependencies.
if not hasattr(fractions, "gcd"):
    fractions.gcd = math.gcd

for _name, _value in (
    ("int", int),
    ("float", float),
    ("bool", bool),
    ("object", object),
):
    if _name not in np.__dict__:
        setattr(np, _name, _value)

from urdfpy import URDF


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42
MAX_ITERATIONS = 5000
GOAL_BIAS = 0.15
STEP_SIZE = 0.25
GOAL_TOLERANCE = 0.30
EDGE_RESOLUTION = 0.05

# Educational assumption, not a real robot specification.
LINK_CLEARANCE = 0.15

DEFAULT_OBSTACLE_RADIUS = 0.10

WINDOW_SIZE = (1600, 1000)
SHOW_PATH_POSES = 5

BACKGROUND = "#050B18"
PANEL = "#0B1729"
TEXT = "#EAF3FF"
MUTED = "#8FA9C8"
CYAN = "#36D9FF"
GREEN = "#43E0B0"
ORANGE = "#FFB454"
RED = "#FF536B"
PURPLE = "#A58BFF"

LINK_PALETTE = [
    "#D8E5F5", "#A8C7E8", "#7FB5D8", "#65D6E8",
    "#49C8B8", "#78A7FF", "#D5E2F0", "#FFFFFF",
]

MODEL_REPOSITORY_ZIP = (
    "https://github.com/vyas-shubham/TraceableRobotModels/"
    "archive/refs/heads/master.zip"
)


# ============================================================
# URDF LOADING
# ============================================================

def find_canadarm2_urdf(search_root):
    """Find a Canadarm2 URDF in a directory tree."""
    search_root = Path(search_root)

    if not search_root.exists():
        return None

    files = sorted(search_root.rglob("*.urdf"))

    if not files:
        return None

    preferred = [
        file for file in files
        if "ssrms_canadarm2" in str(file).lower()
        or "canadarm2" in str(file).lower()
    ]

    return (preferred or files)[0]


def obtain_default_urdf():
    """Find the local reference URDF or download its repository."""
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "robot_models"
    extract_dir = models_dir / "TraceableRobotModels-master"

    existing = find_canadarm2_urdf(models_dir)

    if existing is not None:
        print(f"Using previously downloaded URDF: {existing}")
        return existing

    models_dir.mkdir(parents=True, exist_ok=True)
    archive_path = models_dir / "TraceableRobotModels-master.zip"

    print("Downloading public robot reference-model repository...")

    request = urllib.request.Request(
        MODEL_REPOSITORY_ZIP,
        headers={"User-Agent": "Educational-Space-Robotics-Project"},
    )

    with urllib.request.urlopen(request, timeout=45) as response:
        archive_path.write_bytes(response.read())

    destination = models_dir.resolve()

    with zipfile.ZipFile(archive_path, "r") as archive:
        # Reject ZIP entries that attempt to escape the model directory.
        for member in archive.infolist():
            target = (models_dir / member.filename).resolve()

            if (
                target != destination
                and destination not in target.parents
            ):
                raise RuntimeError("Unsafe path in downloaded ZIP.")

        archive.extractall(models_dir)

    result = find_canadarm2_urdf(extract_dir)

    if result is None:
        result = find_canadarm2_urdf(models_dir)

    if result is None:
        raise FileNotFoundError(
            f"No URDF was found in {models_dir}"
        )

    print(f"Downloaded reference URDF: {result}")
    print("This is a community model, not an official operational model.")

    return result


# ============================================================
# MATH AND JOINT-SPACE UTILITIES
# ============================================================

def point_to_segment_distance(point, start, end):
    """Shortest Euclidean distance between a point and a line segment."""
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    segment = end - start
    length_squared = float(np.dot(segment, segment))

    if length_squared < 1e-12:
        return float(np.linalg.norm(point - start))

    t = float(np.dot(point - start, segment) / length_squared)
    t = np.clip(t, 0.0, 1.0)

    closest = start + t * segment
    return float(np.linalg.norm(point - closest))


def interpolate(q1, q2, fraction):
    """Linear interpolation between two joint configurations."""
    return q1 + fraction * (q2 - q1)


def joint_distance(q1, q2):
    """Euclidean distance in joint space, in radians."""
    return float(np.linalg.norm(np.asarray(q2) - np.asarray(q1)))


# ============================================================
# ROBOT MODEL AND FORWARD KINEMATICS
# ============================================================

class RobotModel:
    """URDF robot wrapper for kinematics and joint-limit checks."""

    def __init__(self, urdf_path):
        urdf_path = Path(urdf_path)

        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        self.urdf_path = urdf_path
        self.robot = URDF.load(str(urdf_path))

        self.joints = [
            joint for joint in self.robot.joints
            if joint.joint_type != "fixed"
        ]

        if not self.joints:
            raise ValueError("No movable joints were found.")

        self.joint_names = [joint.name for joint in self.joints]

        lower = []
        upper = []

        for joint in self.joints:
            limit = joint.limit

            if joint.joint_type == "continuous":
                low, high = -math.pi, math.pi
            elif (
                limit is not None
                and limit.lower is not None
                and limit.upper is not None
            ):
                low = float(limit.lower)
                high = float(limit.upper)
            else:
                # Assumed fallback, not verified robot limits.
                low, high = -math.pi, math.pi
                print(
                    f"WARNING: Assuming +/-180 degrees for "
                    f"joint {joint.name}"
                )

            if low >= high:
                raise ValueError(f"Invalid limits for {joint.name}")

            lower.append(low)
            upper.append(high)

        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)

        self.connections = [
            (
                self.robot.link_map[joint.parent],
                self.robot.link_map[joint.child],
            )
            for joint in self.robot.joints
        ]

    def config_dict(self, q):
        return {
            name: float(value)
            for name, value in zip(self.joint_names, q)
        }

    def valid_configuration(self, q):
        q = np.asarray(q, dtype=float)

        if q.shape != self.lower.shape or not np.all(np.isfinite(q)):
            return False

        return bool(
            np.all(q >= self.lower - 1e-9)
            and np.all(q <= self.upper + 1e-9)
        )

    def forward_kinematics(self, q):
        if not self.valid_configuration(q):
            raise ValueError("Invalid joint configuration.")

        return self.robot.link_fk(cfg=self.config_dict(q))

    def link_segments(self, q):
        """
        Approximate links as lines connecting joint-frame origins.
        This is not exact mesh-based collision detection.
        """
        transforms = self.forward_kinematics(q)
        segments = []

        for parent, child in self.connections:
            p1 = transforms[parent][:3, 3]
            p2 = transforms[child][:3, 3]
            segments.append((p1.copy(), p2.copy()))

        return segments

    def terminal_positions(self, q):
        transforms = self.forward_kinematics(q)

        return {
            link.name: transforms[link][:3, 3].copy()
            for link in self.robot.end_links
        }

    def report(self):
        print("\n" + "=" * 65)
        print("ROBOT MODEL REPORT")
        print("=" * 65)
        print(f"Model: {self.robot.name}")
        print(f"Base link: {self.robot.base_link.name}")
        print(f"Links: {len(self.robot.links)}")
        print(f"URDF joints: {len(self.robot.joints)}")
        print(f"Movable joints: {len(self.joints)}")

        print("\nJoint limits:")

        for joint, low, high in zip(
            self.joints, self.lower, self.upper
        ):
            print(
                f"  {joint.name}: "
                f"{math.degrees(low):.1f} to "
                f"{math.degrees(high):.1f} degrees "
                f"({joint.joint_type})"
            )

        print("\nTerminal links:")
        for link in self.robot.end_links:
            print(f"  {link.name}")


# ============================================================
# OBSTACLES AND COLLISION CHECKING
# ============================================================

class SphericalObstacle:
    """Spherical keep-out zone in the robot base coordinate frame."""

    def __init__(self, center, radius):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)

        if self.center.shape != (3,):
            raise ValueError("Obstacle centre must have three coordinates.")

        if not np.all(np.isfinite(self.center)):
            raise ValueError("Obstacle coordinates must be finite.")

        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("Obstacle radius must be positive.")

    def collides(self, model, q):
        for start, end in model.link_segments(q):
            distance = point_to_segment_distance(
                self.center, start, end
            )

            if distance <= self.radius + LINK_CLEARANCE:
                return True

        return False


def configuration_is_safe(model, q, obstacle=None):
    if not model.valid_configuration(q):
        return False

    try:
        return obstacle is None or not obstacle.collides(model, q)
    except Exception:
        return False


def edge_is_safe(model, q1, q2, obstacle=None):
    """Sample a joint-space segment for collisions and joint limits."""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    if not model.valid_configuration(q1):
        return False

    if not model.valid_configuration(q2):
        return False

    maximum_change = float(np.max(np.abs(q2 - q1)))
    sample_count = max(
        2,
        int(math.ceil(maximum_change / EDGE_RESOLUTION)) + 1,
    )

    for fraction in np.linspace(0.0, 1.0, sample_count):
        q = interpolate(q1, q2, fraction)

        if not configuration_is_safe(model, q, obstacle):
            return False

    return True


def find_test_obstacle(model, start, goal, radius=0.10):
    """
    Search for a sphere that blocks the direct path while keeping
    the start and goal clear. Returns None if none is found.
    """
    candidates = []
    endpoint_segments = (
        model.link_segments(start),
        model.link_segments(goal),
    )

    for fraction in np.linspace(0.05, 0.95, 91):
        q = interpolate(start, goal, fraction)

        for p1, p2 in model.link_segments(q):
            for t in np.linspace(0.15, 0.85, 8):
                point = p1 + t * (p2 - p1)

                endpoint_distance = min(
                    point_to_segment_distance(point, a, b)
                    for segments in endpoint_segments
                    for a, b in segments
                )

                if endpoint_distance > radius + LINK_CLEARANCE + 0.02:
                    candidates.append((endpoint_distance, point.copy()))

    candidates.sort(key=lambda item: item[0], reverse=True)

    for _, center in candidates:
        obstacle = SphericalObstacle(center, radius)

        if (
            configuration_is_safe(model, start, obstacle)
            and configuration_is_safe(model, goal, obstacle)
            and not edge_is_safe(model, start, goal, obstacle)
        ):
            return obstacle

    return None


# ============================================================
# RRT PATH PLANNER
# ============================================================

class TreeNode:
    def __init__(self, q, parent=None):
        self.q = np.asarray(q, dtype=float)
        self.parent = parent


def nearest_node(nodes, target):
    distances = [joint_distance(node.q, target) for node in nodes]
    return nodes[int(np.argmin(distances))]


def steer(q_start, q_target, step_size):
    difference = q_target - q_start
    distance = float(np.linalg.norm(difference))

    if distance <= step_size:
        return q_target.copy()

    return q_start + difference * (step_size / distance)


def reconstruct_path(node):
    path = []

    while node is not None:
        path.append(node.q.copy())
        node = node.parent

    return list(reversed(path))


def plan_path(
    model,
    start,
    goal,
    obstacle=None,
    max_iterations=MAX_ITERATIONS,
):
    """Plan a route using a rapidly-exploring random tree (RRT)."""
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    if not model.valid_configuration(start):
        raise ValueError("Start configuration violates joint limits.")

    if not model.valid_configuration(goal):
        raise ValueError("Goal configuration violates joint limits.")

    if not configuration_is_safe(model, start, obstacle):
        raise ValueError("Start configuration is not collision-free.")

    if not configuration_is_safe(model, goal, obstacle):
        raise ValueError("Goal configuration is not collision-free.")

    started = time.perf_counter()

    if edge_is_safe(model, start, goal, obstacle):
        return [start.copy(), goal.copy()], 1, time.perf_counter() - started

    nodes = [TreeNode(start)]

    for _ in range(max_iterations):
        if random.random() < GOAL_BIAS:
            sample = goal.copy()
        else:
            sample = np.random.uniform(model.lower, model.upper)

        nearest = nearest_node(nodes, sample)
        candidate = steer(nearest.q, sample, STEP_SIZE)

        if not edge_is_safe(model, nearest.q, candidate, obstacle):
            continue

        new_node = TreeNode(candidate, nearest)
        nodes.append(new_node)

        if joint_distance(candidate, goal) <= GOAL_TOLERANCE:
            if edge_is_safe(model, candidate, goal, obstacle):
                goal_node = TreeNode(goal.copy(), new_node)
                elapsed = time.perf_counter() - started

                return (
                    reconstruct_path(goal_node),
                    len(nodes),
                    elapsed,
                )

    return None, len(nodes), time.perf_counter() - started


def calculate_path_length(path):
    if path is None or len(path) < 2:
        return 0.0

    return sum(
        joint_distance(a, b)
        for a, b in zip(path[:-1], path[1:])
    )


def validate_path(model, path, obstacle=None):
    if path is None or len(path) < 2:
        return False

    if not all(
        configuration_is_safe(model, q, obstacle)
        for q in path
    ):
        return False

    return all(
        edge_is_safe(model, a, b, obstacle)
        for a, b in zip(path[:-1], path[1:])
    )


# ============================================================
# PYVISTA MESH AND SCENE HELPERS
# ============================================================

def resolve_mesh_path(filename, urdf_path):
    filename = str(filename)
    urdf_dir = Path(urdf_path).parent
    candidates = []

    if filename.startswith("package://"):
        remainder = filename[len("package://"):]
        candidates.append(urdf_dir / remainder)

        parts = remainder.split("/", 1)
        if len(parts) == 2:
            candidates.append(urdf_dir / parts[1])
    else:
        mesh_path = Path(filename)

        if mesh_path.is_absolute():
            candidates.append(mesh_path)
        else:
            candidates.extend([urdf_dir / mesh_path, mesh_path])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not resolve mesh: {filename}")


def visual_to_trimesh(visual, urdf_path):
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
            sections=48,
        )

    if geometry.sphere is not None:
        return trimesh.creation.icosphere(
            subdivisions=3,
            radius=float(geometry.sphere.radius),
        )

    raise ValueError("Unsupported URDF visual geometry.")


def add_robot_pose(plotter, model, q, urdf_path, opacity=1.0, label="Robot"):
    """Render URDF visual meshes at one robot configuration."""
    transforms = model.forward_kinematics(q)
    mesh_count = 0
    warnings = []

    for link_index, link in enumerate(model.robot.links):
        transform = transforms.get(link)

        if transform is None:
            continue

        color = LINK_PALETTE[link_index % len(LINK_PALETTE)]

        for visual in link.visuals or []:
            try:
                mesh = visual_to_trimesh(visual, urdf_path)
                mesh.apply_transform(transform @ visual.origin)

                plotter.add_mesh(
                    pv.wrap(mesh),
                    color=color,
                    opacity=opacity,
                    smooth_shading=True,
                    ambient=0.25,
                    diffuse=0.8,
                    specular=0.45,
                    specular_power=25,
                    label=label if mesh_count == 0 else None,
                )

                mesh_count += 1

            except Exception as exc:
                warnings.append(f"{link.name}: {exc}")

    # Simple fallback if the URDF visual meshes cannot be rendered.
    if mesh_count == 0:
        points = []

        for p1, p2 in model.link_segments(q):
            points.extend([p1, p2])

        if points:
            points = np.asarray(points, dtype=float)
            lines = np.asarray([
                value
                for i in range(0, len(points), 2)
                for value in (2, i, i + 1)
            ])

            polyline = pv.PolyData(points)
            polyline.lines = lines

            plotter.add_mesh(
                polyline,
                color=CYAN,
                line_width=5,
                point_size=10,
                render_points_as_spheres=True,
                label="Robot link fallback",
            )

    return mesh_count, warnings


def add_sphere(plotter, center, radius, color, opacity, label, style="surface"):
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


def add_glowing_line(plotter, points, color=CYAN, width=5):
    if len(points) < 2:
        return

    line = pv.lines_from_points(np.asarray(points))

    plotter.add_mesh(
        line, color=color, line_width=width * 3, opacity=0.12
    )
    plotter.add_mesh(
        line, color=color, line_width=width, opacity=0.95
    )


def add_starfield(plotter, bounds, count=450, seed=12):
    rng = np.random.default_rng(seed)

    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    points = np.column_stack([
        rng.uniform(xmin, xmax, count),
        rng.uniform(ymin, ymax, count),
        rng.uniform(zmin, zmax, count),
    ])

    plotter.add_mesh(
        pv.PolyData(points),
        color="#8FB7E8",
        point_size=2,
        opacity=0.5,
        render_points_as_spheres=True,
    )


def add_floor_grid(plotter, bounds):
    xmin, xmax, ymin, ymax, zmin, _ = bounds
    z = zmin + 0.02

    points = []
    lines = []

    for x in np.linspace(xmin, xmax, 15):
        i = len(points)
        points.extend([[x, ymin, z], [x, ymax, z]])
        lines.extend([2, i, i + 1])

    for y in np.linspace(ymin, ymax, 15):
        i = len(points)
        points.extend([[xmin, y, z], [xmax, y, z]])
        lines.extend([2, i, i + 1])

    grid = pv.PolyData(np.asarray(points))
    grid.lines = np.asarray(lines)

    plotter.add_mesh(
        grid, color="#18304A", line_width=1, opacity=0.5
    )


def add_hud(plotter, title, subtitle, status, metrics):
    plotter.add_text(
        title,
        position="upper_left",
        font_size=19,
        color=TEXT,
        shadow=True,
    )

    plotter.add_text(
        subtitle,
        position=(12, 76),
        font_size=10,
        color=MUTED,
    )

    plotter.add_text(
        status,
        position="upper_right",
        font_size=12,
        color=GREEN if "PATH FOUND" in status else ORANGE,
        shadow=True,
    )

    metric_text = "\n".join(
        f"{name:<18} {value}"
        for name, value in metrics.items()
    )

    plotter.add_text(
        metric_text,
        position="lower_left",
        font_size=10,
        color=TEXT,
        shadow=True,
    )

    plotter.add_text(
        "SPACE ROBOTICS  /  MOTION PLANNING DEMO",
        position="lower_right",
        font_size=9,
        color=MUTED,
    )


# ============================================================
# DEMONSTRATION SETUP
# ============================================================

def create_demo(model, obstacle_argument=None, radius=DEFAULT_OBSTACLE_RADIUS):
    """Create repeatable start and goal configurations."""
    start = (model.lower + model.upper) / 2.0
    goal = start.copy()

    step = math.radians(45.0)
    changed = False

    for index in range(len(goal)):
        if goal[index] + step <= model.upper[index]:
            goal[index] += step
            changed = True
            break

        if goal[index] - step >= model.lower[index]:
            goal[index] -= step
            changed = True
            break

    if not changed:
        raise ValueError("Could not create a demonstration goal.")

    if obstacle_argument is not None:
        obstacle = SphericalObstacle(obstacle_argument, radius)
    else:
        obstacle = find_test_obstacle(model, start, goal, radius)

    return start, goal, obstacle


# ============================================================
# MAIN VISUALIZATION
# ============================================================

def visualize(model, start, goal, path, obstacle, metrics, urdf_path):
    pv.set_plot_theme("dark")

    plotter = pv.Plotter(
        window_size=WINDOW_SIZE,
        lighting="three lights",
    )

    plotter.set_background(BACKGROUND)
    plotter.enable_anti_aliasing("ssaa")

    segment_points = []

    for q in [start, goal]:
        for p1, p2 in model.link_segments(q):
            segment_points.extend([p1, p2])

    if obstacle is not None:
        segment_points.append(obstacle.center)

    points = np.asarray(segment_points, dtype=float)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = max(float(np.max(maxs - mins)), 1.0)
    center = (mins + maxs) / 2.0
    half = span * 0.9

    bounds = (
        center[0] - half, center[0] + half,
        center[1] - half, center[1] + half,
        mins[2] - 0.4 * span, maxs[2] + 0.35 * span,
    )

    add_starfield(plotter, bounds)
    add_floor_grid(plotter, bounds)

    path_to_show = path if path is not None else [start]

    selected = sorted(set(
        np.linspace(
            0,
            len(path_to_show) - 1,
            min(SHOW_PATH_POSES, len(path_to_show)),
            dtype=int,
        ).tolist()
    ))

    total_meshes = 0
    warnings = []

    for pose_number, path_index in enumerate(selected):
        if path is None and pose_number > 0:
            continue

        opacity = (
            1.0
            if path is None or pose_number == len(selected) - 1
            else 0.17
        )

        count, pose_warnings = add_robot_pose(
            plotter,
            model,
            path_to_show[path_index],
            urdf_path,
            opacity=opacity,
            label="Robot / planned pose",
        )

        total_meshes += count
        warnings.extend(pose_warnings)

    # End-effector path.
    if path is not None and len(path) > 1:
        terminals = [
            model.terminal_positions(q)
            for q in path
        ]

        terminal_name = next(iter(terminals[0]))

        positions = np.asarray([
            item[terminal_name]
            for item in terminals
        ])

        add_glowing_line(plotter, positions)

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

    # Goal marker.
    goal_positions = model.terminal_positions(goal)

    if goal_positions:
        goal_position = np.asarray(
            next(iter(goal_positions.values()))
        )

        plotter.add_mesh(
            pv.PolyData(goal_position.reshape(1, 3)),
            color=PURPLE,
            point_size=20,
            render_points_as_spheres=True,
            label="Goal end-effector position",
        )

    # Obstacle and assumed clearance envelope.
    if obstacle is not None:
        add_sphere(
            plotter, obstacle.center, obstacle.radius,
            RED, 0.78, "Obstacle",
        )

        add_sphere(
            plotter,
            obstacle.center,
            obstacle.radius + LINK_CLEARANCE,
            ORANGE,
            0.10,
            "Assumed clearance boundary",
        )

        add_sphere(
            plotter,
            obstacle.center,
            obstacle.radius + LINK_CLEARANCE,
            ORANGE,
            0.9,
            "Clearance wireframe",
            style="wireframe",
        )

    status = (
        "● PATH FOUND"
        if path is not None and metrics["verified"]
        else "● NO VERIFIED PATH"
    )

    add_hud(
        plotter,
        "CANADARM2  /  MOTION PLANNER",
        "REFERENCE MODEL  •  RRT SEARCH  •  SIMPLIFIED COLLISION CHECKS",
        status,
        {
            "Joints": str(len(model.joint_names)),
            "RRT nodes": f'{metrics["nodes"]:,}',
            "Plan time": f'{metrics["elapsed"]:.3f} s',
            "Waypoints": str(len(path) if path is not None else 0),
            "Path length": f'{metrics["length"]:.3f} rad',
            "Direct path": "SAFE" if metrics["direct_safe"] else "BLOCKED",
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

    plotter.add_legend(
        bcolor=PANEL,
        border=True,
        size=(0.20, 0.20),
        loc="upper_right",
    )

    plotter.camera_position = "iso"
    plotter.camera.zoom(1.15)
    plotter.reset_camera()
    plotter.camera.zoom(1.15)

    print(f"\nURDF visual meshes rendered: {total_meshes}")

    if warnings:
        print("\nMesh warnings (first 15):")
        for warning in warnings[:15]:
            print(" -", warning)

    print("\nClose the 3D window to exit.")
    plotter.show()


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Canadarm2-inspired educational motion planner."
    )

    parser.add_argument(
        "urdf_path",
        nargs="?",
        default=None,
        help="Optional path to a local URDF.",
    )

    parser.add_argument(
        "--obstacle",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Specify a spherical obstacle centre in metres.",
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_OBSTACLE_RADIUS,
        help="Obstacle radius in metres.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=MAX_ITERATIONS,
        help="Maximum RRT iterations.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    if args.radius <= 0:
        parser.error("--radius must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        urdf_path = (
            Path(args.urdf_path)
            if args.urdf_path
            else obtain_default_urdf()
        )

        model = RobotModel(urdf_path)
        model.report()

        start, goal, obstacle = create_demo(
            model,
            obstacle_argument=args.obstacle,
            radius=args.radius,
        )

    except Exception as exc:
        print(f"\nSetup error: {exc}")
        return

    if obstacle is None:
        print(
            "\nNo suitable automatic test obstacle was found. "
            "Try another seed or specify --obstacle X Y Z."
        )
        return

    print("\n" + "=" * 65)
    print("MOTION PLANNING DEMONSTRATION")
    print("=" * 65)

    print(f"Start (rad): {np.round(start, 3)}")
    print(f"Goal  (rad): {np.round(goal, 3)}")
    print(f"Obstacle centre (m): {np.round(obstacle.center, 3)}")
    print(f"Obstacle radius (m): {obstacle.radius}")
    print(f"Assumed link clearance (m): {LINK_CLEARANCE}")

    direct_safe = edge_is_safe(model, start, goal, obstacle)

    print(f"\nDirect path safe: {direct_safe}")
    print("Running RRT planner...")

    try:
        path, nodes, elapsed = plan_path(
            model,
            start,
            goal,
            obstacle=obstacle,
            max_iterations=args.iterations,
        )
    except ValueError as exc:
        print(f"\nPlanning error: {exc}")
        return

    verified = (
        validate_path(model, path, obstacle)
        if path is not None
        else False
    )

    path_length = calculate_path_length(path)

    print("\n" + "=" * 65)
    print("PLANNING RESULTS")
    print("=" * 65)

    print(f"Path found: {path is not None}")
    print(f"RRT nodes: {nodes}")
    print(f"Planning time: {elapsed:.4f} seconds")
    print(f"Waypoints: {len(path) if path is not None else 0}")
    print(f"Joint-space path length: {path_length:.4f} rad")
    print(f"Path verification passed: {verified}")

    if path is None:
        print(
            "\nNo path was found within the iteration budget. "
            "This does not prove that no path exists."
        )

    elif not verified:
        print(
            "\nWARNING: Validation failed. "
            "The path will not be labelled as verified."
        )

    else:
        print(
            "\nPath passed the configured sampled checks. "
            "These checks do not establish real-world robot safety."
        )

    metrics = {
        "nodes": nodes,
        "elapsed": elapsed,
        "length": path_length,
        "direct_safe": direct_safe,
        "verified": verified,
    }

    visualize(
        model,
        start,
        goal,
        path if verified else None,
        obstacle,
        metrics,
        urdf_path,
    )


if __name__ == "__main__":
    main()
