"""
Canadarm2-Inspired 3D Robotic Arm Motion Planner

Features:
- Loads a URDF robot model
- Reports movable joints and their limits
- Calculates forward kinematics
- Plans paths using RRT in joint space
- Checks a spherical keep-out zone
- Validates waypoints and interpolated path segments
- Visualizes the robot and planned trajectory
- Reports planning metrics

Run:
    python Pathplanner.py

Optional obstacle:
    python Pathplanner.py --obstacle 2.0 0.0 1.0 --radius 0.4

Educational prototype only. Collision checking is approximate and
must not be used to control a real robot.
"""

import argparse
import fractions
import math
import random
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Compatibility with older dependencies.
if not hasattr(fractions, "gcd"):
    fractions.gcd = math.gcd

# Compatibility aliases for older libraries.
# Use np.__dict__ to avoid triggering NumPy FutureWarnings.
for _alias, _replacement in (
    ("int", int),
    ("float", float),
    ("bool", bool),
    ("object", object),
):
    if _alias not in np.__dict__:
        setattr(np, _alias, _replacement)

from urdfpy import URDF


# ============================================================
# AUTOMATIC REFERENCE MODEL DOWNLOAD
# ============================================================

MODEL_REPOSITORY_ZIP = (
    "https://github.com/vyas-shubham/TraceableRobotModels/"
    "archive/refs/heads/master.zip"
)


def find_canadarm2_urdf(search_root):
    """Find a URDF file in the downloaded reference repository."""
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
    """Download the public reference model if it is not already present."""
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "robot_models"
    extract_dir = models_dir / "TraceableRobotModels-master"

    existing = find_canadarm2_urdf(extract_dir)

    if existing is not None:
        print(f"Using previously downloaded URDF: {existing}")
        return existing

    models_dir.mkdir(parents=True, exist_ok=True)
    archive_path = models_dir / "TraceableRobotModels-master.zip"

    print("Downloading the public robot reference-model repository...")

    try:
        request = urllib.request.Request(
            MODEL_REPOSITORY_ZIP,
            headers={
                "User-Agent": "Canadarm2-Educational-Motion-Planner"
            },
        )

        with urllib.request.urlopen(request, timeout=45) as response:
            archive_path.write_bytes(response.read())

        with zipfile.ZipFile(archive_path, "r") as archive:
            destination = models_dir.resolve()

            # Prevent archive entries from writing outside models_dir.
            for member in archive.infolist():
                target = (models_dir / member.filename).resolve()

                if (
                    target != destination
                    and destination not in target.parents
                ):
                    raise RuntimeError(
                        "The downloaded ZIP contains an unsafe file path."
                    )

            archive.extractall(models_dir)

    except Exception as exc:
        raise RuntimeError(
            "Could not download the reference robot model. "
            "Check your internet connection. You can also download "
            "the repository ZIP from "
            "https://github.com/vyas-shubham/TraceableRobotModels "
            f"and extract it inside {models_dir}. Error: {exc}"
        ) from exc

    urdf_path = find_canadarm2_urdf(extract_dir)

    if urdf_path is None:
        urdf_path = find_canadarm2_urdf(models_dir)

    if urdf_path is None:
        raise FileNotFoundError(
            f"No URDF file was found under {models_dir}."
        )

    print(f"Downloaded reference URDF: {urdf_path}")
    print(
        "Note: this is a community reference model, "
        "not an official operational robot model."
    )

    return urdf_path


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

MAX_ITERATIONS = 5000
GOAL_BIAS = 0.15
STEP_SIZE = 0.25
GOAL_TOLERANCE = 0.30
EDGE_RESOLUTION = 0.05

# Assumed link clearance in metres, not a real robot specification.
LINK_CLEARANCE = 0.15

DEFAULT_OBSTACLE = None
DEFAULT_OBSTACLE_RADIUS = 0.30


# ============================================================
# DISTANCE AND INTERPOLATION
# ============================================================

def point_to_segment_distance(point, start, end):
    """Calculate the shortest distance from a point to a line segment."""
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
    """Interpolate between two joint configurations."""
    return q1 + fraction * (q2 - q1)


def joint_distance(q1, q2):
    """Calculate Euclidean distance in joint space."""
    return float(np.linalg.norm(q2 - q1))


# ============================================================
# ROBOT MODEL
# ============================================================

class RobotModel:
    """Load a robot URDF and provide kinematics and collision helpers."""

    def __init__(self, urdf_path):
        urdf_path = Path(urdf_path)

        if not urdf_path.is_file():
            raise FileNotFoundError(
                f"URDF file was not found: {urdf_path}"
            )

        self.robot = URDF.load(str(urdf_path))

        # Fixed joints are not independent planning variables.
        self.joints = [
            joint for joint in self.robot.joints
            if joint.joint_type != "fixed"
        ]

        if not self.joints:
            raise ValueError("No movable joints were found in the URDF.")

        self.joint_names = [joint.name for joint in self.joints]
        self.lower = []
        self.upper = []

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
                # Assumed fallback limits; not verified robot limits.
                low, high = -math.pi, math.pi
                print(
                    f"WARNING: Joint '{joint.name}' has no complete "
                    "position limits. Assuming +/-180 degrees."
                )

            if low >= high:
                raise ValueError(
                    f"Invalid limits for joint '{joint.name}'."
                )

            self.lower.append(low)
            self.upper.append(high)

        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)

        # Parent-child connections from the URDF.
        self.connections = []

        for joint in self.robot.joints:
            parent = self.robot.link_map[joint.parent]
            child = self.robot.link_map[joint.child]

            self.connections.append((parent, child))

    def config_dict(self, q):
        """Convert a joint vector into a URDF configuration dictionary."""
        return {
            name: float(value)
            for name, value in zip(self.joint_names, q)
        }

    def valid_configuration(self, q):
        """Check joint vector size, numeric values, and joint limits."""
        q = np.asarray(q, dtype=float)

        if q.shape != self.lower.shape:
            return False

        if not np.all(np.isfinite(q)):
            return False

        return bool(
            np.all(q >= self.lower - 1e-9)
            and np.all(q <= self.upper + 1e-9)
        )

    def forward_kinematics(self, q):
        """Calculate each link's transform relative to the base."""
        if not self.valid_configuration(q):
            raise ValueError(
                "Joint configuration is invalid or outside its limits."
            )

        return self.robot.link_fk(cfg=self.config_dict(q))

    def link_segments(self, q):
        """
        Approximate each joint connection by a segment between
        parent and child link-frame origins.

        These segments do not represent the exact physical shapes
        of the robot's links.
        """
        transforms = self.forward_kinematics(q)
        segments = []

        for parent, child in self.connections:
            p1 = transforms[parent][:3, 3]
            p2 = transforms[child][:3, 3]

            segments.append((p1.copy(), p2.copy()))

        return segments

    def terminal_positions(self, q):
        """Return positions of the model's terminal links."""
        transforms = self.forward_kinematics(q)

        return {
            link.name: transforms[link][:3, 3].copy()
            for link in self.robot.end_links
        }

    def report(self):
        """Print robot details and joint limits."""
        print("\n" + "=" * 65)
        print("ROBOT MODEL REPORT")
        print("=" * 65)

        print(f"Model name: {self.robot.name}")
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
                f"{math.degrees(low):.2f} to "
                f"{math.degrees(high):.2f} degrees "
                f"[{joint.joint_type}]"
            )

        print("\nTerminal links:")

        for link in self.robot.end_links:
            print(f"  {link.name}")


# ============================================================
# COLLISION CHECKING
# ============================================================

class SphericalObstacle:
    """A spherical keep-out zone in the robot base coordinate frame."""

    def __init__(self, center, radius):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)

        if self.center.shape != (3,):
            raise ValueError(
                "Obstacle center must contain x, y, and z."
            )

        if not np.all(np.isfinite(self.center)):
            raise ValueError("Obstacle coordinates must be finite.")

        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("Obstacle radius must be positive.")

    def collides(self, model, q):
        """Check whether any approximated link segment intersects it."""
        for start, end in model.link_segments(q):
            distance = point_to_segment_distance(
                self.center, start, end
            )

            if distance <= self.radius + LINK_CLEARANCE:
                return True

        return False


def configuration_is_safe(model, q, obstacle=None):
    """Check joint limits and optional obstacle clearance."""
    if not model.valid_configuration(q):
        return False

    try:
        if obstacle is not None and obstacle.collides(model, q):
            return False
    except Exception:
        return False

    return True


def edge_is_safe(model, q1, q2, obstacle=None):
    """Sample intermediate joint configurations along a path segment."""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    if not model.valid_configuration(q1):
        return False

    if not model.valid_configuration(q2):
        return False

    maximum_change = float(np.max(np.abs(q2 - q1)))

    samples = max(
        2,
        int(math.ceil(maximum_change / EDGE_RESOLUTION)) + 1,
    )

    for fraction in np.linspace(0.0, 1.0, samples):
        q = interpolate(q1, q2, fraction)

        if not configuration_is_safe(model, q, obstacle):
            return False

    return True


# ============================================================
# RRT MOTION PLANNER
# ============================================================

class TreeNode:
    """A node in the rapidly-exploring random tree."""

    def __init__(self, q, parent=None):
        self.q = np.asarray(q, dtype=float)
        self.parent = parent


def nearest_node(nodes, target):
    """Find the tree node closest to a sampled target."""
    distances = [
        joint_distance(node.q, target)
        for node in nodes
    ]

    return nodes[int(np.argmin(distances))]


def steer(q_start, q_target, step_size):
    """Move toward a target by at most step_size."""
    difference = q_target - q_start
    distance = float(np.linalg.norm(difference))

    if distance <= step_size:
        return q_target.copy()

    return q_start + difference * (step_size / distance)


def reconstruct_path(node):
    """Reconstruct the route using parent pointers."""
    path = []

    while node is not None:
        path.append(node.q.copy())
        node = node.parent

    path.reverse()

    return path


def plan_path(
    model,
    start,
    goal,
    obstacle=None,
    max_iterations=MAX_ITERATIONS,
):
    """Plan a joint-space route using RRT.

    Returns:
        path, number of tree nodes, planning time
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    if not model.valid_configuration(start):
        raise ValueError("Start violates joint limits.")

    if not model.valid_configuration(goal):
        raise ValueError("Goal violates joint limits.")

    if not configuration_is_safe(model, start, obstacle):
        raise ValueError("Start configuration is not collision-free.")

    if not configuration_is_safe(model, goal, obstacle):
        raise ValueError("Goal configuration is not collision-free.")

    started = time.perf_counter()

    if edge_is_safe(model, start, goal, obstacle):
        return [start.copy(), goal.copy()], 1, (
            time.perf_counter() - started
        )

    nodes = [TreeNode(start)]

    for _ in range(max_iterations):
        if random.random() < GOAL_BIAS:
            sample = goal.copy()
        else:
            sample = np.random.uniform(model.lower, model.upper)

        nearest = nearest_node(nodes, sample)
        candidate = steer(nearest.q, sample, STEP_SIZE)

        if not model.valid_configuration(candidate):
            continue

        if not edge_is_safe(model, nearest.q, candidate, obstacle):
            continue

        new_node = TreeNode(candidate, nearest)
        nodes.append(new_node)

        if joint_distance(candidate, goal) <= GOAL_TOLERANCE:
            if edge_is_safe(model, candidate, goal, obstacle):
                goal_node = TreeNode(goal.copy(), new_node)
                path = reconstruct_path(goal_node)
                elapsed = time.perf_counter() - started

                return path, len(nodes), elapsed

    elapsed = time.perf_counter() - started

    return None, len(nodes), elapsed


# ============================================================
# PATH METRICS AND VALIDATION
# ============================================================

def calculate_path_length(path):
    """Calculate total joint-space distance in radians."""
    if path is None or len(path) < 2:
        return 0.0

    return sum(
        joint_distance(a, b)
        for a, b in zip(path[:-1], path[1:])
    )


def validate_path(model, path, obstacle=None):
    """Verify all waypoints and interpolated path segments."""
    if path is None or len(path) < 2:
        return False

    for q in path:
        if not configuration_is_safe(model, q, obstacle):
            return False

    for q1, q2 in zip(path[:-1], path[1:]):
        if not edge_is_safe(model, q1, q2, obstacle):
            return False

    return True


def direct_path_is_safe(model, start, goal, obstacle=None):
    """Check whether direct joint-space interpolation is safe."""
    return edge_is_safe(model, start, goal, obstacle)


# ============================================================
# VISUALIZATION
# ============================================================

def draw_sphere(ax, center, radius):
    """Draw a spherical obstacle in 3D."""
    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 18)

    x = center[0] + radius * np.outer(
        np.cos(u), np.sin(v)
    )
    y = center[1] + radius * np.outer(
        np.sin(u), np.sin(v)
    )
    z = center[2] + radius * np.outer(
        np.ones_like(u), np.cos(v)
    )

    ax.plot_surface(x, y, z, alpha=0.35)



def visualize(model, path, obstacle=None):
    """Visualize the reference robot, sampled path, and obstacle."""
    if path is None or len(path) == 0:
        print("No path available to visualize.")
        return

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Draw the obstacle, if one was supplied.
    if obstacle is not None:
        draw_sphere(ax, obstacle.center, obstacle.radius)
        ax.scatter(
            [obstacle.center[0]],
            [obstacle.center[1]],
            [obstacle.center[2]],
            marker="x",
            s=100,
            label="Obstacle centre",
        )

    # Draw the robot at up to five points along the planned path.
    selected = sorted(set(
        np.linspace(
            0,
            len(path) - 1,
            min(5, len(path)),
            dtype=int,
        ).tolist()
    ))

    for step, index in enumerate(selected):
        for p1, p2 in model.link_segments(path[index]):
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                marker="o",
                markersize=4,
                linewidth=2.5,
                alpha=0.8,
                label="Robot links" if step == 0 else None,
            )

    # Plot the terminal-link trajectory.
    terminal_links = model.terminal_positions(path[0])

    for link_name in terminal_links:
        positions = np.array([
            model.terminal_positions(q)[link_name]
            for q in path
        ])

        ax.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            linestyle="--",
            marker=".",
            markersize=5,
            linewidth=2,
            label=f"{link_name} trajectory",
        )

        # Mark the start and goal positions.
        ax.scatter(
            positions[0, 0],
            positions[0, 1],
            positions[0, 2],
            marker="o",
            s=75,
            label=f"{link_name} start",
        )
        ax.scatter(
            positions[-1, 0],
            positions[-1, 1],
            positions[-1, 2],
            marker="*",
            s=140,
            label=f"{link_name} goal",
        )

    ax.set_title(
        "Canadarm2 Reference Model: Motion Planning",
        fontsize=14,
        pad=18,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="best", fontsize=8)

    # Use equal scale on all three axes where supported.
    try:
        ax.set_aspect("equal", adjustable="box")
    except (AttributeError, NotImplementedError, ValueError):
        pass

    plt.tight_layout()
    plt.show()



# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plan motions for a URDF robot model."
    )

    parser.add_argument(
        "urdf_path",
        nargs="?",
        default=None,
        help=(
            "Optional path to a local robot URDF. "
            "Downloads the reference model if omitted."
        ),
    )

    parser.add_argument(
        "--obstacle",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_OBSTACLE,
        help="Optional spherical obstacle centre in metres",
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_OBSTACLE_RADIUS,
        help="Obstacle radius in metres",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=MAX_ITERATIONS,
        help="Maximum RRT iterations",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for repeatable planning attempts",
    )

    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        urdf_path = args.urdf_path or obtain_default_urdf()
        model = RobotModel(urdf_path)
        model.report()

    except Exception as exc:
        print(f"\nERROR loading robot model: {exc}")
        return

    obstacle = None

    if args.obstacle is not None:
        try:
            obstacle = SphericalObstacle(
                args.obstacle,
                args.radius,
            )
        except ValueError as exc:
            parser.error(str(exc))

    # Demonstration start configuration: midpoint of each joint range.
    start = (model.lower + model.upper) / 2.0

    # Demonstration goal: move the first joint by 15 degrees if possible.
    goal = start.copy()
    changed = False

    for index in range(len(goal)):
        plus = goal[index] + math.radians(15.0)
        minus = goal[index] - math.radians(15.0)

        if plus <= model.upper[index]:
            goal[index] = plus
            changed = True
            break

        if minus >= model.lower[index]:
            goal[index] = minus
            changed = True
            break

    if not changed:
        print("Could not create a demonstration goal.")
        return

    print("\n" + "=" * 65)
    print("MOTION PLANNING")
    print("=" * 65)

    print(f"Start configuration (rad): {np.round(start, 3)}")
    print(f"Goal configuration  (rad): {np.round(goal, 3)}")

    if obstacle is None:
        print("Obstacle: disabled (no obstacle specified)")
    else:
        print(f"Obstacle centre (m): {obstacle.center}")
        print(f"Obstacle radius (m): {obstacle.radius}")
        print(f"Assumed link clearance (m): {LINK_CLEARANCE}")

    try:
        baseline_safe = direct_path_is_safe(
            model,
            start,
            goal,
            obstacle,
        )

        print(f"\nDirect path safe: {baseline_safe}")
        print("Running RRT planner...")

        path, nodes, elapsed = plan_path(
            model,
            start,
            goal,
            obstacle=obstacle,
            max_iterations=args.iterations,
        )

    except ValueError as exc:
        print(f"\nPlanning error: {exc}")
        print(
            "Try a different obstacle location or inspect the model "
            "and its joint limits."
        )
        return

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)

    print(f"Path found: {path is not None}")
    print(f"Tree nodes: {nodes}")
    print(f"Planning time: {elapsed:.6f} seconds")

    if path is None:
        print("No path found within the iteration budget.")
        print(
            "This does not prove no path exists. Try more iterations "
            "or a different obstacle position."
        )
        return

    verified = validate_path(model, path, obstacle)

    print(f"Waypoints: {len(path)}")
    print(
        f"Joint-space path length: "
        f"{calculate_path_length(path):.4f} rad"
    )
    print(f"Path verification passed: {verified}")

    if baseline_safe:
        print("Baseline: direct movement is safe under this model.")
    else:
        print("Baseline: direct movement is blocked by the obstacle.")

    if verified:
        print(
            "\nThe route passed sampled checks using simplified "
            "link-frame segments."
        )
        visualize(model, path, obstacle)
    else:
        print(
            "\nWARNING: verification failed. "
            "The route will not be presented as safe."
        )


if __name__ == "__main__":
    main()