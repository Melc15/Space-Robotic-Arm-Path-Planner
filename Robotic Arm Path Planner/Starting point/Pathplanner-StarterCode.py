
import math
import heapq
import time
import matplotlib.pyplot as plt
import numpy as np


# Set robot arm dimensions (metres)
LINK_1 = 1.0
LINK_2 = 0.8
LINK_RADIUS = 0.035

# Simulated satellite keep-out zone
OBSTACLE_CENTER = np.array([0.8, 1.25])
OBSTACLE_RADIUS = 0.22

# Joint-space search resolution-set to 10 degrees (in radians)
STEP = math.radians(10)


# 1. Robot kinematics

def forward_kinematics(q):
    #Calculate the positions of the arm joints.
    shoulder, elbow = q

    base = np.array([0.0, 0.0])

    joint = np.array([
        LINK_1 * math.cos(shoulder),
        LINK_1 * math.sin(shoulder)
    ])

    end = joint + LINK_2 * np.array([
        math.cos(shoulder + elbow),
        math.sin(shoulder + elbow)
    ])

    return base, joint, end

# 2. Detect collisions between the robot arm and the obstacle

def point_to_segment_distance(point, start, end):
    #Find the shortest distance from a point to a line segment
    segment = end - start
    length_squared = np.dot(segment, segment)

    if length_squared == 0:
        return np.linalg.norm(point - start)

    t = np.dot(point - start, segment) / length_squared
    t = np.clip(t, 0.0, 1.0)

    closest = start + t * segment
    return np.linalg.norm(point - closest)


def collides(q):
    #Return True if either robot link intersects the obstacle
    base, joint, end = forward_kinematics(q)
    clearance = OBSTACLE_RADIUS + LINK_RADIUS

    first_distance = point_to_segment_distance(
        OBSTACLE_CENTER, base, joint
    )

    second_distance = point_to_segment_distance(
        OBSTACLE_CENTER, joint, end
    )

    return (
        first_distance <= clearance
        or second_distance <= clearance
    )


def transition_is_safe(q1, q2, samples=10):
    #Check intermediate configurations between two joint states
    for i in range(samples + 1):
        t = i / samples

        q = (
            q1[0] + t * (q2[0] - q1[0]),
            q1[1] + t * (q2[1] - q1[1])
        )

        if collides(q):
            return False

    return True

# 3. Implement A* path planning in joint space

def heuristic(a, b):
    #Euclidean distance between integer grid configurations
    return math.hypot(a[0] - b[0], a[1] - b[1])


def get_neighbors(q):
    #Generate neighbors using integer grid moves in joint space
    moves = [
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, 1), (1, -1),
        (-1, 1), (-1, -1)
    ]

    neighbors = []

    # Joint limits are set to -180 to +180 degrees
    max_index = round(math.pi / STEP)

    for dx, dy in moves:
        candidate = (q[0] + dx, q[1] + dy)

        if not (
            -max_index <= candidate[0] <= max_index
            and -max_index <= candidate[1] <= max_index
        ):
            continue

        angles = (
            candidate[0] * STEP,
            candidate[1] * STEP
        )

        if not collides(angles):
            neighbors.append(candidate)

    return neighbors


def plan_path(start, goal):
    #Plan a collision-free path using A* on a grid in joint space
    start_time = time.perf_counter()

    def to_grid(q):
        return (
            round(q[0] / STEP),
            round(q[1] / STEP)
        )

    def to_angles(q):
        return (
            q[0] * STEP,
            q[1] * STEP
        )

    if collides(start):
        raise ValueError("Start configuration is in collision.")

    if collides(goal):
        raise ValueError("Goal configuration is in collision.")

    grid_start = to_grid(start)
    grid_goal = to_grid(goal)

    if collides(to_angles(grid_start)):
        raise ValueError(
            "The nearest grid configuration to the start is blocked. "
            "Try a smaller STEP."
        )

    if collides(to_angles(grid_goal)):
        raise ValueError(
            "The nearest grid configuration to the goal is blocked. "
            "Try a smaller STEP."
        )

    frontier = [(heuristic(grid_start, grid_goal), grid_start)]
    came_from = {grid_start: None}
    cost_so_far = {grid_start: 0.0}
    expanded_nodes = 0

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == grid_goal:
            break

        expanded_nodes += 1

        for neighbor in get_neighbors(current):
            # Verify that the arm avoids obstacles between grid states
            if not transition_is_safe(
                to_angles(current), to_angles(neighbor)
            ):
                continue

            move_cost = heuristic(current, neighbor)
            new_cost = cost_so_far[current] + move_cost

            if (
                neighbor not in cost_so_far
                or new_cost < cost_so_far[neighbor]
            ):
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, grid_goal)

                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current

    elapsed = time.perf_counter() - start_time

    if grid_goal not in came_from:
        return {
            "success": False,
            "path": [],
            "expanded_nodes": expanded_nodes,
            "planning_time": elapsed
        }

    # Reconstruct the route from the goal back to the start
    grid_path = []
    current = grid_goal

    while current is not None:
        grid_path.append(current)
        current = came_from[current]

    grid_path.reverse()

    # Convert the integer grid route back to joint angles
    path = [to_angles(q) for q in grid_path]

    return {
        "success": True,
        "path": path,
        "expanded_nodes": expanded_nodes,
        "planning_time": elapsed
    }

# 4. Baseline direct-movement check

def direct_move_is_safe(start, goal, samples=200):
    #Test a naive direct joint-space movement for collisions with the obstacle
    for i in range(samples + 1):
        t = i / samples

        q = (
            start[0] + t * (goal[0] - start[0]),
            start[1] + t * (goal[1] - start[1])
        )

        if collides(q):
            return False

    return True


def calculate_path_length(path):
    #Calculate total joint-space path length in radians
    total = 0.0

    for i in range(1, len(path)):
        total += heuristic(path[i - 1], path[i])

    return total

# 5. Visualization of the robot arm and planned path

def visualize(path):
    #Display the robot arm and planned end-effector trajectory
    fig, ax = plt.subplots(figsize=(9, 8))

    obstacle = plt.Circle(
        OBSTACLE_CENTER,
        OBSTACLE_RADIUS,
        color="red",
        alpha=0.4,
        label="Obstacle / keep-out zone"
    )
    ax.add_patch(obstacle)

    end_positions = [
        forward_kinematics(q)[2] for q in path
    ]

    ax.plot(
        [p[0] for p in end_positions],
        [p[1] for p in end_positions],
        "--",
        label="End-effector trajectory"
    )

    # Display the start, midpoint and goal configurations
    indexes = sorted(set([
        0,
        len(path) // 2,
        len(path) - 1
    ]))

    labels = ["Start", "Midpoint", "Goal"]

    for index in indexes:
        base, joint, end = forward_kinematics(path[index])

        ax.plot(
            [base[0], joint[0], end[0]],
            [base[1], joint[1], end[1]],
            marker="o",
            linewidth=2,
            label=f"Arm: waypoint {index}"
        )

    ax.scatter(
        [end_positions[0][0]],
        [end_positions[0][1]],
        marker="s",
        s=80,
        label="Starting end-effector position"
    )

    ax.scatter(
        [end_positions[-1][0]],
        [end_positions[-1][1]],
        marker="*",
        s=150,
        label="Target position"
    )

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("Autonomous Space Robotic-Arm Path Planner")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

# 6. Run main program

def main():
    # Set starting and target joint angles
    start = (
        round(math.radians(-65), 8),
        round(math.radians(60), 8)
    )

    goal = (
        round(math.radians(60), 8),
        round(math.radians(-55), 8)
    )

    print("=" * 50)
    print("Autonomous Space Robotic-Arm Path Planner")
    print("=" * 50)

    print("\nChecking direct movement...")
    baseline_safe = direct_move_is_safe(start, goal)

    print(f"Direct movement collision-free: {baseline_safe}")

    print("\nRunning A* path planner...")
    result = plan_path(start, goal)

    print("\nPlanning results:")
    print("-" * 35)
    print(f"Path found: {result['success']}")
    print(f"Configurations explored: {result['expanded_nodes']}")
    print(f"Planning time: {result['planning_time']:.4f} seconds")

    if result["success"]:
        path = result["path"]

        print(f"Number of waypoints: {len(path)}")
        print(
            "Joint-space path length: "
            f"{calculate_path_length(path):.3f} radians"
        )

        print(
            "Baseline comparison: "
            + (
                "Direct movement is safe."
                if baseline_safe
                else "Direct movement is unsafe; an alternative was found."
            )
        )

        # Final verification of every planned transition
        verified = all(
            not collides(q)
            for q in path
        ) and all(
            transition_is_safe(path[i - 1], path[i])
            for i in range(1, len(path))
        )

        print(f"Path collision verification: {verified}")

        if verified:
            visualize(path)
        else:
            print("WARNING: Path verification failed; not displaying it as safe.")

    else:
        print("No feasible path found at the current search resolution.")


if __name__ == "__main__":
    main()
