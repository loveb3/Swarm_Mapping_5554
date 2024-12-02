import numpy as np
import threading
from shared_queue import move_queue, bot_queue

# Shared signaling objects
dfs_ready = threading.Event()
animation_ready = threading.Event()

class Node:
    def __init__(self, coordinates, imageL=None, imageF=None, imageR=None, movements=None, visited=False):
        self.coordinates = coordinates  # (x, y) position
        self.image_Left = imageL  # Captured image data to the left
        self.image_Front = imageF  # Captured image data to the front
        self.image_Right = imageR  # Captured image data to the right
        self.movements = movements  # Possible movements (e.g., ['forward', 'left'])
        self.front = 'North'  # initial direction the robot is facing
        self.visited = visited  # Boolean visited flag
        self.parent = None  # Pointer to parent node in the tree
        self.is_obstacle = False  # Flag to mark obstacles
        self.obstacle_type = None  # The detected obstacle type

    def set(self, imageL, imageF, imageR, movements, visited):
        self.image_Left = imageL
        self.image_Front = imageF
        self.image_Right = imageR
        self.movements = movements
        self.visited = visited

    def update_direction(self, new_direction):
        self.front = new_direction


# static values
step = 1        # The distance between coordinate points


def multiRobotDFS(initial_positions, initial_data):
    """
        Perform depth-first search with a swarm of robots sharing a common node tree.

        Args:
        - initial_positions: List of starting coordinates for each robot.
        - initial_data: List of (image, movements) tuples for the initial state of each robot.

        Returns:
        - The shared node tree mapping the area.
        """
    from collections import deque

    # Initialize the shared node tree and individual robot stacks
    env_tree = {}
    robot_frontiers = []
    robot_backtrack = []

    # Create initial nodes for each robot
    for position, (image, movements) in zip(initial_positions, initial_data):
        start_node = Node(position, image, movements)
        env_tree[position] = start_node
        robot_frontiers.append(deque([start_node]))  # Each robot gets its own stack
        robot_backtrack.append(deque())  # Each robot gets its own backtracking queue

    # Main exploration loop
    while any(frontier for frontier in robot_frontiers):  # Continue while any robot has nodes to explore
        for bot, (frontier, backtrack) in enumerate(zip(robot_frontiers, robot_backtrack)):
            # print(f"Bot {bot} has frontier {frontier}")

            if not frontier:  # Skip robots with an empty frontier
                continue

            curr_node = frontier.pop()  # Get the current node to explore
            move = None

            if curr_node.visited:
                # Backtracking logic
                while backtrack:
                    print(f'{bot} is backtracking')

                    # Check for unvisited neighbors
                    for direction in ['right', 'left', 'forward']:
                        new_coords, new_front = calcCoords(direction, curr_node.coordinates, curr_node.front)
                        if new_coords in env_tree:
                            neighbor = env_tree[new_coords]
                            if not neighbor.visited and direction in curr_node.movements:
                                # Found an unvisited neighbor
                                frontier.append(neighbor)
                                move = direction
                                backtrack.append(opposite_direction(direction))  # Add backtracking path
                                break

                    if move:
                        break
                    else:
                        # Continue backtracking to parent node
                        move = backtrack.pop()

                # If the backtrack is empty, robot has finished
                if not backtrack and not move:
                    print(f"Robot {bot} has finished its exploration.")
                    continue
            else:
                # Populate the current node's data
                curr_coords = curr_node.coordinates
                imageL, imageF, imageR, possible_movements = populate_node(bot, curr_coords, curr_node.front)
                curr_node.set(imageL, imageF, imageR, possible_movements, True)

                # Explore possible movements
                for direction in ['right', 'left', 'forward']:
                    new_coords, new_front = calcCoords(curr_coords, direction, curr_node.front)

                    print(f"Bot {bot} has possible movements {possible_movements} and direction {direction}")

                    if new_coords not in env_tree:  # Add new node to the tree
                        new_node = Node(new_coords)
                        new_node.front = new_front
                        env_tree[new_coords] = new_node

                        if direction in possible_movements:
                            # Add to frontier if the movement is valid
                            frontier.append(new_node)
                            move = direction

                            # Add the opposite direction to the backtracking stack
                            backtrack.append(opposite_direction(direction))
                        else:
                            # Mark the node as an obstacle
                            new_node.is_obstacle = True
                            new_node.obstacle_type = object_detection()

            # Send the movement command to the robot
            if move:
                print(f"Robot {bot} will make movement {move}.")
                bot_queue.put(bot)
                move_queue.put(move)
                # Notify animation and wait for it to complete
                dfs_ready.clear()
                animation_ready.set()  # Notify the animation thread
                dfs_ready.wait()  # Wait for the animation to signal back

            else:
                print(f"Robot {bot} has no valid moves left and will wait.")

    # Notify animation that DFS is complete
    animation_ready.set()
    print("Exploration complete. All accessible nodes have been visited.")
    return env_tree


def populate_node(robot_id,coordinates, face):
    """Simulate receiving data from the robot."""
    print(f"Bot {robot_id}")
    # coordinates = (robot_id, 0)  # get position of the robot
    imageL, imageF, imageR, depthL, depthF, depthR = captureImage()
    possible_movements = depthCalc(depthL, depthF, depthR, step, coordinates, face)
    return imageL, imageF, imageR, possible_movements


def captureImage():
    """Capture the images to the left, right, and front of the robot."""
    # Need sim controls from Hari

    # rotate 90 degrees CCW
    # imageL = print('Left image captured')
    # depthL = print('Left depth image captured')
    #
    # # rotate 90 degrees CW
    # imageF = print('Front image captured')
    # depthF = print('Front depth image captured')
    #
    # # rotate 90 degrees CW
    # imageR = print('Right image captured')
    # depthR = print('Right depth image captured')

    imageL = None
    imageF = None
    imageR = None
    depthL = None
    depthF = None
    depthR = None

    return imageL, imageF, imageR, depthL, depthF, depthR


# def depthCalc(depthL, depthF, depthR, threshold):
#     """
#     Determine valid movement directions based on depth and field bounds.
#
#     Parameters:
#         depthL, depthF, depthR (int): Depth values for directions.
#         threshold (int): Minimum depth value to consider a path valid.
#
#     Returns:
#         list: Valid directions.
#     """
#     valid_movements = []
#     if depthL > threshold:
#         valid_movements.append("left")
#     if depthF > threshold:
#         valid_movements.append("forward")
#     if depthR > threshold:
#         valid_movements.append("right")
#     return valid_movements

# test code for depthCalc
def depthCalc(depthL, depthF, depthR, step, current_position, face):
    """
    Determine valid movement directions based on arena boundaries, obstacle positions, and robot's facing direction.

    Parameters:
        depthL, depthF, depthR (int): Placeholder values (not used in fake implementation).
        step (int): Step size for movements.
        current_position (tuple): Current position of the robot (x, y).
        face (str): Current facing direction of the robot ('North', 'South', 'East', 'West').
        arena_size (tuple): Size of the arena (rows, cols).
        obstacles (list): List of obstacle positions [(x1, y1), (x2, y2), ...].

    Returns:
        list: Valid directions (['left', 'forward', 'right']).
    """
    x, y = current_position
    rows = 10
    cols = 10
    obstacles = [(3, 3), (5, 5), (6, 7)]
    valid_movements = []

    if face == 'North':
        if y + step < cols and (x, y + step) not in obstacles:  # Check forward movement
            valid_movements.append("forward")
        if x - step >= 0 and (x - step, y) not in obstacles:  # Check left movement
            valid_movements.append("left")
        if x + step < rows and (x + step, y) not in obstacles:  # Check right movement
            valid_movements.append("right")

    elif face == 'South':
        if y - step >= 0 and (x, y - step) not in obstacles:  # Check forward movement
            valid_movements.append("forward")
        if x + step < rows and (x + step, y) not in obstacles:  # Check left movement
            valid_movements.append("left")
        if x - step >= 0 and (x - step, y) not in obstacles:  # Check right movement
            valid_movements.append("right")

    elif face == 'East':
        if x + step < rows and (x + step, y) not in obstacles:  # Check forward movement
            valid_movements.append("forward")
        if y - step >= 0 and (x, y - step) not in obstacles:  # Check left movement
            valid_movements.append("left")
        if y + step < cols and (x, y + step) not in obstacles:  # Check right movement
            valid_movements.append("right")

    elif face == 'West':
        if x - step >= 0 and (x - step, y) not in obstacles:  # Check forward movement
            valid_movements.append("forward")
        if y + step < cols and (x, y + step) not in obstacles:  # Check left movement
            valid_movements.append("left")
        if y - step >= 0 and (x, y - step) not in obstacles:  # Check right movement
            valid_movements.append("right")

    print(f"has valid movements {valid_movements}")

    return valid_movements


def calcCoords(curr_coords, direction, face):
    """
    Calculate the new coordinates and facing direction based on the movement.

    Args:
    - curr_coords: Current coordinates (x, y) of the robot.
    - direction: The movement direction ('forward', 'backward', 'left', 'right').
    - face: The current facing direction ('North', 'South', 'East', 'West').

    Returns:
    - tuple: New coordinates (x, y) and updated facing direction.
    """
    x, y = curr_coords
    new_face = face

    if direction == 'forward':
        if face == 'North':
            return (x, y + step), 'North'
        elif face == 'South':
            return (x, y - step), 'South'
        elif face == 'East':
            return (x + step, y), 'East'
        elif face == 'West':
            return (x - step, y), 'West'

    elif direction == 'backward':
        if face == 'North':
            return (x, y - step), 'South'
        elif face == 'South':
            return (x, y + step), 'North'
        elif face == 'East':
            return (x - step, y), 'West'
        elif face == 'West':
            return (x + step, y), 'East'

    elif direction == 'left':
        if face == 'North':
            return (x - step, y), 'West'
        elif face == 'South':
            return (x + step, y), 'East'
        elif face == 'East':
            return (x, y - step), 'North'
        elif face == 'West':
            return (x, y + step), 'South'

    elif direction == 'right':
        if face == 'North':
            return (x + step, y), 'East'
        elif face == 'South':
            return (x - step, y), 'West'
        elif face == 'East':
            return (x, y + step), 'South'
        elif face == 'West':
            return (x, y - step), 'North'

    return curr_coords, new_face


def object_detection():
    # Dev's detection network

    # obj = print('object type')

    obj = None

    return obj

def opposite_direction(direction):
    if direction == 'forward':
        return 'backward'
    elif direction == 'backward':
        return 'forward'
    elif direction == 'left':
        return 'right'
    elif direction == 'right':
        return 'left'
    return None
