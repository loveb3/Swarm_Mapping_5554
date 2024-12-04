import numpy as np
import threading
from shared_queue import move_queue, bot_queue, face_queue
import time
from collections import deque

from object_detection import object_detection
from Simulation_Environment import model

from depth_estimation2 import depthEst

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
        # self.front = 'North'  # initial direction the robot is facing
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

class Robot:
    def __init__(self, id, initial_position, initial_direction='North'):
        self.id = id
        self.position = initial_position  # Current coordinates
        self.front = initial_direction  # Current front direction
        self.frontier = deque()  # Stack for DFS
        self.backtrack = deque()  # Stack for backtracking
        self.prev_coords = deque()  # History of visited coordinates
        self.prev_nodes = deque()  # History of visited nodes



def multiRobotDFS(initial_positions, initial_data):
    """
        Perform depth-first search with a swarm of robots sharing a common node tree.

        Args:
        - initial_positions: List of starting coordinates for each robot.
        - initial_data: List of (image, movements) tuples for the initial state of each robot.

        Returns:
        - The shared node tree mapping the area.
        """


    # Initialize the shared node tree and individual robot stacks
    env_tree = {}

    robots = []
    for bot_id, (position, (image, movements)) in enumerate(zip(initial_positions, initial_data)):
        robot = Robot(bot_id, position)
        start_node = Node(position, image, movements)
        env_tree[position] = start_node
        robot.frontier.append(start_node)
        robots.append(robot)

    def is_valid_step(c_coords, p_coords):
        """Check if the current node is within one movement step of the previous node."""
        x1, y1 = c_coords
        x2, y2 = p_coords
        return abs(x1 - x2) + abs(y1 - y2) <= step  # Manhattan distance should equal step size

    # Main exploration loop
    while any(robot.frontier for robot in robots):  # Continue while any robot has nodes to explore
        for robot in robots:
            # print(f"Bot {bot} has frontier {frontier}")

            if not robot.frontier:  # Skip robots with an empty frontier
                continue


            curr_node = robot.frontier.pop()  # Get the current node to explore

            move = None


            bot_queue.put(robot.id)
            move_queue.put(curr_node.coordinates)
            face_queue.put(robot.front)
            # Notify animation and wait for it to complete
            dfs_ready.clear()
            animation_ready.set()  # Notify the animation thread
            dfs_ready.wait()  # Wait for the animation to signal back
            time.sleep(1)

            if curr_node.visited:
                # Backtracking logic
                while robot.backtrack:
                    print(f'{robot.id} is backtracking at {curr_node.coordinates}\n\n\n\n\n\n\n\n\n')
                    move = None
                    for node in robot.frontier:
                        print(f'{robot.id} frontier has {node.coordinates}')


                    # Check for unvisited neighbors
                    for direction in ['right', 'left', 'forward']:
                        new_coords, new_front = calcCoords(curr_node.coordinates, direction, robot.front)
                        if new_coords in env_tree:
                            neighbor = env_tree[new_coords]
                            if not neighbor.visited and direction in curr_node.movements:
                                # Found an unvisited neighbor
                                robot.frontier.append(neighbor)
                                move = direction
                                robot.backtrack.append(opposite_direction(direction))  # Add backtracking path
                                break

                    move = robot.backtrack.pop()
                    break
                # If the backtrack is empty, robot has finished
                if not robot.backtrack and not move:
                    print(f"Robot {robot.id} has finished its exploration.")
                    continue
            else:
                # Populate the current node's data
                curr_coords = curr_node.coordinates
                imageL, imageF, imageR, possible_movements = populate_node(robot.id, curr_coords, robot.front)
                curr_node.set(imageL, imageF, imageR, possible_movements, True)
                print(f"Robot {robot.id} is at position {curr_coords} facing {robot.front}.")

                # Explore possible movements
                for direction in ['right', 'left', 'forward']:
                    new_coords, new_front = calcCoords(curr_coords, direction, robot.front)

                    print(f"Bot {robot.id} has possible movements {possible_movements} and direction {direction}")

                    if new_coords in env_tree:
                        new_node = env_tree[new_coords]
                        new_node.front = new_front
                    elif new_coords not in env_tree:  # Add new node to the tree
                        new_node = Node(new_coords)
                        new_node.front = new_front
                        env_tree[new_coords] = new_node

                    if direction in possible_movements:
                        print(f'{direction} is possible')
                        print(f"The new coordinates would be {new_coords} facing {new_front}")
                        # Add to frontier if the movement is valid
                        if not new_node.visited:
                            robot.frontier.append(new_node)
                            move = direction
                            front = new_front

                            # Add the opposite direction to the backtracking stack
                            robot.backtrack.append(opposite_direction(direction))
                            robot.prev_coords.append(new_coords)
                            robot.prev_nodes.append(curr_node)

                    else:
                        # Mark the node as an obstacle
                        new_node.is_obstacle = True
                        new_node.obstacle_type = object_detection(direction,curr_coords[0],curr_coords[1],model)

            # Send the movement command to the robot
            if move:
                robot.front = front
                print(f"Robot {robot.id} will make movement {move}.")
                print(f"and is facing {robot.front}")


            else:
                print(f"Robot {robot.id} has no valid moves left and will wait.")


            remove_from_all_frontiers(env_tree, robots)

    # Notify animation that DFS is complete
    animation_ready.set()
    print("Exploration complete. All accessible nodes have been visited.")
    return env_tree


def populate_node(robot_id, coordinates, face):
    """Simulate receiving data from the robot."""
    print(f"Bot {robot_id}")
    # coordinates = (robot_id, 0)  # get position of the robot
    imageL, imageF, imageR, depthL, depthF, depthR = captureImage()
    possible_movements = depthCalc(depthL, depthF, depthR, step, coordinates, face)
    return imageL, imageF, imageR, possible_movements


def remove_from_all_frontiers(env, robots):
    """
    Removes a node from the frontier of all robots.

    Args:
        robots (list of RobotState): The list of robot states.
        node_to_remove (Node): The node to be removed from all frontiers.
    """
    for coord in env:
        node = env[coord]
        if node.visited:
            for robot in robots:
                # Remove the node from the robot's frontier if it exists
                # robot.frontier = [node for node in robot.frontier if node.coords != node_to_remove.coords]
                if node in robot.frontier:
                    robot.frontier.remove(node)


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

    # Larger Test Grid
    # rows = 10
    # cols = 10
    # obstacles = [(3, 3), (5, 5), (7, 6)]

    # Smaller Test Grid
    rows = 3
    cols = 3
    obstacles = [(0, 0), (0, 2), (2, 0)]

    ############# Hari's code ##################
    valid_movements = depthEst(rows, current_position, face)

    ################# Known Obstacle Code ###############
    # valid_movements = []
    #
    # if face == 'North':
    #     if y - step >= 0 and (x, y - step) not in obstacles:  # Check forward movement
    #         valid_movements.append("forward")
    #     if x - step >= 0 and (x - step, y) not in obstacles:  # Check left movement
    #         valid_movements.append("left")
    #     if x + step < cols and (x + step, y) not in obstacles:  # Check right movement
    #         valid_movements.append("right")
    #
    # elif face == 'South':
    #     if y + step < rows and (x, y + step) not in obstacles:  # Check forward movement
    #         valid_movements.append("forward")
    #     if x + step < cols and (x + step, y) not in obstacles:  # Check left movement
    #         valid_movements.append("left")
    #     if x - step >= 0 and (x - step, y) not in obstacles:  # Check right movement
    #         valid_movements.append("right")
    #
    # elif face == 'East':
    #     if x + step < cols and (x + step, y) not in obstacles:  # Check forward movement
    #         valid_movements.append("forward")
    #     if y - step >= 0 and (x, y - step) not in obstacles:  # Check left movement
    #         valid_movements.append("left")
    #     if y + step < rows and (x, y + step) not in obstacles:  # Check right movement
    #         valid_movements.append("right")
    #
    # elif face == 'West':
    #     if x - step >= 0 and (x - step, y) not in obstacles:  # Check forward movement
    #         valid_movements.append("forward")
    #     if y + step < rows and (x, y + step) not in obstacles:  # Check left movement
    #         valid_movements.append("left")
    #     if y - step >= 0 and (x, y - step) not in obstacles:  # Check right movement
    #         valid_movements.append("right")

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

    print(f'The current pos is {curr_coords} facing {face} and moving {direction}')

    if direction == 'forward':
        if face == 'North':
            return (x, y - step), 'North'
        elif face == 'South':
            return (x, y + step), 'South'
        elif face == 'East':
            return (x + step, y), 'East'
        elif face == 'West':
            return (x - step, y), 'West'

    elif direction == 'backward':
        if face == 'North':
            return (x, y + step), 'South'
        elif face == 'South':
            return (x, y - step), 'North'
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
