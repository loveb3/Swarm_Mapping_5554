import numpy as np


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
            if not frontier:  # Skip robots with an empty frontier
                continue

            curr_node = frontier.pop()  # Get the current node to explore
            move = None

            if curr_node.visited:
                # Backtracking logic
                while backtrack:
                    # Check for unvisited neighbors
                    for direction in ['right', 'left', 'forward']:
                        new_coords = calcCoords(direction, curr_node.coordinates, curr_node.front)
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
                curr_coords, imageL, imageF, imageR, possible_movements = populate_node(bot)
                curr_node.set(imageL, imageF, imageR, possible_movements, True)

                # Explore possible movements
                for direction in ['right', 'left', 'forward']:
                    new_coords = calcCoords(direction, curr_coords, curr_node.front)

                    if new_coords not in env_tree:  # Add new node to the tree
                        new_node = Node(new_coords)
                        env_tree[new_coords] = new_node

                        if direction in curr_node.movements:
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
                send_movement(move)

                # see if we need a wait time
            else:
                print(f"Robot {bot} has no valid moves left and will wait.")

    print("Exploration complete. All accessible nodes have been visited.")
    return env_tree


def populate_node(robot_id):
    """Simulate receiving data from the robot."""
    coordinates = (robot_id, 0)  # get position of the robot
    imageL, imageF, imageR, depthL, depthF, depthR = captureImage()
    possible_movements = depthCalc(depthL, depthF, depthR, step)
    return coordinates, imageL, imageF, imageR, possible_movements


def captureImage():
    """Capture the images to the left, right, and front of the robot."""
    # Need sim controls from Hari

    # rotate 90 degrees CCW
    imageL = print('Left image captured')
    depthL = print('Left depth image captured')

    # rotate 90 degrees CW
    imageF = print('Front image captured')
    depthF = print('Front depth image captured')

    # rotate 90 degrees CW
    imageR = print('Right image captured')
    depthR = print('Right depth image captured')

    return imageL, imageF, imageR, depthL, depthF, depthR


def depthCalc(depthL, depthF, depthR, threshold):
    """
    Determine valid movement directions based on depth images.

    Parameters:
        depthL (np.ndarray): Depth image for the left direction.
        depthF (np.ndarray): Depth image for the forward direction.
        depthR (np.ndarray): Depth image for the right direction.
        threshold (float): Minimum depth value to consider a path valid.

    Returns:
        list: A list of directions with valid movements.
              Example: ["left", "forward"]
    """

    # Define a region of interest (ROI) as a center square of each image
    def is_valid_depth(depth_image):
        # Extract the ROI (e.g., center 20% of the image)
        h, w = depth_image.shape
        roi = depth_image[h // 4:3 * h // 4, w // 4:3 * w // 4]
        # Check if the mean depth in the ROI exceeds the threshold
        return np.mean(roi) > threshold

    # Check validity for each direction and build the result list
    valid_movements = []
    if is_valid_depth(depthL):
        valid_movements.append("left")
    if is_valid_depth(depthF):
        valid_movements.append("forward")
    if is_valid_depth(depthR):
        valid_movements.append("right")

    return valid_movements

def calcCoords(curr_coords, direction, face):
    x, y = curr_coords
    if direction == 'forward':
        if face == 'North':
            return (x, y + step)
        elif face == 'South':
            return (x, y - step)
        elif face == 'East':
            return (x + step, y)
        elif face == 'West':
            return (x - step, y)
    elif direction == 'backward':
        if face == 'North':
            return (x, y - step)
        elif face == 'South':
            return (x, y + step)
        elif face == 'East':
            return (x - step, y)
        elif face == 'West':
            return (x + step, y)
    elif direction == 'left':
        if face == 'North':
            return (x - step, y)
        elif face == 'South':
            return (x + step, y)
        elif face == 'East':
            return (x, y - step)
        elif face == 'West':
            return (x, y + step)
    elif direction == 'right':
        if face == 'North':
            return (x + step, y)
        elif face == 'South':
            return (x - step, y)
        elif face == 'East':
            return (x, y + step)
        elif face == 'West':
            return (x, y - step)
    return curr_coords


def object_detection():
    # Dev's detection network

    obj = print('object type')

    return obj

def send_movement(move):
    # Send movement to Gazebo

    print(move)

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
