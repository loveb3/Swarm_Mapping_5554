import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.animation import FuncAnimation
from Env_Mapping import multiRobotDFS, Node  # Import the DFS logic
from Env_Mapping import dfs_ready, animation_ready # Import the signaling objects
import threading
from threading import Thread
from shared_queue import move_queue, bot_queue, face_queue

class MultiRobotEnvironment:
    def __init__(self, rows, cols, obstacle_coords=None):
        self.rows = rows
        self.cols = cols
        self.obstacle_coords = obstacle_coords if obstacle_coords else []
        self.robots = []  # List of robots in the environment
        self.grid = np.zeros((rows, cols), dtype=int)  # 0 for free, 1 for obstacle
        self.visited_nodes = set()  # Set to track visited nodes

        # Place obstacles
        for x, y in self.obstacle_coords:
            self.grid[x, y] = 1

    def add_robot(self, position, direction='South'):
        """Add a robot to the environment."""
        self.robots.append({'position': position, 'direction': direction})
        self.visited_nodes.add(position)  # Mark the initial position as visited

    def move_robot(self, robot_idx, direction, face):
        """
        Move a robot in the specified direction relative to its face if possible.

        Args:
            robot_idx (int): Index of the robot to move.
            direction (str): Direction to move ('forward', 'backward', 'left', 'right').
            face (str): New orientation of the robot ('North', 'South', 'East', 'West').
        """
        # Get the robot's current position
        # x, y = self.robots[robot_idx]['position']
        #
        # # Define movement deltas based on the relative direction and face
        # movement = {
        #     'North': {'forward': (-1, 0), 'backward': (1, 0), 'left': (0, -1), 'right': (0, 1)},
        #     'South': {'forward': (1, 0), 'backward': (-1, 0), 'left': (0, 1), 'right': (0, -1)},
        #     'East': {'forward': (0, 1), 'backward': (0, -1), 'left': (-1, 0), 'right': (1, 0)},
        #     'West': {'forward': (0, -1), 'backward': (0, 1), 'left': (1, 0), 'right': (-1, 0)},
        # }
        #
        # # Get the delta for the movement
        # dx, dy = movement[face][direction]
        #
        # # Calculate the new position
        # new_x, new_y = x + dx, y + dy

        new_y, new_x = direction

        # Check if the new position is within bounds and not blocked
        if 0 <= new_x < self.rows and 0 <= new_y < self.cols and self.grid[new_x, new_y] == 0:
            # Update the robot's position and direction
            self.robots[robot_idx]['position'] = (new_x, new_y)
            self.robots[robot_idx]['direction'] = face
            self.visited_nodes.add((new_x, new_y))  # Mark the new position as visited

    def get_robot_position(self, robot_idx):
        """Get the position of the specified robot."""
        return self.robots[robot_idx]['position']

    def display_environment(self, ax):
        """Draw the environment on the given Matplotlib axis."""
        ax.clear()
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_xticks(range(self.cols + 1))
        ax.set_yticks(range(self.rows + 1))
        ax.grid(True)

        # Draw visited nodes with a green tint
        for x, y in self.visited_nodes:
            ax.add_patch(Rectangle((y, self.rows - 1 - x), 1, 1, color='green', alpha=0.3))

        # Draw obstacles
        for x, y in self.obstacle_coords:
            ax.add_patch(Rectangle((y, self.rows - 1 - x), 1, 1, color='red'))

        # Draw robots
        for robot in self.robots:
            x, y = robot['position']
            direction = robot['direction']
            robot_x = y + 0.5
            robot_y = self.rows - 1 - x + 0.5

            # Triangle points for each direction
            if direction == 'North':
                points = [(robot_x, robot_y + 0.3), (robot_x - 0.2, robot_y - 0.3), (robot_x + 0.2, robot_y - 0.3)]
            elif direction == 'South':
                points = [(robot_x, robot_y - 0.3), (robot_x - 0.2, robot_y + 0.3), (robot_x + 0.2, robot_y + 0.3)]
            elif direction == 'West':
                points = [(robot_x - 0.3, robot_y), (robot_x + 0.3, robot_y + 0.2), (robot_x + 0.3, robot_y - 0.2)]
            elif direction == 'East':
                points = [(robot_x + 0.3, robot_y), (robot_x - 0.3, robot_y + 0.2), (robot_x - 0.3, robot_y - 0.2)]
            # Assign a color to the robot based on its index or other criteria
            robot_index = self.robots.index(robot)
            colors = ['blue', 'green', 'purple', 'yellow', 'red']  # Add more colors as needed
            color = colors[robot_index % len(colors)]

            ax.add_patch(Polygon(points, color=color))

        ax.set_title("Multi-Robot Environment")

# Initialize the environment
# rows, cols = 10, 10
# obstacles = [(3, 3), (5, 5), (6, 7)]
rows, cols = 3, 3
obstacles = [(0, 0), (0, 2), (2, 0)]
env = MultiRobotEnvironment(rows, cols, obstacle_coords=obstacles)

# Add robots to the environment
initial_positions = [(0, 1), (3, 3)]
env.add_robot((0, 1), direction='East')
env.add_robot((2, 2), direction='West')

# Set up DFS mapping
initial_data = [((None, None, None), ['forward', 'right', 'left'])] * len(initial_positions)
dfs_thread = Thread(target=multiRobotDFS, args=(initial_positions, initial_data))
dfs_thread.start()

# Animation function
def update(frame):
    # Wait for the DFS to signal that it's ready
    animation_ready.wait()
    animation_ready.clear()  # Reset the signal for the next iteration

    # for robot_idx, robot in enumerate(env.robots):
    #     # Retrieve the next move from the DFS control logic
    #     position = env.get_robot_position(robot_idx)
    #     if not move_queue.empty():
    #         move = move_queue.get()
    #         env.move_robot(robot_idx, move)
    #
    #     if node and node.movements:
    #         move = node.movements.pop(0)  # Use the next movement

    if not move_queue.empty():
        robot_idx = bot_queue.get()
        move = move_queue.get()
        face = face_queue.get()
        env.move_robot(robot_idx, move, face)
        print(f'Update says bot {robot_idx} will move {move}')


    env.display_environment(ax)

    # Notify DFS that the animation step is complete
    dfs_ready.set()

# Set up the Matplotlib figure
fig, ax = plt.subplots(figsize=(8, 8))
env.display_environment(ax)

# Create the animation
ani = FuncAnimation(fig, update, frames=100, repeat=False, interval=500)

plt.show(block=True)
plt.close(fig)