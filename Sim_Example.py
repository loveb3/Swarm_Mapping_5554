import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.animation import FuncAnimation

class MultiRobotEnvironment:
    def __init__(self, rows, cols, obstacle_coords=None):
        self.rows = rows
        self.cols = cols
        self.obstacle_coords = obstacle_coords if obstacle_coords else []
        self.robots = []  # List of robots in the environment
        self.grid = np.zeros((rows, cols), dtype=int)  # 0 for free, 1 for obstacle

        # Place obstacles
        for x, y in self.obstacle_coords:
            self.grid[x, y] = 1

    def add_robot(self, position, direction='North'):
        """Add a robot to the environment."""
        self.robots.append({'position': position, 'direction': direction})

    def move_robot(self, robot_idx, direction):
        """Move a robot in the specified direction if possible."""
        x, y = self.robots[robot_idx]['position']
        if direction == 'up' and x > 0 and self.grid[x - 1, y] == 0:
            self.robots[robot_idx]['position'] = (x - 1, y)
            self.robots[robot_idx]['direction'] = 'North'
        elif direction == 'down' and x < self.rows - 1 and self.grid[x + 1, y] == 0:
            self.robots[robot_idx]['position'] = (x + 1, y)
            self.robots[robot_idx]['direction'] = 'South'
        elif direction == 'left' and y > 0 and self.grid[x, y - 1] == 0:
            self.robots[robot_idx]['position'] = (x, y - 1)
            self.robots[robot_idx]['direction'] = 'West'
        elif direction == 'right' and y < self.cols - 1 and self.grid[x, y + 1] == 0:
            self.robots[robot_idx]['position'] = (x, y + 1)
            self.robots[robot_idx]['direction'] = 'East'

    def display_environment(self, ax):
        """Draw the environment on the given Matplotlib axis."""
        ax.clear()
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_xticks(range(self.cols + 1))
        ax.set_yticks(range(self.rows + 1))
        ax.grid(True)

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

            ax.add_patch(Polygon(points, color='blue'))

        ax.set_title("Multi-Robot Environment")

# Initialize the environment
rows, cols = 10, 10
obstacles = [(3, 3), (5, 5), (6, 7)]
env = MultiRobotEnvironment(rows, cols, obstacle_coords=obstacles)

# Add robots to the environment
env.add_robot((0, 0), direction='East')
env.add_robot((9, 9), direction='West')

# Define the movement plan
robot_movements = [
    ['right', 'down', 'down', 'down'],
    ['left', 'up', 'up', 'up']
]

# Animation function
def update(frame):
    for robot_idx, moves in enumerate(robot_movements):
        if frame < len(moves):
            env.move_robot(robot_idx, moves[frame])

    env.display_environment(ax)

# Set up the Matplotlib figure
fig, ax = plt.subplots(figsize=(8, 8))
env.display_environment(ax)

# Create the animation
frames = max(len(m) for m in robot_movements)  # Number of frames is the longest movement sequence
ani = FuncAnimation(fig, update, frames=frames, repeat=False, interval=500)

plt.show()
