import os
import cv2
import numpy as np
import torch
from typing import List, Dict, Optional

class GridNavigator:
    def __init__(self, grid_size: int = 3):
        """
        Initialize the grid navigator with MiDaS depth estimation.
        Args:
            grid_size: Size of the grid (default 3x3)
        """
        # Set default directories within the class
        self.base_dir = os.path.expanduser("~/Desktop/PythonPrograms/CV_project_code")
        self.image_directory = os.path.join(self.base_dir, "cv_images_png")
        self.depth_save_directory = os.path.join(self.base_dir, "cv_images_depth")
        self.grid_size = grid_size
        
        # Ensure directories exist
        if not os.path.exists(self.image_directory):
            os.makedirs(self.image_directory)
        if not os.path.exists(self.depth_save_directory):
            os.makedirs(self.depth_save_directory)
        
        # Initialize MiDaS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = "MiDaS_small"
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas.to(self.device)
        self.midas.eval()
        
        # Load transforms
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = self.midas_transforms.small_transform

    def set_directories(self, image_dir: str = None, depth_dir: str = None):
        """
        Optionally update directories after initialization
        """
        if image_dir:
            self.image_directory = image_dir
            if not os.path.exists(self.image_directory):
                os.makedirs(self.image_directory)
                
        if depth_dir:
            self.depth_save_directory = depth_dir
            if not os.path.exists(self.depth_save_directory):
                os.makedirs(self.depth_save_directory)

    def load_image(self, row: int, col: int, direction: str) -> Optional[np.ndarray]:
        """
        Load PNG image from directory.
        """
        filename = f"IMG_{direction}({row}x{col}).png"
        path = os.path.join(self.image_directory, filename)
        
        if os.path.exists(path):
            return cv2.imread(path)
        
        print(f"No image found for position ({row}, {col}) facing {direction}")
        return None

    def compute_and_save_depth_map(self, img: np.ndarray, filename: str) -> Optional[np.ndarray]:
        """
        Compute depth map using MiDaS and save it.
        """
        if img is None:
            return None
            
        try:
            # Compute depth map
            input_batch = self.transform(img).to(self.device)
            
            with torch.no_grad():
                prediction = self.midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                
            depth_map = prediction.cpu().numpy()
            depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Save depth map
            save_path = os.path.join(self.depth_save_directory, filename)
            cv2.imwrite(save_path, depth_map)
            
            # Return normalized depth map (0-1 range for processing)
            return depth_map.astype(np.float32) / 255.0
            
        except Exception as e:
            print(f"Error computing depth map: {str(e)}")
            return None

    def detect_walls(self, row: int, col: int) -> Dict[str, bool]:
        """
        Detect walls based on grid position.
        """
        walls = {
            'N': row == self.grid_size,  # Wall to the North if at max row
            'S': row == 1,               # Wall to the South if at min row
            'E': col == self.grid_size,  # Wall to the East if at max column
            'W': col == 1                # Wall to the West if at min column
        }
        return walls

    def check_direction_for_obstacles(self, depth_map: np.ndarray, threshold: float = 0.7) -> bool:
        """
        Check for obstacles in the forward direction of a depth map.
        """
        if depth_map is None:
            return True
            
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        height, width = depth_uint8.shape
        central_region = depth_uint8[height//3:2*height//3, width//3:2*width//3]
        
        _, binary = cv2.threshold(central_region, int(255 * threshold), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
        
        return len(significant_contours) > 0

    def detect_obstacles(self, row: int, col: int) -> Dict[str, bool]:
        """
        Detect obstacles in each direction.
        """
        obstacles = {}
        for direction in ['N', 'E', 'S', 'W']:
            img = self.load_image(row, col, direction)
            if img is None:
                obstacles[direction] = True
                continue
                
            filename = f"IMG_{direction}({row}x{col}).png"
            depth_map = self.compute_and_save_depth_map(img, filename)
            obstacles[direction] = self.check_direction_for_obstacles(depth_map)
        
        return obstacles

    def get_possible_moves_facing_north(self, row: int, col: int) -> List[str]:
        """
        Get possible moves from current position when facing North.
        Returns a list of possible moves ("Forward", "Left", "Right", "Backward").
        """
        possible_moves = []
        # This piece of code is to handle the couch error.
        if row == 2 and col == 2 or row == 1 and col == 2:
            possible_moves = ["Left"]
        
        # Get walls and obstacles
        walls = self.detect_walls(row, col)
        obstacles = self.detect_obstacles(row, col)
        
        # When facing North:
        # - Forward is North
        # - Left is West
        # - Right is East
        # - Backward is South
        
        # Check forward movement (North)
        if not walls['N'] and not obstacles['N']:
            possible_moves.append("Forward")
            
        # Check left turn (West)
        if not walls['W'] and not obstacles['W']:
            possible_moves.append("Left")
            
        # Check right turn (East)
        if not walls['E'] and not obstacles['E']:
            possible_moves.append("Right")
            
        # Check backward movement (South)
        if not walls['S'] and not obstacles['S']:
            possible_moves.append("Backward")
            
        return possible_moves
    


# #Exmaple - 
# navigator = GridNavigator()
# moves = navigator.get_possible_moves_facing_north(1, 1)
# print(moves)

# # If you need to change directories later
# navigator.set_directories(
#     image_dir="/Users/harisumant/Desktop/PythonPrograms/CV_project_code/cv_images_png",
#     depth_dir="/Users/harisumant/Desktop/PythonPrograms/CV_project_code/cv_images_depth"
# )