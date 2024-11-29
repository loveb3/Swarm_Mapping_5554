import numpy as np
import torch
import cv2
from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt

class DepthEstimataor:
    def __init__(self, model="LiheYoung/depth-anything-base-hf"):
        # Use MPS if available
        # self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize depth estimation pipeline
        self.pipe = pipeline(
            task="depth-estimation", 
            model=model, 
            device=self.device
        )

    def estimate_depth(self, image_path):
        """
        Estimate depth for 416x416 image
        """
        # Open and process the image
        img = Image.open(image_path)
        
        # Perform depth estimation
        depth_result = self.pipe(img)
        depth_array = np.array(depth_result['depth'])
        
        return depth_array
    
    def get_object_depths(self, image_path, objects):
        """
        Calculate depth for multiple objects in an image.
        
        :param image_path: Path to the input image
        :param objects: List of dictionaries with bounding boxes in pixel coordinates
        :return: List of object depth information
        """
        
        # Estimate depth for the entire image
        depth_array = self.estimate_depth(image_path)
        
        # Store results for each object
        object_depths = []
        
        # Process each object
        for obj in objects:
            # Use bounding box coordinates directly (assume input in pixel values)
            xmin = int(obj['bbox'][0])
            ymin = int(obj['bbox'][1])
            xmax = int(obj['bbox'][2])
            ymax = int(obj['bbox'][3])
            
            # # Compute centroid
            # x_centroid = int((xmin + xmax) / 2)
            # y_centroid = int((ymin + ymax) / 2)
            
            # # Ensure centroid coordinates are within bounds
            # x_centroid = np.clip(x_centroid, 0, depth_array.shape[1] - 1)
            # y_centroid = np.clip(y_centroid, 0, depth_array.shape[0] - 1)
            
            # # Get depth at the centroid
            # centroid_depth = depth_array[y_centroid, x_centroid]
            
            # Calculate additional depth metrics for the ROI
            roi_depth = depth_array[ymin:ymax, xmin:xmax]
            obj_depth_info = {
                'class': obj.get('class', 'Unknown'),
                'bbox': {
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'width': xmax - xmin,
                    'height': ymax - ymin
                },
                'depth_stats': {
                    'average_distance': np.mean(roi_depth) if roi_depth.size > 0 else None,
                },
                # 'true_depth': centroid_depth,  # True depth is depth at the centroid
                # 'centroid': (x_centroid, y_centroid),
                'depth_grid': roi_depth
            }
            
            object_depths.append(obj_depth_info)
        
        return object_depths
    
    def closest_object(self, depth_info):
        max_avg_distance_object = max(depth_info, key=lambda obj: obj['depth_stats']['average_distance'])
        close = [max_avg_distance_object['class'], max_avg_distance_object['depth_stats']['average_distance']] 
        return close   

    

# depth_estimator = DepthEstimataor()
# depth_estimator.get_object_depths(image_path, objects)
# depth_value = depth_estimator['depth_stats']['average_distance']