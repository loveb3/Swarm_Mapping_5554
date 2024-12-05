import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Goes in Main()
# MODEL_PATH = r"C:\Users\devoj\OneDrive\Documents\Dev's Documents\Virginia Tech Classes\Fall 2024\ECE 5554 Computer Vision\ECE 5554 CV Project Group 10\obstacle_detection\best_resnet_model-ep4.keras"
# model = load_model(MODEL_PATH)

def object_detection(direction,row,col, model):

    grid_coord = f"/Users/harisumant/Desktop/PythonPrograms/CV_project_code/cv_images_new/IMG_{direction}({row}x{col}).png"
    # Class labels (ensure this matches your trained model's classes)
    CLASS_LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
                    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    # Target classes
    TARGET_CLASSES = {'chair', 'diningtable', 'sofa'}
    
    target_size=(224, 224)

    # Change according to where it is being run, add main Image directory
    image_path = grid_coord #+ ".png"
    # Function to preprocess an image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}. Check if the file exists and is a valid image format.")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    

    results = []  # Store results (image name and confidence scores)
    # Predict using the model
    predictions = model.predict(image)

    # Extract confidence scores for target classes
    target_scores = {cls: predictions[0][CLASS_LABELS.index(cls)] for cls in TARGET_CLASSES}
    results.append({"image_name": grid_coord, "scores": target_scores})

    return results
