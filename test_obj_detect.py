import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Paths
MODEL_PATH = r"C:\Users\devoj\OneDrive\Documents\Dev's Documents\Virginia Tech Classes\Fall 2024\ECE 5554 Computer Vision\ECE 5554 CV Project Group 10\obstacle_detection\output_model\best_resnet_model.keras"
# Change this path for the test images
TEST_IMAGE_DIR = r"C:\Users\devoj\OneDrive\Documents\Dev's Documents\Virginia Tech Classes\Fall 2024\ECE 5554 Computer Vision\ECE 5554 CV Project Group 10\obstacle_detection\Test_images"

# Load the trained model
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# Class labels (ensure this matches your trained model's classes)
CLASS_LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
                'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# Target classes
TARGET_CLASSES = {'chair', 'diningtable', 'sofa'}

# Function to preprocess an image
def preprocess_image(image_path, target_size=(224, 224)):
    print(f"Reading image from: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}. Check if the file exists and is a valid image format.")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Process up to 10 test images
test_images = [f for f in os.listdir(TEST_IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))]

results = []  # Store results (image name and confidence scores)

for image_file in test_images:
    image_path = os.path.join(TEST_IMAGE_DIR, image_file)

    try:
        # Preprocess the image
        image = preprocess_image(image_path)

        # Predict using the model
        predictions = model.predict(image)

        # Extract confidence scores for target classes
        target_scores = {cls: predictions[0][CLASS_LABELS.index(cls)] for cls in TARGET_CLASSES}
        results.append({"image_name": image_file, "scores": target_scores})
    except FileNotFoundError as e:
        print(e)

# Output the results
print("\nResults:")
for result in results:
    print(f"Image: {result['image_name']}")
    for cls in TARGET_CLASSES:
        conf = result["scores"].get(cls, 0.0)
        print(f"  {cls}: {conf:.2f}")
