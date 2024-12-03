import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from xml.etree.ElementTree import parse

# Paths
MODEL_PATH = r"C:\Users\devoj\OneDrive\Documents\Dev's Documents\Virginia Tech Classes\Fall 2024\ECE 5554 Computer Vision\ECE 5554 CV Project Group 10\obstacle_detection\best_resnet_model-ep4.keras"
VOC_BASE_DIR = r"C:\Users\devoj\OneDrive\Documents\Dev's Documents\Virginia Tech Classes\Fall 2024\ECE 5554 Computer Vision\ECE 5554 CV Project Group 10\obstacle_detection\VOCdevkit\VOC2012"
IMAGE_DIR = os.path.join(VOC_BASE_DIR, "JPEGImages")
ANNOTATIONS_DIR = os.path.join(VOC_BASE_DIR, "Annotations")
OUTPUT_DIR = r"C:\Users\devoj\OneDrive\Documents\Dev's Documents\Virginia Tech Classes\Fall 2024\ECE 5554 Computer Vision\ECE 5554 CV Project Group 10\obstacle_detection\output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the trained model
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# Class labels (ensure this matches your trained model's classes)
CLASS_LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
                'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# Function to preprocess an image
def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    original_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image, original_image

# Function to draw bounding boxes
def draw_boxes(image, annotation_file, predictions, confidence_threshold=0.5):
    tree = parse(annotation_file)
    root = tree.getroot()

    for obj in root.findall("object"):
        label = obj.find("name").text
        confidence = predictions[0][CLASS_LABELS.index(label)]

        if confidence > confidence_threshold:
            # Get bounding box coordinates
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            # Draw bounding box
            color = (255, 0, 0)  # Blue color
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

            # Draw label and confidence
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(image, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

# Select 10 test images
image_files = sorted(os.listdir(IMAGE_DIR))[:10]

# Process each image
for image_file in image_files:
    image_path = os.path.join(IMAGE_DIR, image_file)
    annotation_file = os.path.join(ANNOTATIONS_DIR, f"{os.path.splitext(image_file)[0]}.xml")

    if not os.path.exists(annotation_file):
        print(f"Annotation file missing for image: {image_file}")
        continue

    # Preprocess the image
    image, original_image = preprocess_image(image_path)

    # Predict using the model
    predictions = model.predict(image)
    print(f"Predictions for {image_file}: {predictions}")

    # Draw bounding boxes
    output_image = draw_boxes(original_image, annotation_file, predictions)

    # Save the output image
    output_path = os.path.join(OUTPUT_DIR, image_file)
    cv2.imwrite(output_path, output_image)

    # Display the image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Predictions for {image_file}")
    plt.axis("off")
    plt.show()

print("Inference completed. Results saved in:", OUTPUT_DIR)
