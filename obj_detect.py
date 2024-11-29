import os
import tensorflow as tf
import pandas as pd  # Ensure pandas is imported
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from collections import Counter

# Paths to Pascal VOC 2012 dataset
BASE_DIR = r"C:\Users\devoj\OneDrive\Documents\Dev's Documents\Virginia Tech Classes\Fall 2024\ECE 5554 Computer Vision\ECE 5554 CV Project Group 10\obstacle_detection\VOCdevkit\VOC2012"
IMAGE_DIR = os.path.join(BASE_DIR, "JPEGImages")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "Annotations")
TRAIN_TXT = os.path.join(BASE_DIR, "ImageSets", "Main", "train.txt")
VAL_TXT = os.path.join(BASE_DIR, "ImageSets", "Main", "val.txt")

# Check dataset structure
if not os.path.exists(IMAGE_DIR):
    raise FileNotFoundError(f"JPEGImages directory not found: {IMAGE_DIR}")
if not os.path.exists(ANNOTATIONS_DIR):
    raise FileNotFoundError(f"Annotations directory not found: {ANNOTATIONS_DIR}")
if not os.path.exists(TRAIN_TXT):
    raise FileNotFoundError(f"train.txt not found: {TRAIN_TXT}")
if not os.path.exists(VAL_TXT):
    raise FileNotFoundError(f"val.txt not found: {VAL_TXT}")

# Parameters
BATCH_SIZE = 16
EPOCHS = 20
IMG_SIZE = (224, 224)
LEARNING_RATE = 1e-4
OUTPUT_DIR = r"C:\Users\devoj\OneDrive\Documents\Dev's Documents\Virginia Tech Classes\Fall 2024\ECE 5554 Computer Vision\ECE 5554 CV Project Group 10\obstacle_detection\output_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Debugging Function: Count classes in annotations
def count_classes(txt_file, annotations_dir):
    class_counts = Counter()
    with open(txt_file, "r") as f:
        for line in f:
            image_id = line.strip()
            annotation_file = os.path.join(annotations_dir, f"{image_id}.xml")
            if os.path.exists(annotation_file):
                tree = ET.parse(annotation_file)
                root = tree.getroot()
                for obj in root.findall("object"):
                    class_name = obj.find("name").text
                    class_counts[class_name] += 1
            else:
                print(f"Missing annotation file: {annotation_file}")
    return class_counts

# Display class distribution
print("Class distribution in training set:")
print(count_classes(TRAIN_TXT, ANNOTATIONS_DIR))
print("\nClass distribution in validation set:")
print(count_classes(VAL_TXT, ANNOTATIONS_DIR))

# Create data generator from Pascal VOC train/val text files
def load_pascal_voc_data(txt_file, image_dir, annotations_dir, target_size, batch_size):
    """
    Load Pascal VOC data and create a Keras-compatible data generator.
    """
    image_paths = []
    labels = []
    class_set = set()

    # Read the Pascal VOC txt file and gather image paths and labels
    with open(txt_file, "r") as f:
        for line in f:
            image_id = line.strip()
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            annotation_file = os.path.join(annotations_dir, f"{image_id}.xml")
            
            if not os.path.exists(image_path):
                print(f"Warning: Missing image {image_path}")
                continue
            if not os.path.exists(annotation_file):
                print(f"Warning: Missing annotation {annotation_file}")
                continue
            
            # Parse the XML annotation file
            tree = ET.parse(annotation_file)
            root = tree.getroot()
            object_classes = []
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                object_classes.append(class_name)
                class_set.add(class_name)

            if not object_classes:
                print(f"Warning: No objects found in {annotation_file}")
                continue

            # Store the image path and corresponding class (pick the first object for simplicity)
            image_paths.append(image_path)
            labels.append(object_classes[0])  # Use single-label classification

    # Map classes to indices
    class_indices = {class_name: idx for idx, class_name in enumerate(sorted(class_set))}
    print(f"Detected classes: {class_indices}")

    # Convert string labels to class indices
    label_indices = [class_indices[label] for label in labels]

    # Create a DataFrame
    data = pd.DataFrame({
        "filename": image_paths,
        "class": labels
    })

    # Create ImageDataGenerator and flow from the DataFrame
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    dataset = datagen.flow_from_dataframe(
        dataframe=data,
        x_col="filename",
        y_col="class",
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    return dataset, class_indices



# Load training and validation datasets
train_generator, train_class_indices = load_pascal_voc_data(TRAIN_TXT, IMAGE_DIR, ANNOTATIONS_DIR, IMG_SIZE, BATCH_SIZE)
valid_generator, val_class_indices = load_pascal_voc_data(VAL_TXT, IMAGE_DIR, ANNOTATIONS_DIR, IMG_SIZE, BATCH_SIZE)

# Build ResNet50 model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation="relu")(x)

x = Dense(len(train_class_indices), activation="softmax")(x)  # Multi-class classification
loss_function = "categorical_crossentropy"

model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=loss_function,
    metrics=["accuracy"],
)

# Callbacks
checkpoint = ModelCheckpoint(
    filepath=os.path.join(OUTPUT_DIR, "best_resnet_model.keras"),
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=1,
    restore_best_weights=True,
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping],
    verbose=1,
)

# Save the final model
model.save(os.path.join(OUTPUT_DIR, "final_resnet_model.keras"))

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()
