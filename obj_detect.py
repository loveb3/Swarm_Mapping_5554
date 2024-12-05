import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import matplotlib.pyplot as plt
from IPython.display import clear_output
import xml.etree.ElementTree as ET

# Paths to Pascal VOC 2012 dataset
BASE_DIR = r"C:\Users\devoj\OneDrive\Documents\Dev's Documents\Virginia Tech Classes\Fall 2024\ECE 5554 Computer Vision\ECE 5554 CV Project Group 10\obstacle_detection\VOCdevkit\VOC2012"
IMAGE_DIR = os.path.join(BASE_DIR, "JPEGImages")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "Annotations")
TRAIN_TXT = os.path.join(BASE_DIR, "ImageSets", "Main", "train.txt")
VAL_TXT = os.path.join(BASE_DIR, "ImageSets", "Main", "val.txt")

# Parameters
BATCH_SIZE = 16
EPOCHS = 20
IMG_SIZE = (224, 224)
LEARNING_RATE = 1e-4
OUTPUT_DIR = r"C:\Users\devoj\OneDrive\Documents\Dev's Documents\Virginia Tech Classes\Fall 2024\ECE 5554 Computer Vision\ECE 5554 CV Project Group 10\obstacle_detection\output_model\vis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to load Pascal VOC data
def load_pascal_voc_data(txt_file, image_dir, annotations_dir, target_size, batch_size):
    image_paths = []
    labels = []
    class_set = set()

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

            image_paths.append(image_path)
            labels.append(object_classes[0])  # Use single-label classification

    class_indices = {class_name: idx for idx, class_name in enumerate(sorted(class_set))}
    label_indices = [class_indices[label] for label in labels]

    data = pd.DataFrame({"filename": image_paths, "class": labels})
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

# Load datasets
train_generator, train_class_indices = load_pascal_voc_data(TRAIN_TXT, IMAGE_DIR, ANNOTATIONS_DIR, IMG_SIZE, BATCH_SIZE)
valid_generator, val_class_indices = load_pascal_voc_data(VAL_TXT, IMAGE_DIR, ANNOTATIONS_DIR, IMG_SIZE, BATCH_SIZE)

# Build ResNet50 model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation="relu")(x)
x = Dense(len(train_class_indices), activation="softmax")(x)  # Multi-class classification

model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="categorical_crossentropy", metrics=["accuracy"])

# Callback for batch-wise graph
class BatchAccuracyPlot(Callback):
    def __init__(self):
        super().__init__()
        self.batch_losses = []
        self.batch_accuracies = []

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.batch_losses.append(logs.get("loss"))
        self.batch_accuracies.append(logs.get("accuracy"))

    def on_epoch_end(self, epoch, logs=None):
        # Plot batch-wise loss and accuracy
        clear_output(wait=True)
        plt.figure(figsize=(10, 5))

        # Plot batch-wise loss
        plt.subplot(1, 2, 1)
        plt.plot(self.batch_losses, label="Batch Loss")
        plt.title(f"Loss During Epoch {epoch + 1}")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.legend()

        # Plot batch-wise accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.batch_accuracies, label="Batch Accuracy")
        plt.title(f"Accuracy During Epoch {epoch + 1}")
        plt.xlabel("Batch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Reset batch metrics for the next epoch
        self.batch_losses = []
        self.batch_accuracies = []

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
batch_plot = BatchAccuracyPlot()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping, batch_plot],
    verbose=1,
)

# Save the final model
model.save(os.path.join(OUTPUT_DIR, "final_resnet_model.keras"))
