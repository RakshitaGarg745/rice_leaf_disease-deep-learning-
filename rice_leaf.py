# ==============================================================
# RICE LEAF DISEASE CLASSIFIER
# All plots are automatically saved to the 'results' folder
# Run this file in VS Code terminal:
#   python rice_leaf.py
# ==============================================================

import os
import json
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # <-- THIS IS THE KEY FIX for VS Code .py files
                        # 'Agg' = save to file instead of opening popup windows
                        # Without this, plt.show() opens a window BUT doesn't save
import seaborn as sns
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from models.resnet50_model import build_resnet50
from models.vgg16_model import build_vgg16
from models.efficientnet_model import build_efficientnet
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

print("TensorFlow version:", tf.__version__)
print("All libraries imported successfully! ✅")


# ==============================================================
# STEP 1 — CREATE RESULTS FOLDER
# All charts/plots will be saved here automatically
# ==============================================================

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"\n📁 Results folder ready: {os.path.abspath(RESULTS_DIR)}")
print("   Every plot will be saved there as a .png file automatically.\n")


# ==============================================================
# STEP 2 — SET YOUR DATA PATH
# *** CHANGE THIS to match your actual folder name ***
# ==============================================================

DATA_DIR = '/Users/rakshitagarg/rice_leaf_disease-deep-learning-/rice_leaf_diseases'

# Safety check — tells you exactly what's found
print("=" * 55)
print("CHECKING YOUR DATASET FOLDER")
print("=" * 55)

if not os.path.exists(DATA_DIR):
    print(f"❌ Folder not found: {DATA_DIR}")
    print("   Please update DATA_DIR in this file to your correct path.")
    exit()

# List all subfolders (= disease classes)
all_items = os.listdir(DATA_DIR)
classes   = sorted([
    c for c in all_items
    if os.path.isdir(os.path.join(DATA_DIR, c)) and not c.startswith('.')
])

print(f"✅ Folder found!")
print(f"\n📋 Disease classes detected ({len(classes)} total):")
total_images = 0
for cls in classes:
    cls_path = os.path.join(DATA_DIR, cls)
    imgs     = [f for f in os.listdir(cls_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images += len(imgs)
    print(f"   {cls:30s} → {len(imgs)} images")

print(f"\n📊 Total images: {total_images}")


# ==============================================================
# STEP 3 — SAVE SAMPLE IMAGES PREVIEW
# Shows 2 sample images from each disease class
# Saved as: results/01_sample_images.png
# ==============================================================

print("\n" + "=" * 55)
print("SAVING PLOT 1/4 — Sample images preview")
print("=" * 55)

cols = len(classes)
fig, axes = plt.subplots(2, cols, figsize=(cols * 4, 8))
fig.suptitle('Rice Leaf Disease — Sample Images\n(2 per class)',
             fontsize=15, fontweight='bold', y=1.01)

for col, cls in enumerate(classes):
    cls_path  = os.path.join(DATA_DIR, cls)
    img_files = [f for f in os.listdir(cls_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for row in range(2):
        if row < len(img_files):
            img = Image.open(os.path.join(cls_path, img_files[row]))
            axes[row][col].imshow(img)
        else:
            axes[row][col].set_facecolor('#f0f0f0')

        if row == 0:
            axes[row][col].set_title(
                cls.replace('_', '\n'), fontsize=10, fontweight='bold'
            )
        axes[row][col].axis('off')

plt.tight_layout()

save_path = os.path.join(RESULTS_DIR, '01_sample_images.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()   # close the figure to free memory
print(f"✅ Saved → {save_path}")


# ==============================================================
# STEP 4 — IMAGE PREPROCESSING AND AUGMENTATION
# ==============================================================

print("\n" + "=" * 55)
print("SETTING UP IMAGE PREPROCESSING")
print("=" * 55)

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16
NUM_CLASSES = len(classes)

# Training data — with augmentation (creates fake variety)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,        # normalize pixels from 0-255 to 0.0-1.0
    rotation_range=40,         # rotate randomly up to 40 degrees
    horizontal_flip=True,      # randomly flip left-right
    zoom_range=0.2,            # randomly zoom in by 20%
    width_shift_range=0.15,    # shift left/right
    height_shift_range=0.15,   # shift up/down
    shear_range=0.1,           # slight skew distortion
    brightness_range=[0.8, 1.2],
    validation_split=0.2       # 20% of images reserved for validation
)

# Validation data — NO augmentation, just normalize
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

CLASS_NAMES  = list(train_generator.class_indices.keys())
print(f"\n✅ Data ready!")
print(f"   Training images  : {train_generator.samples}")
print(f"   Validation images: {val_generator.samples}")
print(f"   Class labels     : {CLASS_NAMES}")


# ==============================================================
# STEP 5 — BUILD THE MODEL (Transfer Learning)
# Uses MobileNetV2 pre-trained on ImageNet
# ==============================================================

print("\n" + "=" * 55)
print("BUILDING ALL MODELS")
print("=" * 55)

def build_mobilenet_model():
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=outputs)

# Build all models
models_dict = {
    "MobileNetV2": build_mobilenet_model(),
    "ResNet50": build_resnet50(),
    "VGG16": build_vgg16(),
    "EfficientNet": build_efficientnet()
}
print("\n" + "=" * 55)
print("TRAINING ALL MODELS + METRICS")
print("=" * 55)

import pandas as pd

metrics_dict = {}

for name, model in models_dict.items():
    print(f"\n🚀 Training {name}...")

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        verbose=1
    )

    # Predictions for metrics
    val_generator.reset()
    y_pred_probs = model.predict(val_generator, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_generator.classes

    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
    )

    # Store metrics
    metrics_dict[name] = {
        "accuracy": report["accuracy"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1_score": report["macro avg"]["f1-score"]
    }

    # Save model
    model.save(f"{name}.keras")

    print(f"✅ {name} completed!")

# ==============================================================
# CREATE COMPARISON TABLE
# ==============================================================

df = pd.DataFrame(metrics_dict).T
df = df.round(4)

print("\n📊 MODEL COMPARISON TABLE:")
print(df)

table_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
df.to_csv(table_path)
print(f"✅ Table saved → {table_path}")

# ==============================================================
# CREATE COMPARISON GRAPH
# ==============================================================

plt.figure(figsize=(10,6))
df.plot(kind='bar')
plt.title("Model Comparison (Accuracy, Precision, Recall, F1)")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.ylim(0,1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

graph_path = os.path.join(RESULTS_DIR, "model_comparison_graph.png")
plt.savefig(graph_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Graph saved → {graph_path}")

# ==============================================================
# FIND BEST MODEL
# ==============================================================

best_model_name = df["accuracy"].idxmax()
best_accuracy = df["accuracy"].max()

print("\n🏆 BEST MODEL:")
print(f"{best_model_name} with accuracy: {best_accuracy*100:.2f}%")
print("\n" + "=" * 55)
print("TRAINING ALL MODELS")
print("=" * 55)

from sklearn.metrics import classification_report

metrics_dict = {}

val_generator.reset()
y_pred_probs = model.predict(val_generator, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_generator.classes

report = classification_report(
    y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
)

metrics_dict[name] = {
    "accuracy": report["accuracy"],
    "precision": report["macro avg"]["precision"],
    "recall": report["macro avg"]["recall"],
    "f1_score": report["macro avg"]["f1-score"]
}

print(f"✅ {name} done!")