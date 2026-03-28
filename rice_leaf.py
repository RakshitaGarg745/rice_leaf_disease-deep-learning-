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

DATA_DIR = r'C:\Users\divya\OneDrive\Desktop\deep learning\rice_leaf_diseases'

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
print("BUILDING MODEL")
print("=" * 55)

print("Loading MobileNetV2 pre-trained weights...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False   # freeze base — don't change ImageNet weights

# Add our custom disease classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

total_params     = model.count_params()
trainable_params = sum(
    tf.size(v).numpy() for v in model.trainable_variables
)
print(f"✅ Model built!")
print(f"   Total parameters    : {total_params:,}")
print(f"   Trainable (ours)    : {trainable_params:,}")
print(f"   Frozen (ImageNet)   : {total_params - trainable_params:,}")


# ==============================================================
# STEP 6 — CALLBACKS (smart training helpers)
# ==============================================================

callbacks = [
    ModelCheckpoint(
        filepath='best_rice_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=12,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]


# ==============================================================
# STEP 7 — TRAIN PHASE 1 (frozen base, train only custom head)
# ==============================================================

print("\n" + "=" * 55)
print("TRAINING — PHASE 1 (Custom head only)")
print("=" * 55)

PHASE1_EPOCHS = 20

history1 = model.fit(
    train_generator,
    epochs=PHASE1_EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

best_p1 = max(history1.history['val_accuracy'])
print(f"\n✅ Phase 1 done! Best val accuracy: {best_p1:.2%}")


# ==============================================================
# STEP 8 — SAVE TRAINING CURVES PLOT (Phase 1)
# Saved as: results/02_training_curves_phase1.png
# ==============================================================

print("\n" + "=" * 55)
print("SAVING PLOT 2/4 — Training curves (Phase 1)")
print("=" * 55)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Phase 1 Training — Custom Head Only', fontsize=14, fontweight='bold')

epochs_ran = range(1, len(history1.history['accuracy']) + 1)

# Accuracy chart
axes[0].plot(epochs_ran, history1.history['accuracy'],
             label='Train', color='#2196F3', linewidth=2)
axes[0].plot(epochs_ran, history1.history['val_accuracy'],
             label='Validation', color='#4CAF50', linewidth=2, linestyle='--')
axes[0].set_title('Accuracy per Epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1])

# Loss chart
axes[1].plot(epochs_ran, history1.history['loss'],
             label='Train', color='#F44336', linewidth=2)
axes[1].plot(epochs_ran, history1.history['val_loss'],
             label='Validation', color='#FF9800', linewidth=2, linestyle='--')
axes[1].set_title('Loss per Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(RESULTS_DIR, '02_training_curves_phase1.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved → {save_path}")


# ==============================================================
# STEP 9 — FINE-TUNING PHASE 2 (unfreeze top 30 base layers)
# ==============================================================

print("\n" + "=" * 55)
print("TRAINING — PHASE 2 (Fine-tuning)")
print("=" * 55)

base_model.trainable = True
UNFREEZE_FROM = len(base_model.layers) - 30

for i, layer in enumerate(base_model.layers):
    layer.trainable = (i >= UNFREEZE_FROM)

trainable_now = sum(1 for l in base_model.layers if l.trainable)
print(f"Unfroze top {trainable_now} layers of MobileNetV2")

model.compile(
    optimizer=Adam(learning_rate=1e-5),   # very small — don't destroy ImageNet knowledge!
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

initial_epoch = len(history1.history['accuracy'])

history2 = model.fit(
    train_generator,
    epochs=initial_epoch + 25,
    initial_epoch=initial_epoch,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

if history2.history['val_accuracy']:
    best_p2 = max(history2.history['val_accuracy'])
    print(f"\n✅ Phase 2 done! Best val accuracy: {best_p2:.2%}")


# ==============================================================
# STEP 10 — SAVE TRAINING CURVES PLOT (Phase 2)
# Saved as: results/03_training_curves_phase2.png
# ==============================================================

print("\n" + "=" * 55)
print("SAVING PLOT 3/4 — Training curves (Phase 2)")
print("=" * 55)

if history2.history['val_accuracy']:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Phase 2 Training — Fine-Tuning', fontsize=14, fontweight='bold')

    epochs_ran2 = range(initial_epoch + 1,
                        initial_epoch + len(history2.history['accuracy']) + 1)

    axes[0].plot(epochs_ran2, history2.history['accuracy'],
                 label='Train', color='#2196F3', linewidth=2)
    axes[0].plot(epochs_ran2, history2.history['val_accuracy'],
                 label='Validation', color='#4CAF50', linewidth=2, linestyle='--')
    axes[0].set_title('Accuracy per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    axes[1].plot(epochs_ran2, history2.history['loss'],
                 label='Train', color='#F44336', linewidth=2)
    axes[1].plot(epochs_ran2, history2.history['val_loss'],
                 label='Validation', color='#FF9800', linewidth=2, linestyle='--')
    axes[1].set_title('Loss per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, '03_training_curves_phase2.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved → {save_path}")


# ==============================================================
# STEP 11 — EVALUATE: Accuracy, Precision, Recall
# ==============================================================

print("\n" + "=" * 55)
print("EVALUATING MODEL")
print("=" * 55)

print("Loading best saved model...")
best_model = load_model('best_rice_model.keras')

val_generator.reset()
print("Running predictions on all validation images...")
y_pred_probs = best_model.predict(val_generator, verbose=1)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = val_generator.classes

print("\n" + "=" * 55)
print("FULL CLASSIFICATION REPORT")
print("=" * 55)
report = classification_report(
    y_true, y_pred, target_names=CLASS_NAMES, digits=3
)
print(report)

# Save the report as a text file too
report_path = os.path.join(RESULTS_DIR, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write("RICE LEAF DISEASE — CLASSIFICATION REPORT\n")
    f.write("=" * 55 + "\n")
    f.write(report)
print(f"✅ Report saved → {report_path}")


# ==============================================================
# STEP 12 — SAVE CONFUSION MATRIX PLOT
# Saved as: results/04_confusion_matrix.png
# ==============================================================

print("\n" + "=" * 55)
print("SAVING PLOT 4/4 — Confusion matrix")
print("=" * 55)

cm      = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Confusion Matrix — Rice Leaf Disease Classifier',
             fontsize=14, fontweight='bold')

# Left: raw counts
disp1 = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
disp1.plot(ax=axes[0], cmap='Blues', colorbar=False, xticks_rotation=30)
axes[0].set_title('Raw Counts\n(how many images per cell)', fontsize=11)

# Right: normalized percentages
disp2 = ConfusionMatrixDisplay(
    np.round(cm_norm, 2), display_labels=CLASS_NAMES
)
disp2.plot(ax=axes[1], cmap='Greens', colorbar=False, xticks_rotation=30)
axes[1].set_title('Percentage Correct Per Class\n(diagonal = correct)',
                  fontsize=11)

plt.tight_layout()
save_path = os.path.join(RESULTS_DIR, '04_confusion_matrix.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved → {save_path}")


# ==============================================================
# STEP 13 — PREDICT ON A RANDOM SAMPLE IMAGE
# Saved as: results/05_sample_prediction.png
# ==============================================================

print("\n" + "=" * 55)
print("SAVING BONUS PLOT — Sample prediction")
print("=" * 55)

random_class = random.choice(CLASS_NAMES)
cls_path     = os.path.join(DATA_DIR, random_class)
img_files    = [f for f in os.listdir(cls_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
test_path    = os.path.join(cls_path, random.choice(img_files))

img       = keras_image.load_img(test_path, target_size=(224, 224))
img_array = keras_image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

preds      = best_model.predict(img_array, verbose=0)[0]
pred_idx   = np.argmax(preds)
pred_class = CLASS_NAMES[pred_idx]
confidence = preds[pred_idx] * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Sample Prediction', fontsize=14, fontweight='bold')

axes[0].imshow(img)
axes[0].set_title(f'Actual class: {random_class}', fontsize=11)
axes[0].axis('off')

colors = ['#4CAF50' if i == pred_idx else '#90A4AE'
          for i in range(len(CLASS_NAMES))]
bars = axes[1].barh(CLASS_NAMES, preds * 100, color=colors, edgecolor='none')
axes[1].set_xlabel('Confidence (%)')
axes[1].set_xlim([0, 110])
axes[1].set_title(
    f'Predicted: {pred_class}\nConfidence: {confidence:.1f}%', fontsize=11
)
for bar, val in zip(bars, preds * 100):
    axes[1].text(val + 1, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}%', va='center', fontsize=10)

plt.tight_layout()
save_path = os.path.join(RESULTS_DIR, '05_sample_prediction.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved → {save_path}")

print(f"\nPrediction: {pred_class}  (confidence: {confidence:.1f}%)")
print(f"Actual    : {random_class}")


# ==============================================================
# STEP 14 — SAVE THE FINAL MODEL
# ==============================================================

best_model.save('rice_disease_final_model.keras')

with open('class_names.json', 'w') as f:
    json.dump(CLASS_NAMES, f)

# ==============================================================
# DONE — Print summary of everything saved
# ==============================================================

print("\n" + "=" * 55)
print("🎉 ALL DONE! Here is everything saved:")
print("=" * 55)
print(f"\n📁 results/ folder:")
for fname in sorted(os.listdir(RESULTS_DIR)):
    fpath = os.path.join(RESULTS_DIR, fname)
    fsize = os.path.getsize(fpath) // 1024
    print(f"   ✅ {fname:45s} ({fsize} KB)")

print(f"\n💾 Model files:")
print(f"   ✅ best_rice_model.keras")
print(f"   ✅ rice_disease_final_model.keras")
print(f"   ✅ class_names.json")
print(f"\n   Open the results/ folder to view all your plots!")