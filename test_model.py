import os
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# === Paths ===
MODEL_PATH = "model/densenet121_idc.h5"  # or 'densenet121_idc_final.h5'
TEST_DIR = "data/val"
IMG_SIZE = 224
BATCH_SIZE = 32

# === Load Model ===
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# === Data Generator ===
test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# === Predict ===
pred_probs = model.predict(test_data)
pred_labels = (pred_probs > 0.5).astype(int).flatten()
true_labels = test_data.classes
class_names = list(test_data.class_indices.keys())

# === Classification Report ===
report = classification_report(true_labels, pred_labels, target_names=class_names, output_dict=True)

# Save JSON report
with open("evaluation/classification_report.json", "w") as f:
    json.dump(report, f, indent=4)
print("📄 Saved: classification_report.json")

# === Confusion Matrix ===
cm = confusion_matrix(true_labels, pred_labels)
np.savetxt("evaluation/confusion_matrix.csv", cm, delimiter=",", fmt="%d")
print("📄 Saved: confusion_matrix.csv")

# === ROC Curve ===
fpr, tpr, _ = roc_curve(true_labels, pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.savefig("roc_curve.png")
plt.close()
print("📊 Saved: roc_curve.png")

print("\n✅ Evaluation Complete.")
