import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# === Paths ===
BASE_DIR = "data"
test_dir = os.path.join(BASE_DIR, "test")
MODEL_PATH = "vgg16_idc.h5"

# === Parameters ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 64

# === Load test data ===
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# === Load trained model ===
model = load_model(MODEL_PATH)

# === Evaluate on test set ===
loss, acc = model.evaluate(test_gen)
print(f"✅ Test Loss: {loss:.4f}")
print(f"✅ Test Accuracy: {acc:.4f}")

# === Predictions ===
y_pred_probs = model.predict(test_gen)  # probabilities
y_pred_classes = (y_pred_probs > 0.5).astype("int32").ravel()
y_true = test_gen.classes

# === Classification Report ===
report = classification_report(
    y_true,
    y_pred_classes,
    target_names=["no_cancer", "cancer"],
    output_dict=True
)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv("classification_report.csv", index=True)
print("✅ Classification report saved as classification_report.csv")

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred_classes)
cm_df = pd.DataFrame(cm,
                     index=["no_cancer", "cancer"],
                     columns=["Pred_no_cancer", "Pred_cancer"])
cm_df.to_csv("confusion_matrix.csv")
print("✅ Confusion matrix saved as confusion_matrix.csv")

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["no_cancer", "cancer"],
            yticklabels=["no_cancer", "cancer"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# === ROC Curve & AUC ===
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.close()

print(f"✅ ROC AUC Score: {roc_auc:.4f}")

# === Save metrics to JSON ===
metrics = {
    "loss": float(loss),
    "accuracy": float(acc),
    "roc_auc": float(roc_auc),
    "precision": report["cancer"]["precision"],
    "recall": report["cancer"]["recall"],
    "f1_score": report["cancer"]["f1-score"]
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Metrics saved as metrics.json")
print("All evaluation files saved: metrics.json, classification_report.csv, confusion_matrix.csv, confusion_matrix.png, roc_curve.png")
