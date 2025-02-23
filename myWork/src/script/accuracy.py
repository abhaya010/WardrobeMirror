'''import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Simulated ground truth and predictions (for 5 test images, 5 recommendations each = 25 total recommendations)
# 1 = similar (positive), 0 = not similar (negative)
true_labels = np.array([1, 1, 1, 0, 0,  # Image 1: 5 recommendations
                        1, 1, 0, 1, 0,  # Image 2: 5 recommendations
                        1, 1, 0, 0, 1,  # Image 3: 5 recommendations
                        0, 1, 1, 0, 0,  # Image 4: 5 recommendations
                        1, 0, 1, 0, 1])  # Image 5: 5 recommendations

# Simulated KNN distances (lower distance = more similar, normalized to probabilities)
knn_distances = np.array([0.1, 0.2, 0.3, 0.7, 0.9,  # Image 1: 5 distances
                          0.2, 0.5, 0.4, 0.6, 0.8,  # Image 2
                          0.1, 0.3, 0.5, 0.7, 0.9,  # Image 3
                          0.2, 0.4, 0.3, 0.6, 0.8,  # Image 4
                          0.1, 0.5, 0.4, 0.7, 0.9])  # Image 5

# Normalize distances to probabilities (0 to 1, higher = more similar)
max_distance = np.max(knn_distances)
similarity_scores = 1 - (knn_distances / max_distance)  # Invert and normalize

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(true_labels, similarity_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Fashion Recommendation System')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Print AUC
print(f"Area Under Curve (AUC): {roc_auc:.2f}")'''

import numpy as np

# Simulated ground truth and predictions (for 10 test images, 5 recommendations each = 50 total recommendations)
# 1 = similar (positive), 0 = not similar (negative)
true_labels = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 0,  # Image 1: 5 recommendations
                       1, 0, 1, 0, 1, 1, 0, 0, 1, 1,  # Image 2: 5 recommendations
                       1, 1, 0, 0, 1, 1, 0, 0, 1, 0,  # Image 3: 5 recommendations
                       0, 1, 1, 0, 0, 1, 1, 0, 0, 1,  # Image 4: 5 recommendations
                       1, 0, 1, 0, 1])                 # Image 5: 5 recommendations

predicted_labels = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1,  # System recommendations for Image 1
                           1, 0, 1, 0, 0, 1, 0, 0, 1, 0,  # Image 2
                           1, 1, 0, 0, 0, 1, 0, 0, 1, 1,  # Image 3
                           0, 1, 1, 0, 0, 1, 0, 0, 0, 1,  # Image 4
                           1, 0, 1, 0, 0])                 # Image 5

# Calculate Confusion Matrix components
TP = np.sum((true_labels == 1) & (predicted_labels == 1))  # True Positives
FP = np.sum((true_labels == 0) & (predicted_labels == 1))  # False Positives
FN = np.sum((true_labels == 1) & (predicted_labels == 0))  # False Negatives
TN = np.sum((true_labels == 0) & (predicted_labels == 0))  # True Negatives (less relevant here)

# Print Confusion Matrix
print("Confusion Matrix:")
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"True Negatives (TN): {TN}")

# Calculate Precision, Recall, and F1-Score
precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # Avoid division by zero
recall = TP / (TP + FN) if (TP + FN) > 0 else 0     # Avoid division by zero
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  # Avoid division by zero

# Print Results
print("\nEvaluation Metrics:")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-Score: {f1_score:.2%}")

# Calculate Accuracy (for completeness, though less relevant for imbalanced data)
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
print(f"Accuracy: {accuracy:.2%}")