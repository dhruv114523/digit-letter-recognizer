import numpy as np
import os
import tensorflow as tf
import torchvision
import torch
from torchvision.transforms import functional as F
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading EMNIST-Balanced dataset...")

# Load data
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x.numpy())
])

train_dataset = torchvision.datasets.EMNIST(
    root='./data', 
    split='balanced', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset = torchvision.datasets.EMNIST(
    root='./data', 
    split='balanced', 
    train=False, 
    download=True, 
    transform=transform
)

X_train = np.array([train_dataset[i][0].squeeze() for i in range(len(train_dataset))])
Y_train = np.array([train_dataset[i][1] for i in range(len(train_dataset))])

X_test = np.array([test_dataset[i][0].squeeze() for i in range(len(test_dataset))])
Y_test = np.array([test_dataset[i][1] for i in range(len(test_dataset))])

# Apply EMNIST orientation correction: rotate -90° and horizontal flip
print("Applying EMNIST orientation correction...")
X_train_corrected = []
for img in X_train:
    img_tensor = torch.tensor(img).unsqueeze(0)  # Add channel dimension
    corrected = F.hflip(F.rotate(img_tensor, -90))
    X_train_corrected.append(corrected.squeeze().numpy())
X_train = np.array(X_train_corrected)

X_test_corrected = []
for img in X_test:
    img_tensor = torch.tensor(img).unsqueeze(0)  # Add channel dimension
    corrected = F.hflip(F.rotate(img_tensor, -90))
    X_test_corrected.append(corrected.squeeze().numpy())
X_test = np.array(X_test_corrected)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Image shape: {X_train.shape[1:]} (28x28 grayscale)")
print(f"Number of classes: {len(np.unique(Y_train))}")

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(47, activation='softmax')  # 47 classes for EMNIST-Balanced
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#training the model
history = model.fit(
    X_train, Y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

model.save('emnist_recognizer.keras')
print("Saved model")

# Evaluate the model
print("\n" + "="*60)
print("EVALUATING MODEL PERFORMANCE")
print("="*60)

loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Calculate top-5 accuracy manually
Y_pred_probs = model.predict(X_test, verbose=0)
top5_correct = 0
for i, true_label in enumerate(Y_test):
    top5_predictions = np.argsort(Y_pred_probs[i])[-5:]  # Get top 5 indices
    if true_label in top5_predictions:
        top5_correct += 1
top5_accuracy = top5_correct / len(Y_test)
print(f"Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")

# Generate predictions
print("\nGenerating predictions...")
Y_pred = Y_pred_probs  # We already have probabilities from above
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Detailed metrics
print("\n" + "="*60)
print("DETAILED PERFORMANCE METRICS")
print("="*60)

# Per-class metrics
precision = precision_score(Y_test, Y_pred_classes, average=None, zero_division=0)
recall = recall_score(Y_test, Y_pred_classes, average=None, zero_division=0)
f1 = f1_score(Y_test, Y_pred_classes, average=None, zero_division=0)

print("Per-Class Metrics (showing first 20 classes):")
print("-" * 50)
for class_idx in range(min(20, len(precision))):
    print(f"Class {class_idx:2d}: Precision={precision[class_idx]:.4f}, Recall={recall[class_idx]:.4f}, F1={f1[class_idx]:.4f}")

if len(precision) > 20:
    print(f"... and {len(precision) - 20} more classes")

# Overall metrics
print("\nOverall Metrics:")
print("-" * 30)
print(f"Macro Average Precision: {precision_score(Y_test, Y_pred_classes, average='macro', zero_division=0):.4f}")
print(f"Macro Average Recall: {recall_score(Y_test, Y_pred_classes, average='macro', zero_division=0):.4f}")
print(f"Macro Average F1-Score: {f1_score(Y_test, Y_pred_classes, average='macro', zero_division=0):.4f}")
print(f"Weighted Average Precision: {precision_score(Y_test, Y_pred_classes, average='weighted', zero_division=0):.4f}")
print(f"Weighted Average Recall: {recall_score(Y_test, Y_pred_classes, average='weighted', zero_division=0):.4f}")
print(f"Weighted Average F1-Score: {f1_score(Y_test, Y_pred_classes, average='weighted', zero_division=0):.4f}")

# Error analysis
total_errors = np.sum(Y_test != Y_pred_classes)
error_rate = total_errors / len(Y_test)
print(f"\nError Analysis:")
print(f"Total Errors: {total_errors:,}")
print(f"Error Rate: {error_rate:.4f} ({error_rate*100:.2f}%)")

# Find most confused classes
print("\nTop 10 Most Confused Class Pairs:")
print("-" * 40)
cm = confusion_matrix(Y_test, Y_pred_classes)
# Get off-diagonal elements (misclassifications)
confusion_pairs = []
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        if i != j and cm[i, j] > 0:
            confusion_pairs.append((i, j, cm[i, j]))

# Sort by confusion count
confusion_pairs.sort(key=lambda x: x[2], reverse=True)
for i, (true_class, pred_class, count) in enumerate(confusion_pairs[:10]):
    print(f"{i+1:2d}. True: {true_class:2d} → Predicted: {pred_class:2d} ({count:4d} times)")

print("\n" + "="*60)
print("EMNIST TRAINING COMPLETE!")
print("="*60)
print(f"Final Accuracy: {accuracy*100:.2f}%")
print("="*60)
