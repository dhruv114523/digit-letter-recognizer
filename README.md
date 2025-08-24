# Digit + Letter Recognizer

A neural network-based character recognition system that can identify **47 different characters** including digits (0-9), uppercase letters (A-Z), and common symbols. Built with TensorFlow and trained on the EMNIST-Balanced dataset.

## 🎯 Project Overview

This project implements a deep learning solution for handwritten character recognition, achieving **85.38% accuracy** on a challenging 47-class classification problem. Unlike simple digit-only recognizers, this model can handle letters, numbers, and symbols, making it suitable for real-world document processing applications.

## 🏆 Performance Metrics

- **Test Accuracy**: 85.38%
- **Top-5 Accuracy**: 98.99%
- **Classes**: 47 (26 uppercase letters + 10 digits + 11 lowercase letters)
- **Training Samples**: 112,800
- **Test Samples**: 18,800
- **Error Rate**: 14.62%
- **Total Errors**: 2,749

### Detailed Performance Breakdown
- **Macro Average Precision**: 85.84%
- **Macro Average Recall**: 85.38%
- **Macro Average F1-Score**: 85.12%
- **Weighted Average Precision**: 85.84%
- **Weighted Average Recall**: 85.38%
- **Weighted Average F1-Score**: 85.12%

## 🧠 Model Architecture

```
Input Layer:        784 neurons (28×28 flattened)
Hidden Layer 1:     512 neurons + ReLU + 30% Dropout
Hidden Layer 2:     256 neurons + ReLU + 30% Dropout  
Hidden Layer 3:     128 neurons + ReLU + 20% Dropout
Output Layer:       47 neurons + Softmax
```

**Total Parameters**: ~500,000+ trainable parameters  
**Optimizer**: Adam  
**Loss Function**: Sparse Categorical Crossentropy  

## 📊 Key Features

- **Multi-class Recognition**: Handles 47 different character types
- **Robust Architecture**: Dropout layers prevent overfitting
- **Comprehensive Evaluation**: Precision, recall, F1-score analysis
- **Production Ready**: Saved model files for deployment
- **Cross-platform**: PyTorch data loading + TensorFlow training

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow torch torchvision scikit-learn matplotlib seaborn numpy
```

### Training the Model

```bash
python main.py
```

The script will:
1. Download the EMNIST-Balanced dataset (automatically)
2. Train the neural network for 20 epochs
3. Evaluate performance on test set
4. Save trained model as `emnist_recognizer.keras`

### Expected Output

```
Training samples: 112800
Test samples: 18800
Image shape: (28, 28) (28x28 grayscale)
Number of classes: 47

Final Accuracy: 85.38%
Top-5 Accuracy: 98.99%
```

## 📁 Project Structure

```
digit_recognizer/
├── main.py                    # Main training script
├── README.md                  # Project documentation
├── .gitignore                # Git ignore rules
└── requirements.txt          # Python dependencies
```

*Note: Model files (`*.keras`, `*.h5`) and dataset (`data/`) are excluded from Git due to size.*

## 🎮 Usage Examples

### Loading the Trained Model

```python
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('emnist_recognizer.keras')

# Prepare your 28x28 grayscale image
image = your_preprocessed_image.reshape(1, 28, 28)

# Make prediction
prediction = model.predict(image)
predicted_class = np.argmax(prediction[0])

print(f"Predicted class: {predicted_class}")
```

## 📈 Performance Analysis

### Class Distribution
The model performs well across all character types:
- **Digits (0-9)**: Good accuracy with some confusion between similar shapes
- **Uppercase Letters (A-Z)**: Strong performance except for characters similar to digits/lowercase
- **Lowercase Letters (a,b,d,e,f,g,h,n,q,r,t)**: Reasonable accuracy given visual similarities

### Top 10 Most Confused Character Pairs

| Rank | True Character | Predicted Character | Error Count | Issue Type |
|------|----------------|-------------------|-------------|------------|
| 1    | F              | f                 | 229         | Case confusion (orientation artifact) |
| 2    | O              | 0                 | 166         | Classic OCR problem |
| 3    | L              | 1                 | 148         | Classic OCR problem |
| 4    | q              | 9                 | 144         | Shape similarity |
| 5    | I              | 1                 | 108         | Classic OCR problem |
| 6    | g              | 9                 | 70          | Shape similarity |
| 7    | 0              | O                 | 63          | Classic OCR problem (reverse) |
| 8    | 1              | L                 | 58          | Classic OCR problem (reverse) |
| 9    | L              | I                 | 46          | Similar vertical shapes |
| 10   | 5              | S                 | 45          | Curved shape similarity |

### Common Confusions
- **O vs 0** (letter O vs digit zero) - bidirectional confusion
- **I vs l vs 1** (uppercase I vs lowercase l vs digit one)
- **L vs 1** (uppercase L vs digit one) - bidirectional confusion
- **F vs f** (case confusion caused by orientation correction)
- **Shape similarities**: q/9, g/9, 5/S

## 🔄 Future Enhancements

- **Web Interface**: Upload images for text extraction
- **API Development**: REST endpoints for integration
- **Mobile App**: Camera-based character recognition
- **Word Segmentation**: Full document text extraction
- **Model Optimization**: Quantization for edge deployment

## 🛠️ Technical Details

### Dataset
- **EMNIST-Balanced**: Extended MNIST with letters and symbols
- **Image Size**: 28×28 pixels, grayscale
- **Normalization**: Pixel values scaled to [0, 1]
- **Orientation Correction**: -90° rotation + horizontal flip applied
- **Character Set**: 10 digits + 26 uppercase + 11 lowercase letters

### Training Configuration
- **Epochs**: 20
- **Batch Size**: 128
- **Validation Split**: 10%
- **Early Stopping**: Not implemented (stable convergence observed)
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy

### Evaluation Metrics
- **Accuracy**: Overall correct predictions (85.38%)
- **Top-5 Accuracy**: Model's top 5 predictions include correct answer (98.99%)
- **Precision/Recall**: Per-class performance analysis
- **F1-Score**: Balanced performance measure
- **Confusion Matrix**: Detailed error analysis showing character pair confusions
- **Error Rate**: 14.62% with 2,749 misclassifications out of 18,800 samples

## 📊 Comparison with Baselines

| Model Type | Classes | Accuracy | Notes |
|------------|---------|----------|--------|
| Basic MNIST | 10 digits | ~97% | Simple digit recognition |
| **This Project** | **47 chars** | **85.38%** | **Full character recognition** |
| Human Performance | 47 chars | ~95% | Estimated for comparison |

## 🤝 Contributing

This is a personal learning project, but suggestions and improvements are welcome! Feel free to:
- Report issues with the training process
- Suggest architectural improvements
- Share ideas for new features

## 📄 License

This project is open source and available under the MIT License.

## 🎓 Learning Outcomes

Building this project demonstrates:
- **Deep Learning**: Neural network architecture design
- **Data Engineering**: Large dataset handling and preprocessing
- **Model Evaluation**: Comprehensive performance analysis
- **Framework Integration**: PyTorch + TensorFlow workflow
- **Production Considerations**: Model saving and deployment preparation

---

*Built with ❤️ for learning and exploring the fascinating world of computer vision and machine learning.*
