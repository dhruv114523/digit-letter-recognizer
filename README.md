# Digit + Letter Recognizer

## Purpose

This repository trains and evaluates a neural network that recognizes handwritten
digits, uppercase letters, and selected lowercase letters. It uses the
47-class EMNIST-Balanced dataset to provide a compact OCR starting point for
experiments and document-processing applications.

## Frameworks

The project uses both PyTorch and TensorFlow for practical reasons. TensorFlow
is used for the model, training, and evaluation, while PyTorch's
`torchvision.datasets.EMNIST` is used to download and load the data. TensorFlow
data loading caused problems in this project, so the data pipeline was switched
to PyTorch while the rest of the workflow remained in TensorFlow.

## Results

- Test accuracy: **85.38%**
- Top-5 accuracy: **98.99%**
- Classes: **47** (10 digits, 26 uppercase letters, 11 lowercase letters)
- Training samples: **112,800**
- Test samples: **18,800**
- Error rate: **14.62%**
- Total errors: **2,749**
- Macro precision / recall / F1: **85.84% / 85.38% / 85.12%**
- Weighted precision / recall / F1: **85.84% / 85.38% / 85.12%**

### Most common confusions

| Rank | Actual | Predicted | Errors |
|---:|:---:|:---:|---:|
| 1 | F | f | 229 |
| 2 | O | 0 | 166 |
| 3 | L | 1 | 148 |
| 4 | q | 9 | 144 |
| 5 | I | 1 | 108 |
| 6 | g | 9 | 70 |
| 7 | 0 | O | 63 |
| 8 | 1 | L | 58 |
| 9 | L | I | 46 |
| 10 | 5 | S | 45 |

The main sources of error are visually similar characters, especially `O/0`,
`I/l/1`, `L/1`, `F/f`, `q/9`, `g/9`, and `5/S`.

## Model

The TensorFlow model uses flattened 28x28 grayscale images and three dense
hidden layers:

```text
784 -> 512 -> 256 -> 128 -> 47
```

ReLU activations, dropout (30%, 30%, and 20%), Adam, and sparse categorical
cross-entropy are used. The model has approximately 500,000 trainable
parameters.

## Quick start

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run training and evaluation:

```bash
python main.py
```

The script downloads EMNIST-Balanced, corrects image orientation, trains for
20 epochs with a batch size of 128 and a 10% validation split, evaluates the
model, and saves it as `emnist_recognizer.keras`.

## Dataset and preprocessing

- Dataset: EMNIST-Balanced
- Image format: 28x28 grayscale
- Pixel normalization: values scaled to `[0, 1]`
- Orientation correction: -90-degree rotation followed by a horizontal flip
- Training optimizer: Adam
- Loss: sparse categorical cross-entropy

## Repository

```text
main.py
requirements.txt
README.md
```
