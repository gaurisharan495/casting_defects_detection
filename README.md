# Casting Defect Detection using CNN

An end-to-end deep learning pipeline for automated industrial 
defect detection on casting product images, built as part of 
Manufacturing Technology I at IIT Guwahati.

---

## What It Does

- Classifies casting images as Defective or OK using a custom CNN
- Provides Grad-CAM heatmap overlays highlighting defect-prone 
  regions for quality engineer interpretability
- Converts prediction confidence into actionable severity levels:
  PASS / LOW / MODERATE / CRITICAL, each mapped to a production 
  recommendation (accept, monitor, inspect, reject)

---

## Model Architecture

Custom CNN built in TensorFlow/Keras:
- Multiple convolutional blocks with Conv2D, Batch Normalisation,
  and MaxPooling layers
- Fully connected classification head
- Binary output: Defective / OK

---

## Dataset

Kaggle — Real-Life Industrial Dataset of Casting Products  
https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

- Grayscale images resized to 300×300 pixels
- Augmentation: rotation, flipping, zooming, brightness 
  adjustment, shearing
- 80/20 train-validation split

---

## Training

- Up to 25 epochs
- Early Stopping, Model Checkpointing, Learning Rate Reduction 
  on Plateau
- Best model saved automatically as `.h5`

---

## Evaluation

Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix,
Precision-Recall Curves, Training/Validation learning curves

---

## Explainability

Grad-CAM visualisations overlaid on input images to highlight 
regions influencing the defect classification decision.

---

## How To Run

```bash
pip install -r requirements.txt
# Open notebook
jupyter notebook casting_defect_detection.ipynb
```

Dataset must be downloaded from Kaggle and placed in the 
working directory before running.

---

## Tech Stack

Python, TensorFlow, Keras, OpenCV, NumPy, Matplotlib, 
Seaborn, Scikit-learn
