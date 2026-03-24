
# Face Mask Detection with Deep Learning

This project is a deep learning-based image classification system that detects whether a person is wearing a face mask or not. The implementation includes both a **custom Convolutional Neural Network (CNN)** and a **Transfer Learning approach using MobileNetV2**.

---

## Dataset

The dataset is obtained from Kaggle and consists of two classes:
- `with_mask`
- `without_mask`

It is loaded using `opendatasets` and split into training and validation sets.

---

## Model Architectures

### 1. Custom CNN
- Built from scratch using Conv2D, MaxPooling, Dropout, and Dense layers
- Designed to demonstrate understanding of CNN fundamentals

### 2. MobileNetV2 (Transfer Learning)
- Pre-trained on ImageNet
- Only the top classifier is trained
- Offers high accuracy with less training time

---

## Data Augmentation

Data augmentation is applied to improve model generalization:
- Rotation
- Zoom
- Shear
- Horizontal Flip
- Rescaling (1./255)

A sample of augmented images is visualized.

---

## Evaluation Metrics

Each model is evaluated using:
- Accuracy/Loss plots
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

---

## Results Summary

| Model        | Accuracy | Comments                    |
|--------------|----------|-----------------------------|
| Custom CNN   | ~85-90%  | Good performance with tuning|
| MobileNetV2  | ~95%+    | Fast training, generalizes better |

---

## Requirements

- TensorFlow / Keras
- Matplotlib
- scikit-learn
- seaborn
- opendatasets

---

## Future Work

- Convert to real-time webcam detector using OpenCV
- Deploy with Streamlit/Gradio
- Expand dataset to include improper mask usage

---

## Credits

Dataset: [Kaggle - Face Mask Detection](https://www.kaggle.com/datasets)
