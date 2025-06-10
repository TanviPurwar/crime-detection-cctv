# Crime Detection Through CCTV Surveillance

Made for Bachelor's Minor Project 
A deep learning-based video classification system to detect criminal activities using CCTV footage. Built using the UCF Crime dataset.

## 📌 Project Overview

- Classifies five types of activity: **abuse, arson, burglary, shoplifting**, and **normal** behavior.
- Combines **CNN-based feature extraction** with a **bi-directional LSTM** for sequence modeling.
- Compared performance using 7–8 popular pretrained CNN models.
- Achieved **77.5% accuracy** on the test set.

## 🧠 Architecture

- **Frame Extraction**: Processed video using `FFmpeg` and `OpenCV`.
- **Feature Extraction**: Evaluated multiple pretrained CNNs (Xception, VGG16, VGG19, InceptionV3, DenseNet121, DenseNet201, ResNet50, ResNet152).
- **Classification**: Used a Bi-LSTM layer with temporal sequence input.

## 🛠️ Tech Stack

- Python, TensorFlow, OpenCV, FFmpeg, Scikit-learn
- NumPy, pandas, Matplotlib for visualization

## 📁 Dataset

- UCF-Crime dataset (https://www.crcv.ucf.edu/projects/real-world/)
- Includes 13 real-world crime categories
