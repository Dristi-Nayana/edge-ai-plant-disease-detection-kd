# Edge-AI Powered Plant Disease Detection using Knowledge Distillation

This project presents a lightweight and deployable Edge-AI system for plant disease detection using Knowledge Distillation and real-time inference.

## 📄 Project Report
📥 [View Full Thesis](./bachelors_thesis.pdf)

## 🚀 Overview
- Developed a high-accuracy plant disease detection system for edge devices
- Used Knowledge Distillation to compress deep learning models
- Achieved real-time inference using smartphone + laptop setup

## 🧠 Methodology

### 🔹 Teacher Model
- MobileViT (high accuracy ~98.89%)

### 🔹 Student Model
- MobileNetV3 (lightweight, efficient)

### 🔹 Knowledge Distillation
- Logit-based distillation
- Combined loss: Cross-Entropy + KL Divergence
- Temperature scaling (T = 4)

### 🔹 Deployment
- Model converted to ONNX format
- Real-time inference using OpenCV + ONNX Runtime

## 📊 Results

- Accuracy: **98.69%**
- Model size reduced from **19.1 MB → 6.07 MB (3× compression)**
- Inference speed improved from **15.13 ms → 7.32 ms**
- Real-time performance: **20–25 FPS**

## 📂 Dataset
- PlantVillage Dataset (~54,000 images, 38 classes)
- Real-world dataset (CGIAR crop disease dataset)

## 🛠 Tech Stack
Python, PyTorch, OpenCV, ONNX Runtime, NumPy, Pandas

## 💻 Code
- Jupyter Notebook: `edge_ai_detection.ipynb`
- Python Script: `main.py`

## 📸 Sample Results

### Knowledge Distillation Pipeline
(Add Fig 3.1 or 3.2)

### Model Performance
(Add accuracy vs size graph)

### Real-Time Deployment
(Add smartphone + laptop setup image)

## 📦 Note
Dataset and trained models are not included due to size limitations.

## 🔗 Future Work
- Deploy on Raspberry Pi / Jetson Nano
- Multi-modal inputs (temperature, humidity)
- Mobile app integration
