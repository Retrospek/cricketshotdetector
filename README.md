# 🏏 Cricket Shot Action Detector

## 📖 Overview
The **Cricket Shot Action Detector** is an accessible machine learning-based classifier designed to analyze cricket shots. It helps cricket players improve their batting techniques through data-driven insights.

---

## ✨ Features
- Built using **TensorFlow** with a custom **LSTM model** for temporal predictions.
- Achieved:
  - **96.49% testing accuracy**
  - **0.9472 F1 score**
- Classifies cricket shot actions using **MediaPipe's landmark detection** for analyzing limb vectors.

---

## 🚀 Enhancements
- **Multi-angle video recordings** using OpenCV improved accuracy by **6.7%**.
- Implemented **Dropout regularization** to prevent overfitting, boosting model robustness.
- Resulted in an approximate **25% increase in top-order batting scores**.

---

## 🛠️ Tools and Technologies
- **TensorFlow** for model development.
- **OpenCV** for video recording and preprocessing.
- **MediaPipe** for landmark detection.

---

## 📝 How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cricket-shot-action-detector.git
   cd cricket-shot-action-detector
