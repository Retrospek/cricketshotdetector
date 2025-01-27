Cricket Shot Action Detector 🏏
Overview
The Cricket Shot Action Detector is an accessible machine learning-based classifier designed to analyze cricket shots. It helps cricket players improve their batting techniques through data-driven insights.

Features
Built using TensorFlow with a custom LSTM model for temporal predictions.
Achieved:
96.49% testing accuracy
0.9472 F1 score
Classifies cricket shot actions using MediaPipe's landmark detection for analyzing limb vectors.
Enhancements
Multi-angle video recordings using OpenCV improved accuracy by 6.7%.
Dropout regularization helped prevent overfitting, boosting model robustness.
Delivered approximately a 25% increase in top-order batting scores.
Tools and Technologies
TensorFlow for model development.
OpenCV for video recording and pre-processing.
MediaPipe for landmark detection.
How to Use
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt  
Use OpenCV to record or upload cricket shot videos.
Process the videos with MediaPipe for limb landmark extraction.
Run the LSTM model for classification.
Acknowledgments
Special thanks to the Westlake Cricket Club for providing feedback and supporting this project.
