from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import cv2 as cv
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.models import load_model

# Methods
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (100, 50, 100), (50, 100, 50)]

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv.putText(output_frame, actions[num], (0, 85 + num * 40), cv.FONT_HERSHEY_SIMPLEX, 1, (240, 230, 210), 2, cv.LINE_AA)
    return output_frame

def mediapipe_detection(image, model):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # BGR to RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Detecting
    image.flags.writeable = True  # Image is now writeable
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)  # RGB to BGR
    return image, results

def draw_landmarks(image, results):  # Draw Landmarks with styling
    # Draw Pose Connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=3, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=3, circle_radius=2))

def extract_keypoints(results):
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)
    return pose

# Load model
tf.keras.backend.clear_session()
model = load_model('model.h5')

# Compile the loaded model to ensure metrics are built
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Basic Data properties
DATA_PATH = os.path.join('cricket_data')
actions = np.array(['drive', 'pull shot', 'ready', 'not ready', 'cut shot'])
no_sequences = 30
sequence_length = 30

mp_holistic = mp.solutions.holistic  # Holistic Model
mp_drawing = mp.solutions.drawing_utils  # Drawing Utilities

sequence = []
threshold = 0.4

cap = cv.VideoCapture('Cricket_Pull.mp4')

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    drive = 0
    pull = 0
    cut = 0
    while cap.isOpened():
        
        # Read frame
        ret, frame = cap.read()

        # Check if frame is read correctly
        if not ret:
            print("Failed to grab frame")
            break

        # Make Detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw the Landmarks
        draw_landmarks(image, results)

        # Key point extraction
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            if actions[np.argmax(res)] == 'drive':
                drive +=1
            if actions[np.argmax(res)] == 'pull shot':
                pull +=1
            if actions[np.argmax(res)] == 'cut shot':
                cut +=1
            
            cv.putText(image, "drive: {}, pull: {}, cut: {}".format(drive, pull, cut), (3, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
            image = prob_viz(res, actions, image, colors)
            

        # Show to Screen
        cv.imshow("Feed", image)  # Changed to display the processed image

        # Breaking out of video feed
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
