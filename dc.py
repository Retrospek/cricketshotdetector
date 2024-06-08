import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic  # Holistic Model
mp_drawing = mp.solutions.drawing_utils  # Drawing Utilities

##
##      HELPER METHODS
##

def mediapipe_detection(image, model):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # BGR to RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Detecting
    image.flags.writeable = True  # Image is now writeable
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)  # RGB to BGR
    return image, results

def draw_landmarks(image, results): # Draw Landmarks with styling
    # Draw Face Connections
    #if results.face_landmarks:
    #    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    
    # Draw Pose Connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=3, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=3, circle_radius=2))

def extract_keypoints(results):
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)
    return pose

# Basic Data properties
DATA_PATH = os.path.join('cricket_data')
actions = np.array(['drive', 'pull shot', 'ready', 'not ready', 'cut shot'])
no_sequences = 30
sequence_length = 30


cap = cv.VideoCapture(1)

cv.namedWindow("FullScreen", cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty("FullScreen", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

# Access MediaPipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # Loop through actions
    for action in actions:
        # Loop through sequences or videos
        for sequence in range(no_sequences):
            # Loop through video length or number of frames per video
            for frame_num in range(sequence_length):

                # Read frame
                ret, frame = cap.read()
                # Make Detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw the Landmarks
                draw_landmarks(image, results)
                
                # Collection needs some waiting period because I'm no the flash when moving
                if frame_num == 0:
                    cv.putText(image, 'STARTING COLLECTION', (120, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv.LINE_AA)
                    cv.putText(image, 'COLLECTING FRAMES FOR {} VIDEO NUMBER {}'.format(action, sequence), (15, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)  # Moved y-coordinate down
                    cv.waitKey(2000)
                else:
                    cv.putText(image, 'COLLECTING FRAMES FOR {} VIDEO NUMBER {}'.format(action, sequence), (15, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)  # Moved y-coordinate down
                
                #Key point extraction
                keypoints = extract_keypoints(results)
                npy_path =  os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Show to Screen
                cv.imshow("FullScreen", image)  # Changed to display the processed image

                # Breaking out of video feed
                if cv.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv.destroyAllWindows()
