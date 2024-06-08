from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import cv2 as cv
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard

# Basic Data properties
DATA_PATH = os.path.join('cricket_data')
actions = np.array(['drive', 'pull shot', 'ready', 'not ready', 'cut shot'])
no_sequences = 30
sequence_length = 30

# Creating encoding
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)  # shape = 150, 30, 132
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Actual Neural Network

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 132)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=230, callbacks=[tb_callback])
model.save('model.h5')
print("Saved the model")

loss, accuracy = model.evaluate(X_test, y_test)
print(accuracy)
