import os
import numpy as np

DATA_PATH = os.path.join('cricket_data')
actions = np.array(['drive', 'pull shot', 'ready', 'not ready', 'cut shot'])
no_sequences = 30
sequence_length = 30

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass