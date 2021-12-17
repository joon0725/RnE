import Keypoints
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

label = np.array([i for i in range(10) for _ in range(5)])
cnt = 0
r = ['D', 'F', 'L', 'R', 'U']
li = np.array([])
for k in range(1, 11):
    for i in r:
        cnt += 1
        a, b, c = map(np.array, Keypoints.getkey_from_vid(f'./원시데이터/NIA_SL_SEN000{k}_REAL01_{i}.mp4'))
        print(f"Data Loading... {cnt}/{50}")
        data = np.concatenate([a, b, c], axis=1)
        data = np.array([data[-10:-5][:]])
        if i == 'D' and k == 1:
            li = data
        else:
            li = np.vstack([li, data])
li = np.array(li)

model = keras.Sequential()
model.add(layers.GRU(256, return_sequences=True))
model.add(layers.SimpleRNN(128))
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(li, label, epochs=5)
