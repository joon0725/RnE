import Keypoints
import numpy as np

label = np.array([i for i in range(1) for _ in range(5)])
cnt = 0
r = ['D', 'F', 'L', 'R', 'U']
li = np.array([])
for k in range(1, 2):
    for i in r:
        cnt += 1
        a, b, c = map(np.array, Keypoints.getkey_from_vid(f'./원시데이터/NIA_SL_SEN000{k}_REAL01_{i}.mp4'))
        print(f"Data Loading... {cnt}/{5}")
        data = np.concatenate([a, b, c], axis=1)
        data = np.array([data[-10:-5][:]])
        if i == 'D' and k == 1:
            li = data
        else:
            li = np.vstack([li, data])
        print(li.shape)
li = np.array(li)

