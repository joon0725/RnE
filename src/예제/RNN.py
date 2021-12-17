import Keypoints
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 64
lr = 0.001
EPOCHS = 10
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("다음 기기로 학습합니다:", DEVICE)


label = torch.tensor([i for i in range(10) for _ in range(5)])
cnt = 0
r = ['D', 'F', 'L', 'R', 'U']
dataset = torch.tensor([])
for k in range(10):
    for i in r:
        cnt += 1
        a, b, c = map(torch.tensor, Keypoints.getkey_from_vid(f'./원시데이터/NIA_SL_SEN000{k}_REAL01_{i}.mp4'))
        print(f"Data Loading... {cnt}/{k*5}")
        data = torch.cat([a, b, c], dim=1)
        data = data[20:30][:].unsqueeze(0)
        dataset = torch.cat([dataset, data], dim=0)
