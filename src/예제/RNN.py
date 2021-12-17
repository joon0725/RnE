import torch
import torch.nn as nn
import Keypoints

r = ['D', 'F', 'L', 'R', 'U']
dataset = torch.tensor([])
for i in r:
    a, b, c = map(torch.tensor, Keypoints.getkey_from_vid(f'./data/NIA_SL_SEN0001_REAL01_{i}.mp4'))
    data = torch.cat([a, b, c], dim=1)
    data = data[:35][:].unsqueeze(0)
    dataset = torch.cat([dataset, data], dim=0)
print(dataset.shape)
