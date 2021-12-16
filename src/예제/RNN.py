import torch
import torch.nn as nn
import Keypoints
import numpy as np

a, b, c = map(torch.tensor, Keypoints.getkey_from_cam())
data = torch.cat([a, b, c], dim=1)
print(data.shape)
