import Keypoints
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

label = torch.tensor([i for i in range(3) for _ in range(5)])
cnt = 0
r = ['D', 'F', 'L', 'R', 'U']
li = torch.tensor([])
for k in range(1, 4):
    for i in r:
        cnt += 1
        a, b, c = map(torch.tensor, Keypoints.getkey_from_vid(f'./원시데이터/NIA_SL_SEN000{k}_REAL01_{i}.mp4'))
        print(f"Data Loading... {cnt}/{15}")
        data = torch.cat([a, b, c], dim=1)
        data = data[-10:-5][:].unsqueeze(0)
        if k == 0 and i == 'D':
            li = data
        else:
            li = torch.cat([li, data], dim=0)


class MyData(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        return x, y


dataset = MyData(li, label)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

BATCH_SIZE = 64
lr = 0.001
EPOCHS = 10
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("다음 기기로 학습합니다:", DEVICE)


class BasicGRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, input_dim, n_classes, dropout=0.1):
        super(BasicGRU, self).__init__()
        print("준비중")
        self.n_layers = n_layers

        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(input_dim, self.hidden_dim, num_layers=self.n_layers)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        h_0 = self._init_state(batch_size=x.size(0))
        x, _ = self.gru(x, h_0)

        h_t = x[:, -1, :]

        self.dropout(h_t)

        logit = self.out(h_t)
        return logit


def train(model, optimizer, train_iter):
    model.train()
    corrects, total_loss = 0, 0
    for b, batch in enumerate(train_iter):
        x, y = batch.x.to(DEVICE), batch.y.to(DEVICE)
        optimizer.zero_grad()
        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(train_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy


model = BasicGRU(2, 256, 128, 10, 0.1).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=lr)

best_val_loss = None
for e in range(10):
    train_loss, train_accuracy = train(model, optim, dataloader)
    print("[이폭: %d] 검증 오차:%5.2f | 검증 정확도:%5.2f" % (e, train_loss, train_accuracy))
