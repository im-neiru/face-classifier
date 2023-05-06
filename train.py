import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from dataset import FaceDataset

from model import Classifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = FaceDataset()
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

num_epochs = 10
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)

        loss = loss_fn(outputs, targets)

        print('Loss: ', loss)
        loss.backward()

        optimizer.step()

torch.save(model.state_dict(), 'classifier.pt')