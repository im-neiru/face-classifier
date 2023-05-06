import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=216, kernel_size=(4, 4), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=3, out_channels=180, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(
            in_channels=3, out_channels=144, kernel_size=(4, 4), stride=(2, 2))
        self.conv4 = nn.Conv2d(
            in_channels=3, out_channels=108, kernel_size=(4, 4), stride=(2, 2))
        self.conv5 = nn.Conv2d(
            in_channels=3, out_channels=72, kernel_size=(4, 4), stride=(2, 2))
        self.pool = nn.MaxPool2d(kernel_size=(5, 5), stride=(5, 5))
        self.flt = nn.Flatten();
        self.fc1 = nn.Linear(72, 36)
        self.fc2 = nn.Linear(36, 18)

    def forward(self, x):
        x = nn.functional.silu(self.conv1(x))
        x = nn.functional.silu(self.conv2(x))
        x = nn.functional.silu(self.conv3(x))
        x = nn.functional.silu(self.conv4(x))
        x = nn.functional.silu(self.conv5(x))
        x = self.pool(x)
        x = self.flt(x)
        x = nn.functional.silu(self.fc1(x))
        x = nn.functional.sigmoid(self.fc2(x))
        
        return x