import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=(2, 2), stride=(1, 1))
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(2, 2), stride=(1, 1))
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(1, 1))
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=(2, 2), stride=(1, 1))
        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(2, 2), stride=(1, 1))
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(2, 2), stride=(1, 1))
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=(2, 2), stride=(1, 1))
        self.conv9 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=(2, 2), stride=(1, 1))
        self.conv8 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=(2, 2), stride=(1, 1))
        self.conv10 = nn.Conv2d(
            in_channels=512, out_channels=1024, kernel_size=(2, 2), stride=(1, 1))
        self.conv11= nn.Conv2d(
            in_channels=1024, out_channels=1024, kernel_size=(2, 2), stride=(1, 1))
        self.conv12 = nn.Conv2d(
            in_channels=1024, out_channels=1024, kernel_size=(2, 2), stride=(1, 1))
        self.conv13 = nn.Conv2d(
            in_channels=1024, out_channels=2048, kernel_size=(2, 2), stride=(1, 1))
        self.conv14 = nn.Conv2d(
            in_channels=2048, out_channels=2048, kernel_size=(2, 2), stride=(2, 2))
        self.conv15 = nn.Conv2d(
            in_channels=2048, out_channels=4096, kernel_size=(2, 2), stride=(1, 1))
        self.conv16 = nn.Conv2d(
            in_channels=4096, out_channels=4096, kernel_size=(2, 2), stride=(1, 1))
        self.conv17 = nn.Conv2d(
            in_channels=4096, out_channels=4096, kernel_size=(2, 2), stride=(1, 1))
        self.conv18 = nn.Conv2d(
            in_channels=4096, out_channels=8192, kernel_size=(2, 2), stride=(1, 1))
        self.conv18 = nn.Conv2d(
            in_channels=8192, out_channels=8192, kernel_size=(4, 4), stride=(2,2))
        self.conv19 = nn.Conv2d(
            in_channels=8192, out_channels=8192, kernel_size=(4, 4), stride=(2,2))
        self.conv20 = nn.Conv2d(
            in_channels=8192, out_channels=8192, kernel_size=(4, 4), stride=(2,2))
        
        self.pool = nn.MaxPool2d(kernel_size=(11, 11), stride=(11, 11))
        self.flt = nn.Flatten();
        self.fc1 = nn.Linear(8192, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, 2048)
        self.fc5 = nn.Linear(2048, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 128)
        self.fc8 = nn.Linear(128, 128)
        self.fc9 = nn.Linear(128, 32)
        self.fc10 = nn.Linear(32, 18)

    def forward(self, x):
        x = nn.functional.mish(self.conv1(x))
        x = nn.functional.silu(self.conv2(x))
        x = nn.functional.silu(self.conv3(x))
        x = nn.functional.silu(self.conv4(x))
        x = nn.functional.silu(self.conv5(x))
        x = nn.functional.silu(self.conv6(x))
        x = nn.functional.silu(self.conv7(x))
        x = nn.functional.silu(self.conv8(x))
        x = nn.functional.silu(self.conv9(x))
        x = nn.functional.silu(self.conv10(x))
        x = nn.functional.silu(self.conv11(x))
        x = nn.functional.silu(self.conv12(x))
        x = nn.functional.silu(self.conv13(x))
        x = nn.functional.silu(self.conv14(x))
        x = nn.functional.silu(self.conv15(x))
        x = nn.functional.silu(self.conv16(x))
        x = nn.functional.silu(self.conv17(x))
        x = nn.functional.silu(self.conv18(x))
        x = nn.functional.silu(self.conv19(x))
        x = nn.functional.silu(self.conv20(x))
        
        x = self.pool(x)
        x = self.flt(x)
        
        x = nn.functional.silu(self.fc1(x))
        x = nn.functional.silu(self.fc2(x))
        x = nn.functional.silu(self.fc3(x))
        x = nn.functional.silu(self.fc4(x))
        x = nn.functional.silu(self.fc5(x))
        x = nn.functional.silu(self.fc6(x))
        x = nn.functional.silu(self.fc7(x))
        x = nn.functional.silu(self.fc8(x))
        x = nn.functional.silu(self.fc9(x))
        x = nn.functional.silu(self.fc10(x))
        
        return x
   
model = Classifier()

torch.save(model.state_dict(), 'ww.pt') 