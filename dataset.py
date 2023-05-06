import os
import torch
import csv
from zipfile import ZipFile
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class FaceDataset(Dataset):
    def __init__(self):
        self._directory = 'dataset'
        if not os.path.isdir('dataset'):
            with ZipFile('dataset.zip', 'r') as zip:
                zip.extractall('dataset')

        self._entries = []
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])

        for row in csv.reader(open(os.path.join(self._directory, 'train_vector.csv'))):
            self._entries.append(row)

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, index):
        image_path = os.path.join(self._directory, self._entries[index][0])
        image = Image.open(image_path)
        vector = self._entries[index][1:]
        vector = [int(char) for string in vector for char in string]
        image = self._transform(image)

        return image, torch.tensor(vector, dtype=torch.float32)

