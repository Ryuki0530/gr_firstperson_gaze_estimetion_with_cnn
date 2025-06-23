import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class GazeDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        self.image_dir = image_dir
        self.annotations = pd.read_csv(csv_path)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image_name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # ラベルを tensor で返す（float32, shape = [2]）
        label = torch.tensor([row['gaze_x'], row['gaze_y']], dtype=torch.float32)
        return image, label
