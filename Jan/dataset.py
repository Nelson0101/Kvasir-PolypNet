from torchvision import transforms
from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image

class PolypDataset(Dataset):
    def __init__(self, images, labels):
        if len(images) != len(labels):
            raise ValueError("images and labels must be the same length")

        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize((676, 650)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label