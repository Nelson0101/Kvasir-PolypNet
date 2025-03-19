from torchvision import transforms
from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image  # Import PIL for image conversion


class PolypDataset(Dataset):
    def __init__(self, images, labels):
        if len(images) != len(labels):
            raise ValueError("images and labels must be the same length")

        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
                        transforms.Resize((676, 650)),  # Ensure consistent image size
                        transforms.ToTensor()  # Convert to tensor with shape [C, H, W]
                        ])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long).squeeze()

        # Convert NumPy array to PIL Image before applying transforms
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)  # Convert NumPy to PIL Image

        image = self.transform(image)  # Apply transformations
        label = torch.tensor(label, dtype=torch.long)

        return image, label
