import torch
import kagglehub
from torch.utils.data import DataLoader
from models.model import CNN
from data.polyp_dataset import PolypDataset
from data.data_preprocessor import DataPreprocessor
from train import evaluate
from torchvision import transforms
import argparse
import os


def main(model_path, data_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1]),
        ]
    )

    # If no data_path provided, use kagglehub default like in main.py
    if data_path is None:
        path = kagglehub.dataset_download("heartzhacker/n-clahe")
        data_path = os.path.join(path, "dataset", "n-clahe")

    # Check if data_path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset path not found: {data_path}. Please specify a valid --data_path or ensure kagglehub download works."
        )

    # Check if model_path exists
    model_path = os.path.abspath(model_path)  # Convert to absolute path for clarity
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    preprocessor = DataPreprocessor(data_path)
    val_dataset = preprocessor.create_dataset("val", transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = CNN(num_classes=2, num_filters=32, in_channels=3).to(device)
    model.load_state_dict(torch.load(model_path))
    criterion = torch.nn.CrossEntropyLoss()

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained polyp classification model."
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to the saved model (.pth file)"
    )
    parser.add_argument(
        "--data_path",
        default=None,
        help="Path to the dataset folder (optional, defaults to kagglehub download)",
    )
    args = parser.parse_args()
    main(args.model_path, args.data_path)
