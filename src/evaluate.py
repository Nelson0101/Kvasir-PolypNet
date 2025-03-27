import torch
from torch.utils.data import DataLoader
from models.model import CNN
from data.polyp_dataset import PolypDataset
from data.data_preprocessor import DataPreprocessor
from train import evaluate
from torchvision import transforms
import argparse


def main(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((676, 650)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.1),
        ]
    )

    preprocessor = DataPreprocessor("../results/models/best_model.pth")
    val_dataset = preprocessor.create_dataset("val", transform)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = CNN(num_classes=2, num_filters=48).to(device)
    model.load_state_dict(torch.load(model_path))
    criterion = torch.nn.CrossEntropyLoss()

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to saved model")
    args = parser.parse_args()
    main(args.model_path)
