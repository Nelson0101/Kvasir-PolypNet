import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from models.model import CNN
from train import evaluate
from data.data_preprocessor import DataPreprocessor
from torchvision import transforms

# Constants
MODEL_PATH = "../models/trial_{trial_number}_best.pth"
DATA_PATH = "path_to_downloaded_dataset/dataset/n-clahe"

transform = transforms.Compose(
    [
        transforms.Resize((676, 650)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.1),
    ]
)


def main(trial_number):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_preprocessor = DataPreprocessor(DATA_PATH)
    val_dataset = data_preprocessor.create_dataset("val", transform)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Load model
    model = CNN(num_classes=2, num_filters=48).to(device)
    model.load_state_dict(torch.load(MODEL_PATH.format(trial_number=trial_number)))
    model.eval()

    # Evaluate
    criterion = CrossEntropyLoss()
    val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <trial_number>")
        sys.exit(1)
    trial_number = sys.argv[1]
    main(trial_number)
