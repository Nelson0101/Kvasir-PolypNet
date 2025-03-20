import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import kagglehub
import cv2
from torch.utils.data import DataLoader
import torch
import torch.cuda as cuda
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import optuna
from data.PolypDataset import PolypDataset
from models.Model import CNN
from train import train_one_epoch, evaluate
from datetime import datetime

# Debug CUDA setup
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Running on CPU.")

path = kagglehub.dataset_download("heartzhacker/n-clahe")
main_path = path + "/dataset/n-clahe"


def load_data():
    train_images_polyp = []
    train_images_normal = []
    val_images_normal = []
    val_images_polyp = []

    print(f"Loading data from: {main_path}")
    for subdirectory in os.listdir(main_path):
        if subdirectory == ".DS_Store":
            continue
        is_val = subdirectory == "val"
        subdirectory_path = os.path.join(main_path, subdirectory)

        print(f"Scanning subdirectory: {subdirectory_path}")
        for sub_subdirectory in os.listdir(subdirectory_path):
            if sub_subdirectory == ".DS_Store":
                continue
            sub_subdirectory_path = os.path.join(subdirectory_path, sub_subdirectory)

            print(f"Processing: {sub_subdirectory_path}")
            for filename in os.listdir(sub_subdirectory_path):
                img_path = os.path.join(sub_subdirectory_path, filename)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load: {img_path}")
                    continue

                if sub_subdirectory == "polyps":
                    (val_images_polyp if is_val else train_images_polyp).append(img)
                elif sub_subdirectory == "normal-cecum":
                    (val_images_normal if is_val else train_images_normal).append(img)
                else:
                    print(f"Unexpected subdirectory: {sub_subdirectory}")

    train_images = train_images_polyp + train_images_normal
    val_images = val_images_polyp + val_images_normal
    train_labels = [1] * len(train_images_polyp) + [0] * len(train_images_normal)
    val_labels = [1] * len(val_images_polyp) + [0] * len(val_images_normal)

    print(f"Loaded - Train: {len(train_images)} images, Val: {len(val_images)} images")
    return train_images, train_labels, val_images, val_labels


def objective(trial):
    lr = trial.suggest_float("lr", 0.0015, 0.0020, log=True)
    batch_size = 8
    num_filters = 48
    num_epochs = trial.suggest_int("num_epochs", 9, 10)

    device = torch.device("cuda" if cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f"Trial {trial.number} - Using device: {device}, Params: {trial.params}")

    train_images, train_labels, val_images, val_labels = load_data()

    if len(train_images) == 0 or len(val_images) == 0:
        raise ValueError("No images loaded. Check dataset path and file integrity.")

    train_dataset = PolypDataset(train_images, train_labels)
    val_dataset = PolypDataset(val_images, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CNN(num_classes=2, num_filters=num_filters).to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_accuracy = 0.0
    log_file = f"trial_{trial.number}_log.txt"

    with open(log_file, "a") as f:
        f.write(f"Trial {trial.number} started at {datetime.now()}\n")
        f.write(f"Params: {trial.params}\n\n")

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, use_amp=True
        )
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

        trial.report(val_accuracy, epoch)

        if trial.should_prune():
            with open(log_file, "a") as f:
                f.write(f"Pruned at Epoch {epoch+1}\n")
            raise optuna.TrialPruned()

        best_val_accuracy = max(best_val_accuracy, val_accuracy)

        log_entry = (
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}\n"
        )
        print(log_entry)
        with open(log_file, "a") as f:
            f.write(log_entry)

    with open(log_file, "a") as f:
        f.write(
            f"Trial {trial.number} finished - Best Val Acc: {best_val_accuracy}\n\n"
        )

    return best_val_accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Summary log
    with open("optuna_summary.txt", "a") as f:
        f.write(f"Run completed at {datetime.now()}\n")
        f.write(
            f"Best Trial: {trial.number}, Value: {trial.value}, Params: {trial.params}\n"
        )
        for t in study.trials:
            f.write(f"Trial {t.number}: Value {t.value}, Params {t.params}\n")
        f.write("\n")
