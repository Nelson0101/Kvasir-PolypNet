import os
import kagglehub
from torch.utils.data import DataLoader
import torch
import torch.cuda as cuda
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import optuna
from models.model import CNN
from train import train_one_epoch, evaluate
from datetime import datetime
from data.data_preprocessor import DataPreprocessor
from torchvision import transforms

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ------- CONSTANTS -------
LOG_FOLDER = "../results/logs/"
MODEL_FOLDER = "../models/"
os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Debug CUDA setup
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Running on CPU.")

path = kagglehub.dataset_download("heartzhacker/n-clahe")
main_path = path + "/dataset/n-clahe"

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1]),
    ]
)


def objective(trial):
    lr = trial.suggest_float("lr", 0.0001, 0.01, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 64)
    num_epochs = trial.suggest_int("num_epochs", 5, 15)

    device = torch.device("cuda" if cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f"Trial {trial.number} - Using device: {device}, Params: {trial.params}")

    dataPreprocessor = DataPreprocessor(main_path)
    train_dataset = dataPreprocessor.create_dataset("train", transform)
    val_dataset = dataPreprocessor.create_dataset("val", transform)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("No images loaded. Check dataset path and file integrity.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CNN(num_classes=2, num_filters=32, in_channels=3).to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    best_val_accuracy = 0.0
    log_file = f"{LOG_FOLDER}trial_{trial.number}_log.txt"
    trial_model_path = f"{MODEL_FOLDER}trial_{trial.number}_best.pth"

    with open(log_file, "a") as f:
        f.write(f"Trial {trial.number} started at {datetime.now()}\n")
        f.write(f"Params: {trial.params}\n\n")

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, use_amp=True
        )
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        trial.report(val_accuracy, epoch)

        if epoch >= 3 and trial.should_prune():
            with open(log_file, "a") as f:
                f.write(f"Pruned at Epoch {epoch+1}\n")
            raise optuna.TrialPruned()

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), trial_model_path)
            print(f"Saved trial {trial.number} best model to {trial_model_path}")
            with open(log_file, "a") as f:
                f.write(
                    f"Saved best model at Epoch {epoch+1} with Val Acc: {val_accuracy:.4f} to {trial_model_path}\n"
                )

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
    print(f"  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save the overall best model
    best_model_path = f"{MODEL_FOLDER}best_model.pth"
    best_trial_model_path = f"{MODEL_FOLDER}trial_{trial.number}_best.pth"
    if os.path.exists(best_trial_model_path):
        model = CNN(num_classes=2, num_filters=32, in_channels=3)
        model.load_state_dict(torch.load(best_trial_model_path))
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved overall best model to {best_model_path}")
    else:
        print(f"Warning: Best trial model {best_trial_model_path} not found!")

    with open("optuna_summary.txt", "a") as f:
        f.write(f"Run completed at {datetime.now()}\n")
        f.write(
            f"Best Trial: {trial.number}, Value: {trial.value}, Params: {trial.params}\n"
        )
        f.write(f"Best model saved to: {best_model_path}\n")
        for t in study.trials:
            f.write(f"Trial {t.number}: Value {t.value}, Params {t.params}\n")
        f.write("\n")
