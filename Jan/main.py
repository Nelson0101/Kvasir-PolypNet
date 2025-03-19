import os
import numpy as np
import kagglehub
import cv2
from torch.utils.data import DataLoader
import torch
from torch import device, cuda
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import optuna
from dataset import PolypDataset
from model import CNN
from train import train_one_epoch, evaluate

# Data loading
path = kagglehub.dataset_download("heartzhacker/n-clahe")
main_path = path + '/dataset/n-clahe'

def load_data():
    train_images_polyp = []
    train_images_normal = []
    val_images_normal = []
    val_images_polyp = []
    
    for subdirectory in os.listdir(main_path):
        if subdirectory == '.DS_Store':
            continue
        if subdirectory == 'val':
            out_dir_polyp = val_images_polyp
            out_dir_normal = val_images_normal
        else:
            out_dir_polyp = train_images_polyp
            out_dir_normal = train_images_normal
            
        subdirectory_path = os.path.join(main_path, subdirectory)
        for sub_subdirectory in os.listdir(subdirectory_path):
            if sub_subdirectory == '.DS_Store':
                continue
            sub_subdirectory_path = os.path.join(subdirectory_path, sub_subdirectory)
            for filename in os.listdir(sub_subdirectory_path):
                img_path = os.path.join(sub_subdirectory_path, filename)
                img = cv2.imread(img_path)
                if img is not None and sub_subdirectory == 'polyps':
                    out_dir_polyp.append(img)
                if img is not None and sub_subdirectory == 'normal-cecum':
                    out_dir_normal.append(img)
    
    # Prepare datasets
    train_images = train_images_polyp + train_images_normal
    val_images = val_images_polyp + val_images_normal
    train_labels = [1] * len(train_images_polyp) + [0] * len(train_images_normal)
    val_labels = [1] * len(val_images_polyp) + [0] * len(val_images_normal)
    
    return train_images, train_labels, val_images, val_labels

def objective(trial):
    # Hyperparameters to optimize
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    num_filters = trial.suggest_categorical("num_filters", [32, 64, 128])
    num_epochs = trial.suggest_int("num_epochs", 5, 15)
    
    # Load data
    train_images, train_labels, val_images, val_labels = load_data()
    
    # Create datasets and dataloaders
    train_dataset = PolypDataset(train_images, train_labels)
    val_dataset = PolypDataset(val_images, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup model
    device = device("cuda" if cuda.is_available() else "cpu")
    model = CNN(num_classes=2, num_filters=num_filters).to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, 
                                                   optimizer, criterion, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        
        trial.report(val_accuracy, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
            
        best_val_accuracy = max(best_val_accuracy, val_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    return best_val_accuracy

if __name__ == "__main__":
    # Create and run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    
    # Print results
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")