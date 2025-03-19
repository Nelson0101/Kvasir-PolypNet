import torch
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, dataloader, optimizer, criterion, use_amp=False):
    model.train()
    scaler = GradScaler() if use_amp else None
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(dataloader), correct / total

