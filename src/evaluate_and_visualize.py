import torch
import kagglehub
from torch.utils.data import DataLoader
from models.model import CNN
from data.data_preprocessor import DataPreprocessor
from train import evaluate
from torchvision import transforms
import argparse
import os

# === 🔬 Added ===
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_fscore_support,
)
import torch.nn.functional as F


def plot_confusion_matrix(
    y_true, y_pred, class_names, save_path="results/confusion_matrix.png"
):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_probs, save_path="results/roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_class_metrics(
    y_true, y_pred, class_names, save_path="results/class_metrics.png"
):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    metrics = {"Precision": precision, "Recall": recall, "F1-Score": f1}
    x = np.arange(len(class_names))
    width = 0.25
    plt.figure(figsize=(8, 6))
    for i, (metric, values) in enumerate(metrics.items()):
        plt.bar(x + i * width, values, width=width, label=metric)
    plt.xticks(x + width, class_names)
    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.title("Per-Class Evaluation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def show_misclassified(
    images, labels, preds, class_names, save_dir="results/misclassified"
):
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    for idx, (img_tensor, true, pred) in enumerate(zip(images, labels, preds)):
        if true != pred and count < 12:
            img = img_tensor.permute(1, 2, 0).numpy()
            img = (img * 0.1 + 0.5).clip(0, 1)
            plt.imshow(img)
            plt.title(f"True: {class_names[true]}, Pred: {class_names[pred]}")
            plt.axis("off")
            plt.savefig(f"{save_dir}/wrong_{idx}.png")
            plt.close()
            count += 1


def generate_gradcam(
    model, input_tensor, class_idx, device, save_path="results/gradcam_sample.png"
):
    model.eval()
    input_tensor = input_tensor.unsqueeze(0).to(device)

    def hook_fn(module, input, output):
        model.gradients = output

    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    handle = last_conv.register_forward_hook(hook_fn)
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    score = output[0, class_idx]
    model.zero_grad()
    score.backward()
    gradients = model.gradients.grad[0].cpu().numpy()
    activation = model.gradients[0].cpu().numpy()
    weights = np.mean(gradients, axis=(1, 2))
    cam = np.zeros(activation.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activation[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= cam.max()
    img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * 0.1 + 0.5).clip(0, 1)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    superimposed = heatmap * 0.4 + img
    plt.imshow(superimposed)
    plt.title(f"Grad-CAM for Class {class_idx}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    handle.remove()


def main(model_path, data_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1]),
        ]
    )
    if data_path is None:
        path = kagglehub.dataset_download("heartzhacker/n-clahe")
        data_path = os.path.join(path, "dataset", "n-clahe")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path not found: {data_path}")
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    preprocessor = DataPreprocessor(data_path)
    val_dataset = preprocessor.create_dataset("val", transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = CNN(num_classes=2, num_filters=32, in_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    y_true, y_pred, y_probs = [], [], []
    x_images = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            x_images.extend(inputs.cpu())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Polyp"]))

    os.makedirs("results", exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, class_names=["Normal", "Polyp"])
    plot_roc_curve(y_true, y_probs)
    plot_class_metrics(y_true, y_pred, class_names=["Normal", "Polyp"])
    show_misclassified(x_images, y_true, y_pred, class_names=["Normal", "Polyp"])
    print("\n🔬 Generating Grad-CAM for first validation sample...")
    generate_gradcam(model, x_images[0], y_true[0], device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize a trained polyp classification model."
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to the saved model (.pth file)"
    )
    parser.add_argument(
        "--data_path",
        default=None,
        help="Path to dataset (optional, uses kagglehub if not provided)",
    )
    args = parser.parse_args()
    main(args.model_path, args.data_path)
