import torch
import torch.nn as nn
from dataset import MedicalDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# =========================
# MODEL DEFINITION (same as train.py)
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# EVALUATION
# =========================
def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = MedicalDataset(
        csv_file="data/ABreast-Classification.csv",
        image_dir="data/images",
        mask_dir="data/masks"
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Load model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("best_model.pth", weights_only=False))
    model.eval()

    # Evaluate
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = (outputs > 0.5).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())

    # Results
    print("===== MODEL EVALUATION =====")
    print(f"Total samples: {len(dataset)}")
    print(f"Lesion present: {sum(all_labels)}")
    print(f"Lesion absent: {len(all_labels) - sum(all_labels)}")
    print()

    # Handle single class case
    unique_labels = np.unique(all_labels)
    if len(unique_labels) == 1:
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"All samples have the same class (Lesion Present = {unique_labels[0]})")
        print(f"Model Accuracy: {accuracy:.3f}")
        print("Note: Cannot compute precision/recall/F1 with only one class")
    else:
        print("Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=['No Lesion', 'Lesion']))

    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)


if __name__ == "__main__":
    evaluate_model()