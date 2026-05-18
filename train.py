"""Dr. TABA CHABI - Entraînement du modèle de classification d'images médicales (Pocus)
train.py - Basic Training Loop for Image Classification with PyTorch"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from dataset import MedicalDataset
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
# VALIDATION FUNCTION
# =========================
def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Convert to binary predictions
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return val_loss / len(val_loader), accuracy, precision, recall, f1


# =========================
# DATA
# =========================
dataset = MedicalDataset(
    csv_file="data/ABreast-Classification.csv",
    image_dir="data/images",
    mask_dir="data/masks"
)

# Split dataset: 80% train, 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


# =========================
# TRAIN SETUP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# =========================
# TRAIN LOOP
# =========================
print("===== TRAINING CLASSIFICATION =====")
print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

best_f1 = 0.0
best_model_path = "best_model.pth"

for epoch in range(10):
    # Training phase
    model.train()
    total_train_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation phase
    val_loss, accuracy, precision, recall, f1 = validate_model(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"Acc: {accuracy:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f} | F1: {f1:.3f}")

    # Save best model based on F1 score
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), best_model_path)
        print(f"  → Saved best model (F1: {f1:.3f})")

print(f"\nTraining complete! Best F1: {best_f1:.3f}")
print(f"Best model saved to: {best_model_path}")