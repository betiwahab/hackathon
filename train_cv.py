"""Dr. TABA CHABI - Entraînement et évaluation avancés du modèle de classification d'images médicales (Pocus)
train_cv.py - Cross-Validation Training with Patient-Level Stratification and Referral System Evaluation"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from collections import defaultdict

from dataset import MedicalDataset
from model_advanced import EfficientNetLesionClassifier, LesionReferralSystem


def get_patient_level_data(dataset):
    """
    Extract patient-level data for stratified CV
    Returns: patient_ids, patient_labels, patient_to_indices mapping
    """
    patient_to_indices = defaultdict(list)
    patient_labels = {}

    for idx in range(len(dataset)):
        _, label, patient_id, _ = dataset[idx]
        patient_to_indices[patient_id].append(idx)
        if patient_id not in patient_labels:
            patient_labels[patient_id] = label.item()

    patient_ids = list(patient_labels.keys())
    labels = [patient_labels[pid] for pid in patient_ids]

    return patient_ids, np.array(labels), patient_to_indices


def create_fold_datasets(dataset, train_patients, val_patients, patient_to_indices):
    """
    Create train/val datasets for a specific fold
    """
    train_indices = []
    val_indices = []

    for patient in train_patients:
        train_indices.extend(patient_to_indices[patient])

    for patient in val_patients:
        val_indices.extend(patient_to_indices[patient])

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


def train_one_fold(model, train_loader, val_loader, criterion, optimizer, device, fold, num_epochs=20):
    """
    Train model for one fold
    """
    best_auc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for images, labels, _, _ in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1} Train"):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_probs = []
        val_labels = []

        with torch.no_grad():
            for images, labels, _, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model.get_probabilities(images)
                loss = criterion(model(images), labels.unsqueeze(1))

                val_loss += loss.item()
                val_probs.extend(outputs.cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy())

        val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5

        print(f"Fold {fold+1} Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val AUC: {val_auc:.4f}")

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()

    return best_model_state, best_auc


def evaluate_fold(model, val_loader, referral_system, device, fold):
    """
    Evaluate model on validation set with referral system
    """
    model.eval()
    all_probs = []
    all_labels = []
    all_decisions = []

    with torch.no_grad():
        for images, labels, _, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            probs = model.get_probabilities(images)
            decisions, _ = referral_system.classify_and_refer(probs.cpu().numpy())

            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())
            all_decisions.extend(decisions)

    # Calculate metrics
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5

    # Referral statistics
    certain_predictions = [i for i, d in enumerate(all_decisions) if d != 2]
    uncertain_rate = 1 - len(certain_predictions) / len(all_decisions)

    if certain_predictions:
        certain_labels = [all_labels[i] for i in certain_predictions]
        certain_probs = [all_probs[i] for i in certain_predictions]
        certain_auc = roc_auc_score(certain_labels, certain_probs) if len(np.unique(certain_labels)) > 1 else 0.5
    else:
        certain_auc = 0.5

    return {
        'auc': auc,
        'uncertain_rate': uncertain_rate,
        'certain_auc': certain_auc,
        'total_samples': len(all_labels),
        'certain_samples': len(certain_predictions)
    }


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_folds = 5
    num_epochs = 20
    batch_size = 8
    learning_rate = 1e-4

    print("Loading dataset...")
    dataset = MedicalDataset(
        csv_file="data/ABreast-Classification.csv",
        image_dir="data/images",
        mask_dir="data/masks"
    )

    # Get patient-level data for stratified CV
    patient_ids, patient_labels, patient_to_indices = get_patient_level_data(dataset)

    print(f"Dataset: {len(dataset)} images from {len(patient_ids)} patients")
    print(f"Lesion prevalence: {np.mean(patient_labels):.2f}")

    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_results = []
    fold_models = []

    for fold, (train_patient_idx, val_patient_idx) in enumerate(skf.split(patient_ids, patient_labels)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold+1}/{num_folds}")
        print(f"{'='*50}")

        # Get patient IDs for this fold
        train_patients = [patient_ids[i] for i in train_patient_idx]
        val_patients = [patient_ids[i] for i in val_patient_idx]

        # Create fold datasets
        train_dataset, val_dataset = create_fold_datasets(dataset, train_patients, val_patients, patient_to_indices)

        # Set transforms
        train_dataset.dataset.current_transform = train_dataset.dataset.train_transform
        val_dataset.dataset.current_transform = val_dataset.dataset.img_transform

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

        # Initialize model
        model = EfficientNetLesionClassifier().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train fold
        best_model_state, best_auc = train_one_fold(
            model, train_loader, val_loader, criterion, optimizer, device, fold, num_epochs
        )

        # Load best model
        model.load_state_dict(best_model_state)

        # Initialize and calibrate referral system
        referral_system = LesionReferralSystem()
        if len(val_dataset) > 0:
            # Get validation probabilities for calibration
            model.eval()
            val_probs = []
            val_labels = []
            with torch.no_grad():
                for images, labels, _, _ in val_loader:
                    images = images.to(device)
                    probs = model.get_probabilities(images)
                    val_probs.extend(probs.cpu().numpy().flatten())
                    val_labels.extend(labels.cpu().numpy())

            referral_system.calibrate_thresholds(np.array(val_probs), np.array(val_labels))

        # Evaluate fold
        fold_result = evaluate_fold(model, val_loader, referral_system, device, fold)
        fold_results.append(fold_result)
        fold_models.append(best_model_state)

        print(f"Fold {fold+1} Results:")
        print(f"  AUC: {fold_result['auc']:.4f}")
        print(f"  Uncertainty Rate: {fold_result['uncertain_rate']:.2f}")
        print(f"  Certain AUC: {fold_result['certain_auc']:.4f}")
        print(f"  Certain Samples: {fold_result['certain_samples']}/{fold_result['total_samples']}")

    # Cross-validation summary
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")

    auc_scores = [r['auc'] for r in fold_results]
    uncertain_rates = [r['uncertain_rate'] for r in fold_results]
    certain_aucs = [r['certain_auc'] for r in fold_results]

    print(f"AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"Uncertainty Rate: {np.mean(uncertain_rates):.2f} ± {np.std(uncertain_rates):.2f}")
    print(f"Certain AUC: {np.mean(certain_aucs):.4f} ± {np.std(certain_aucs):.4f}")

    # Save best model (highest AUC)
    best_fold = np.argmax(auc_scores)
    best_model_state = fold_models[best_fold]

    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': best_model_state,
        'fold': best_fold,
        'auc': auc_scores[best_fold],
        'config': {
            'backbone': 'efficientnet-b0',
            'num_folds': num_folds,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    }, 'models/best_model_cv.pth')

    print(f"\nBest model saved to models/best_model_cv.pth (Fold {best_fold+1}, AUC: {auc_scores[best_fold]:.4f})")


if __name__ == "__main__":
    main()