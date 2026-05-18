"""Dr. TABA CHABI - Entraînement et évaluation avancés du modèle de classification d'images médicales (Pocus)"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from dataset import MedicalDataset
from model_advanced import EfficientNetLesionClassifier, LesionReferralSystem


class TrainingHistory:
    """Enregistrer l'historique d'entraînement pour visualisation"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.val_accuracies = []
        self.epochs = []
    
    def add_epoch(self, epoch, train_loss, val_loss, val_auc, val_accuracy):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_aucs.append(val_auc)
        self.val_accuracies.append(val_accuracy)


def create_weighted_train_loader(train_dataset, batch_size):
    """Créer un DataLoader équilibré pour la classe minoritaire."""
    labels = [int(label.item()) for _, label, _, _ in train_dataset]
    class_counts = np.bincount(labels, minlength=2)

    if class_counts[0] == 0 or class_counts[1] == 0:
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), class_counts

    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    return DataLoader(train_dataset, batch_size=batch_size, sampler=sampler), class_counts


def get_patient_level_data(dataset):
    """Extraire données au niveau patient"""
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
    """Créer datasets pour un fold"""
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
    """Entraîner un fold avec historique"""
    
    best_auc = 0.0
    best_model_state = None
    history = TrainingHistory()

    for epoch in range(num_epochs):
        # === TRAINING ===
        model.train()
        train_loss = 0.0

        for images, labels, _, _ in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1} Train", leave=False):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        val_probs = []
        val_labels = []

        with torch.no_grad():
            for images, labels, _, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1))

                val_loss += loss.item()

                probs = model.get_probabilities(images)
                val_probs.extend(probs.cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy())

        # === CALCULER METRIQUES ===
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        
        if len(np.unique(val_labels)) > 1:
            val_auc = roc_auc_score(val_labels, val_probs)
        else:
            val_auc = 0.5
        
        val_preds_binary = (np.array(val_probs) > 0.5).astype(int)
        val_accuracy = np.mean(val_preds_binary == np.array(val_labels))
        
        # Enregistrer historique
        history.add_epoch(epoch, train_loss_avg, val_loss_avg, val_auc, val_accuracy)

        print(f"Fold {fold+1} Epoch {epoch+1}: Train Loss: {train_loss_avg:.4f}, "
              f"Val Loss: {val_loss_avg:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_accuracy:.4f}")

        # === SAVE BEST MODEL ===
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()

    return best_model_state, best_auc, history


def evaluate_fold_detailed(model, val_loader, referral_system, device, fold):
    """Évaluation détaillée pour matrice confusion et ROC"""
    
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

    return np.array(all_probs), np.array(all_labels), np.array(all_decisions)


def plot_training_curves(histories, fold_results, save_dir='results'):
    """Tracer courbes d'entraînement"""
    os.makedirs(save_dir, exist_ok=True)
    
    # === PLOT 1: Loss Curves (Tous les folds) ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for fold_idx, history in enumerate(histories):
        axes[0].plot(history.epochs, history.train_losses, label=f'Fold {fold_idx+1} Train', linewidth=2, alpha=0.7)
        axes[1].plot(history.epochs, history.val_losses, label=f'Fold {fold_idx+1} Val', linewidth=2, alpha=0.7)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Validation Loss Curves', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '01_loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_dir}/01_loss_curves.png")
    
    # === PLOT 2: AUC & Accuracy per Fold ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for fold_idx, history in enumerate(histories):
        axes[0].plot(history.epochs, history.val_aucs, marker='o', label=f'Fold {fold_idx+1}', linewidth=2)
        axes[1].plot(history.epochs, history.val_accuracies, marker='s', label=f'Fold {fold_idx+1}', linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('AUC', fontsize=12)
    axes[0].set_title('Validation AUC per Fold', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Validation Accuracy per Fold', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '02_auc_accuracy_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_dir}/02_auc_accuracy_curves.png")


def plot_confusion_matrices(fold_predictions, save_dir='results'):
    """Tracer matrices de confusion pour chaque fold"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for fold_idx, (probs, labels, decisions) in enumerate(fold_predictions):
        if fold_idx >= 5:
            break
        
        # Prédictions binaires (p > 0.5)
        preds = (probs > 0.5).astype(int)
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[fold_idx], 
                   cbar=False, square=True, annot_kws={'size': 14})
        
        axes[fold_idx].set_title(f'Fold {fold_idx+1} Confusion Matrix', fontsize=12, fontweight='bold')
        axes[fold_idx].set_ylabel('True Label', fontsize=11)
        axes[fold_idx].set_xlabel('Predicted Label', fontsize=11)

        # Ajuster dynamiquement les labels en fonction des classes présentes
        present_classes = np.unique(np.concatenate((labels, preds))).tolist()
        class_names = ['No Lesion' if c == 0 else 'Lesion' for c in present_classes]
        axes[fold_idx].set_xticklabels(class_names)
        axes[fold_idx].set_yticklabels(class_names)
        
        # Calculer métriques pour une matrice 2x2 ou 1x1
        if cm.shape == (1, 1):
            if present_classes[0] == 0:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        else:
            tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics_text = f'Sens: {sensitivity:.2f}\nSpec: {specificity:.2f}'
        axes[fold_idx].text(0.5, -0.15, metrics_text, ha='center', transform=axes[fold_idx].transAxes,
                           fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Masquer dernière subplot
    axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '03_confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_dir}/03_confusion_matrices.png")


def plot_roc_curves(fold_predictions, save_dir='results'):
    """Tracer courbes ROC pour chaque fold"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    aucs = []
    for fold_idx, (probs, labels, _) in enumerate(fold_predictions):
        if len(np.unique(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            
            ax.plot(fpr, tpr, lw=2.5, label=f'Fold {fold_idx+1} (AUC={roc_auc:.3f})')
    
    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC=0.500)')
    
    # Average AUC
    if aucs:
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        ax.text(0.5, 0.1, f'Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}', 
               ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Cross Validation', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '04_roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_dir}/04_roc_curves.png")


def plot_probability_distribution(fold_predictions, save_dir='results'):
    """Tracer distribution des probabilités prédites"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    all_probs_lesion = []
    all_probs_no_lesion = []
    
    for probs, labels, _ in fold_predictions:
        all_probs_lesion.extend(probs[labels == 1])
        all_probs_no_lesion.extend(probs[labels == 0])
    
    # Histograms
    ax.hist(all_probs_no_lesion, bins=15, alpha=0.6, label=f'No Lesion (n={len(all_probs_no_lesion)})', 
           color='blue', edgecolor='black')
    ax.hist(all_probs_lesion, bins=15, alpha=0.6, label=f'Lesion (n={len(all_probs_lesion)})', 
           color='red', edgecolor='black')
    
    # Seuils
    ax.axvline(0.3, color='green', linestyle='--', linewidth=2.5, label='Min Threshold (0.3)')
    ax.axvline(0.7, color='orange', linestyle='--', linewidth=2.5, label='Max Threshold (0.7)')
    
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '05_probability_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_dir}/05_probability_distribution.png")


def plot_cv_summary(histories, fold_results, save_dir='results'):
    """Tracer résumé CV: boxplots des métriques"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extraire dernière epoch pour chaque fold
    final_aucs = [history.val_aucs[-1] for history in histories]
    final_accuracies = [history.val_accuracies[-1] for history in histories]
    final_val_losses = [history.val_losses[-1] for history in histories]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # AUC
    axes[0].bar(range(1, 6), final_aucs, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axhline(np.mean(final_aucs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_aucs):.3f}')
    axes[0].fill_between(np.arange(0.5, 5.5, 1), np.mean(final_aucs) - np.std(final_aucs), 
                        np.mean(final_aucs) + np.std(final_aucs), alpha=0.2, color='red')
    axes[0].set_xlabel('Fold', fontsize=12)
    axes[0].set_ylabel('AUC', fontsize=12)
    axes[0].set_title('Validation AUC per Fold', fontsize=12, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].set_xticks(range(1, 6))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Accuracy
    axes[1].bar(range(1, 6), final_accuracies, color='forestgreen', edgecolor='black', alpha=0.7)
    axes[1].axhline(np.mean(final_accuracies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_accuracies):.3f}')
    axes[1].fill_between(np.arange(0.5, 5.5, 1), np.mean(final_accuracies) - np.std(final_accuracies), 
                        np.mean(final_accuracies) + np.std(final_accuracies), alpha=0.2, color='red')
    axes[1].set_xlabel('Fold', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Validation Accuracy per Fold', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].set_xticks(range(1, 6))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Val Loss
    axes[2].bar(range(1, 6), final_val_losses, color='coral', edgecolor='black', alpha=0.7)
    axes[2].axhline(np.mean(final_val_losses), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_val_losses):.3f}')
    axes[2].fill_between(np.arange(0.5, 5.5, 1), np.mean(final_val_losses) - np.std(final_val_losses), 
                        np.mean(final_val_losses) + np.std(final_val_losses), alpha=0.2, color='red')
    axes[2].set_xlabel('Fold', fontsize=12)
    axes[2].set_ylabel('Validation Loss', fontsize=12)
    axes[2].set_title('Validation Loss per Fold', fontsize=12, fontweight='bold')
    axes[2].set_xticks(range(1, 6))
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '06_cv_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_dir}/06_cv_summary.png")
    
    # Afficher résumé
    print(f"\n{'='*60}")
    print("CV SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"AUC:      {np.mean(final_aucs):.4f} ± {np.std(final_aucs):.4f}")
    print(f"Accuracy: {np.mean(final_accuracies):.4f} ± {np.std(final_accuracies):.4f}")
    print(f"Val Loss: {np.mean(final_val_losses):.4f} ± {np.std(final_val_losses):.4f}")


def main():
    # === CONFIG ===
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

    patient_ids, patient_labels, patient_to_indices = get_patient_level_data(dataset)

    print(f"Dataset: {len(dataset)} images from {len(patient_ids)} patients")
    print(f"Lesion prevalence: {np.mean(patient_labels):.2f}")

    # === CROSS-VALIDATION ===
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_results = []
    fold_models = []
    histories = []
    fold_predictions = []

    for fold, (train_patient_idx, val_patient_idx) in enumerate(skf.split(patient_ids, patient_labels)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{num_folds}")
        print(f"{'='*60}")

        train_patients = [patient_ids[i] for i in train_patient_idx]
        val_patients = [patient_ids[i] for i in val_patient_idx]

        train_dataset, val_dataset = create_fold_datasets(dataset, train_patients, val_patients, patient_to_indices)

        train_dataset.dataset.current_transform = train_dataset.dataset.train_transform
        val_dataset.dataset.current_transform = val_dataset.dataset.img_transform

        train_loader, train_class_counts = create_weighted_train_loader(train_dataset, batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
        print(f"Train class counts: {train_class_counts.tolist()} (balanced sampler used: {train_class_counts[0] > 0 and train_class_counts[1] > 0})")

        model = EfficientNetLesionClassifier().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # === TRAIN WITH HISTORY ===
        best_model_state, best_auc, history = train_one_fold(
            model, train_loader, val_loader, criterion, optimizer, device, fold, num_epochs
        )

        model.load_state_dict(best_model_state)
        histories.append(history)
        fold_models.append(best_model_state)

        # === DETAILED EVALUATION ===
        referral_system = LesionReferralSystem()
        
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

        # Get predictions for confusion matrix & ROC
        probs, labels, decisions = evaluate_fold_detailed(model, val_loader, referral_system, device, fold)
        fold_predictions.append((probs, labels, decisions))

        fold_result = {
            'auc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5,
            'total_samples': len(labels),
        }
        fold_results.append(fold_result)

        print(f"Fold {fold+1} Final AUC: {fold_result['auc']:.4f}")

    # === VISUALIZATIONS ===
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    os.makedirs('results', exist_ok=True)
    
    plot_training_curves(histories, fold_results, save_dir='results')
    plot_confusion_matrices(fold_predictions, save_dir='results')
    plot_roc_curves(fold_predictions, save_dir='results')
    plot_probability_distribution(fold_predictions, save_dir='results')
    plot_cv_summary(histories, fold_results, save_dir='results')
    
    print(f"\n✓ All visualizations saved to: results/")

    # === SAVE BEST MODEL ===
    best_fold = np.argmax([r['auc'] for r in fold_results])
    best_model_state = fold_models[best_fold]

    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': best_model_state,
        'fold': best_fold,
        'auc': fold_results[best_fold]['auc'],
        'config': {
            'backbone': 'efficientnet-b0',
            'num_folds': num_folds,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    }, 'models/best_model_cv.pth')

    print(f"\nBest model saved (Fold {best_fold+1}, AUC: {fold_results[best_fold]['auc']:.4f})")


if __name__ == "__main__":
    main()