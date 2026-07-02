"""
train_compare.py — Entraînement et évaluation comparative : from scratch vs transfer learning.

Usage :
    python train_compare.py                        # scratch (défaut)
    python train_compare.py --mode scratch
    python train_compare.py --mode pretrained
    python train_compare.py --mode both            # les deux à la suite + tableau comparatif

Options supplémentaires :
    --epochs 20
    --folds  5
    --batch_size 8
    --lr 1e-4
    --results_dir results_compare
    --save_model                   # sauvegarder le meilleur modèle .pth
"""

import argparse
import os
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, roc_curve, auc,
                             confusion_matrix, classification_report)
from collections import defaultdict
from tqdm import tqdm

# ── Imports locaux ────────────────────────────────────────────────────────────
from dataset import MedicalDataset
from model_scratch import ScratchCNN, ScratchLesionReferralSystem

try:
    from model_advanced import EfficientNetLesionClassifier, LesionReferralSystem
    PRETRAINED_AVAILABLE = True
except ImportError:
    PRETRAINED_AVAILABLE = False
    print("[WARN] model_advanced.py introuvable — mode 'pretrained' indisponible.")


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires dataset
# ─────────────────────────────────────────────────────────────────────────────

class TrainingHistory:
    def __init__(self):
        self.train_losses = []
        self.val_losses   = []
        self.val_aucs     = []
        self.val_accs     = []
        self.epochs       = []

    def add(self, epoch, tl, vl, auc_, acc):
        self.epochs.append(epoch)
        self.train_losses.append(tl)
        self.val_losses.append(vl)
        self.val_aucs.append(auc_)
        self.val_accs.append(acc)


def get_patient_level_data(dataset):
    """Groupe les images par patient (préfixe ABxxx)."""
    patient_to_indices = defaultdict(list)
    patient_labels     = {}

    for idx in range(len(dataset)):
        _, label, patient_id, _ = dataset[idx]
        patient_to_indices[patient_id].append(idx)
        if patient_id not in patient_labels:
            patient_labels[patient_id] = int(label.item())

    patient_ids = list(patient_labels.keys())
    labels      = [patient_labels[pid] for pid in patient_ids]
    return patient_ids, np.array(labels), patient_to_indices


def create_fold_datasets(dataset, train_patients, val_patients, patient_to_indices):
    train_idx = [i for p in train_patients for i in patient_to_indices[p]]
    val_idx   = [i for p in val_patients   for i in patient_to_indices[p]]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def make_weighted_loader(subset, batch_size):
    from torch.utils.data import WeightedRandomSampler
    labels = [int(subset.dataset[i][1].item()) for i in subset.indices]
    counts = np.bincount(labels, minlength=2)
    if counts[0] == 0 or counts[1] == 0:
        return DataLoader(subset, batch_size=batch_size, shuffle=True), counts
    weights = 1.0 / counts
    sample_w = [weights[l] for l in labels]
    sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
    return DataLoader(subset, batch_size=batch_size, sampler=sampler), counts


# ─────────────────────────────────────────────────────────────────────────────
# Entraînement d'un fold
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(model, train_loader, val_loader, criterion, optimizer,
               device, fold, num_epochs):
    best_auc   = 0.0
    best_state = None
    history    = TrainingHistory()

    for epoch in range(num_epochs):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        tl = 0.0
        for images, labels, _, _ in tqdm(
                train_loader,
                desc=f"  Fold {fold+1} | Epoch {epoch+1:02d}/{num_epochs} [train]",
                leave=False):
            images  = images.to(device)
            labels  = labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            loss    = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            tl += loss.item()

        # ── Val ───────────────────────────────────────────────────────────
        model.eval()
        vl, v_probs, v_labels = 0.0, [], []
        with torch.no_grad():
            for images, labels, _, _ in val_loader:
                images  = images.to(device)
                labels  = labels.to(device)
                out     = model(images)
                vl     += criterion(out, labels.float().unsqueeze(1)).item()
                probs   = model.get_probabilities(images)
                v_probs.extend(probs.cpu().numpy().flatten())
                v_labels.extend(labels.cpu().numpy())

        tl_avg = tl / len(train_loader)
        vl_avg = vl / len(val_loader)
        v_probs  = np.array(v_probs)
        v_labels = np.array(v_labels)
        val_auc  = roc_auc_score(v_labels, v_probs) if len(np.unique(v_labels)) > 1 else 0.5
        val_acc  = np.mean((v_probs > 0.5).astype(int) == v_labels)

        history.add(epoch, tl_avg, vl_avg, val_auc, val_acc)
        print(f"  Fold {fold+1} | Ep {epoch+1:02d} — "
              f"TL: {tl_avg:.4f}  VL: {vl_avg:.4f}  "
              f"AUC: {val_auc:.4f}  Acc: {val_acc:.4f}")

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    return best_state, best_auc, history


# ─────────────────────────────────────────────────────────────────────────────
# Évaluation détaillée d'un fold
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_fold(model, val_loader, referral_system, device):
    model.eval()
    all_probs, all_labels, all_decisions = [], [], []
    with torch.no_grad():
        for images, labels, _, _ in val_loader:
            images  = images.to(device)
            probs   = model.get_probabilities(images)
            decs, _ = referral_system.classify_and_refer(probs.cpu().numpy())
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy())
            all_decisions.extend(decs)
    return np.array(all_probs), np.array(all_labels), np.array(all_decisions)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline complet pour un mode (scratch ou pretrained)
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(mode, dataset, patient_ids, patient_labels, patient_to_indices,
                 args, device, save_dir):
    """
    Lance la cross-validation complète pour un mode donné.
    Retourne un dict de résultats agrégés (pour comparaison finale).
    """
    os.makedirs(save_dir, exist_ok=True)
    label = "From Scratch" if mode == "scratch" else "Transfer Learning (EfficientNetV2)"
    print(f"\n{'═'*65}")
    print(f"  MODE : {label}")
    print(f"{'═'*65}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    histories, fold_results, fold_preds = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(patient_ids, patient_labels)):
        print(f"\n{'─'*55}")
        print(f"  FOLD {fold+1}/{args.folds}")
        print(f"{'─'*55}")

        tr_patients = [patient_ids[i] for i in tr_idx]
        va_patients = [patient_ids[i] for i in va_idx]
        tr_ds, va_ds = create_fold_datasets(
            dataset, tr_patients, va_patients, patient_to_indices)

        # Appliquer les bonnes transforms
        tr_ds.dataset.current_transform = tr_ds.dataset.train_transform
        va_ds.dataset.current_transform = va_ds.dataset.img_transform

        tr_loader, tr_counts = make_weighted_loader(tr_ds, args.batch_size)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)

        print(f"  Train: {len(tr_ds)} images  |  Val: {len(va_ds)} images")
        print(f"  Class balance train — No Lesion: {tr_counts[0]}, Lesion: {tr_counts[1]}")

        # Construire le modèle selon le mode
        if mode == "scratch":
            model = ScratchCNN(dropout=0.5).to(device)
            referral = ScratchLesionReferralSystem()
        else:
            model = EfficientNetLesionClassifier().to(device)
            referral = LesionReferralSystem()

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=1e-4 if mode == "scratch" else 0.0)

        best_state, best_auc, history = train_fold(
            model, tr_loader, va_loader, criterion, optimizer,
            device, fold, args.epochs)

        model.load_state_dict(best_state)
        histories.append(history)

        # Calibration + évaluation finale
        tmp_probs, tmp_labels = [], []
        model.eval()
        with torch.no_grad():
            for images, labels, _, _ in va_loader:
                p = model.get_probabilities(images.to(device))
                tmp_probs.extend(p.cpu().numpy().flatten())
                tmp_labels.extend(labels.numpy())
        referral.calibrate_thresholds(np.array(tmp_probs), np.array(tmp_labels))

        probs, labels, decisions = evaluate_fold(model, va_loader, referral, device)
        fold_preds.append((probs, labels, decisions))

        fold_auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
        preds_b  = (probs > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds_b, labels=[0, 1]).ravel()
        sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0

        fold_results.append({
            'auc': fold_auc,
            'accuracy': np.mean(preds_b == labels),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1': f1,
        })
        print(f"  → Fold {fold+1} — AUC: {fold_auc:.4f} | Sens: {sensitivity:.4f} | Spec: {specificity:.4f}")

    # ── Sauvegarder les graphes ─────────────────────────────────────────────
    _plot_all(histories, fold_preds, fold_results, save_dir, label)

    # ── Résultats agrégés ──────────────────────────────────────────────────
    summary = {k: {
        'mean': np.mean([r[k] for r in fold_results]),
        'std':  np.std( [r[k] for r in fold_results]),
    } for k in fold_results[0]}

    print(f"\n{'═'*65}")
    print(f"  RÉSUMÉ — {label}")
    print(f"{'═'*65}")
    for k, v in summary.items():
        print(f"  {k:15s}: {v['mean']:.4f} ± {v['std']:.4f}")

    # Sauvegarder le résumé JSON
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump({'mode': mode, 'summary': summary}, f, indent=2)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions de visualisation (réutilisées de train_cv_with_plots.py + comparatif)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_all(histories, fold_preds, fold_results, save_dir, mode_label):
    _plot_loss_curves(histories, save_dir)
    _plot_auc_acc_curves(histories, save_dir)
    _plot_confusion_matrices(fold_preds, save_dir)
    _plot_roc_curves(fold_preds, save_dir)
    _plot_cv_summary(histories, save_dir, mode_label)
    print(f"\n  ✓ Graphes sauvegardés dans : {save_dir}/")


def _plot_loss_curves(histories, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, h in enumerate(histories):
        axes[0].plot(h.epochs, h.train_losses, label=f'Fold {i+1}', linewidth=2, alpha=0.8)
        axes[1].plot(h.epochs, h.val_losses,   label=f'Fold {i+1}', linewidth=2, alpha=0.8)
    for ax, title in zip(axes, ['Training Loss', 'Validation Loss']):
        ax.set(xlabel='Epoch', ylabel='Loss', title=title)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '01_loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_auc_acc_curves(histories, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, h in enumerate(histories):
        axes[0].plot(h.epochs, h.val_aucs,  marker='o', label=f'Fold {i+1}', linewidth=2)
        axes[1].plot(h.epochs, h.val_accs,  marker='s', label=f'Fold {i+1}', linewidth=2)
    for ax, (title, ylabel) in zip(axes, [('Validation AUC', 'AUC'), ('Validation Accuracy', 'Accuracy')]):
        ax.set(xlabel='Epoch', ylabel=ylabel, title=title, ylim=[0, 1])
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '02_auc_acc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_confusion_matrices(fold_preds, save_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, (probs, labels, _) in enumerate(fold_preds[:5]):
        preds = (probs > 0.5).astype(int)
        cm    = confusion_matrix(labels, preds, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    cbar=False, square=True, annot_kws={'size': 14})
        tn, fp, fn, tp = cm.ravel()
        sens = tp/(tp+fn) if (tp+fn) > 0 else 0
        spec = tn/(tn+fp) if (tn+fp) > 0 else 0
        axes[i].set(title=f'Fold {i+1} — Sens:{sens:.2f} / Spec:{spec:.2f}',
                    xlabel='Predicted', ylabel='True Label')
        axes[i].set_xticklabels(['No Lesion', 'Lesion'])
        axes[i].set_yticklabels(['No Lesion', 'Lesion'])
    axes[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '03_confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_roc_curves(fold_preds, save_dir):
    fig, ax = plt.subplots(figsize=(9, 7))
    aucs = []
    for i, (probs, labels, _) in enumerate(fold_preds):
        if len(np.unique(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, probs)
            a = auc(fpr, tpr); aucs.append(a)
            ax.plot(fpr, tpr, lw=2, label=f'Fold {i+1} (AUC={a:.3f})')
    ax.plot([0,1],[0,1],'k--', label='Random (0.500)')
    if aucs:
        ax.text(0.55, 0.08, f'Mean AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}',
                fontsize=11, bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
           title='ROC Curves — Cross-Validation', xlim=[0,1], ylim=[0,1])
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '04_roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_cv_summary(histories, save_dir, mode_label):
    final_aucs  = [h.val_aucs[-1]  for h in histories]
    final_accs  = [h.val_accs[-1]  for h in histories]
    final_vloss = [h.val_losses[-1] for h in histories]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Cross-Validation Summary — {mode_label}', fontsize=13, fontweight='bold')

    for ax, vals, color, ylabel in zip(
            axes,
            [final_aucs, final_accs, final_vloss],
            ['steelblue', 'forestgreen', 'coral'],
            ['AUC', 'Accuracy', 'Val Loss']):
        n = len(vals)
        ax.bar(range(1, n+1), vals, color=color, edgecolor='black', alpha=0.7)
        m, s = np.mean(vals), np.std(vals)
        ax.axhline(m, color='red', linestyle='--', linewidth=2, label=f'Mean: {m:.3f}')
        ax.fill_between(np.arange(0.5, n+0.5, 1), m-s, m+s, alpha=0.2, color='red')
        ax.set(xlabel='Fold', ylabel=ylabel,
               title=f'Validation {ylabel} per Fold', xticks=range(1, n+1))
        if ylabel != 'Val Loss':
            ax.set_ylim([0, 1])
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '05_cv_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison(scratch_summary, pretrained_summary, save_dir):
    """Graphe comparatif côte à côte des deux modes."""
    os.makedirs(save_dir, exist_ok=True)
    metrics = ['auc', 'accuracy', 'sensitivity', 'specificity', 'f1']
    labels  = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-Score']
    x       = np.arange(len(metrics))
    width   = 0.35

    s_means = [scratch_summary[m]['mean']    for m in metrics]
    s_stds  = [scratch_summary[m]['std']     for m in metrics]
    p_means = [pretrained_summary[m]['mean'] for m in metrics]
    p_stds  = [pretrained_summary[m]['std']  for m in metrics]

    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - width/2, s_means, width, yerr=s_stds, capsize=5,
                label='From Scratch', color='coral',    edgecolor='black', alpha=0.85)
    b2 = ax.bar(x + width/2, p_means, width, yerr=p_stds, capsize=5,
                label='Transfer Learning (EfficientNetV2)', color='steelblue', edgecolor='black', alpha=0.85)

    # Annoter les valeurs
    for bar in [b1, b2]:
        for rect in bar:
            h = rect.get_height()
            ax.annotate(f'{h:.3f}',
                        xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Score (mean ± std over 5 folds)', fontsize=12)
    ax.set_title('Scratch vs Transfer Learning — 5-Fold Cross-Validation Comparison\n'
                 'ABreast/PACE POCUS Dataset (patient-level split)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim([0, 1.08])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = os.path.join(save_dir, 'comparison_scratch_vs_pretrained.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Graphe comparatif sauvegardé : {out}")

    # Tableau texte
    print(f"\n{'═'*65}")
    print(f"  COMPARAISON FINALE")
    print(f"{'═'*65}")
    print(f"  {'Métrique':<15} {'Scratch':>18} {'Transfer Learning':>22}")
    print(f"  {'-'*57}")
    for m, l in zip(metrics, labels):
        s = scratch_summary[m]
        p = pretrained_summary[m]
        delta = p['mean'] - s['mean']
        sign  = '+' if delta >= 0 else ''
        print(f"  {l:<15} {s['mean']:.4f} ± {s['std']:.4f}   "
              f"{p['mean']:.4f} ± {p['std']:.4f}   Δ={sign}{delta:.4f}")
    print(f"{'═'*65}")


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Comparaison Scratch vs Transfer Learning — ABreast/PACE')
    parser.add_argument('--mode', type=str, default='scratch',
                        choices=['scratch', 'pretrained', 'both'],
                        help='Mode d\'entraînement (défaut: scratch)')
    parser.add_argument('--epochs',      type=int,   default=20)
    parser.add_argument('--folds',       type=int,   default=5)
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--results_dir', type=str,   default='results_compare')
    parser.add_argument('--save_model',  action='store_true',
                        help='Sauvegarder le meilleur modèle .pth')
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Dispositif : {device}")
    print(f"  Mode       : {args.mode}")
    print(f"  Epochs     : {args.epochs} | Folds : {args.folds} | "
          f"Batch : {args.batch_size} | LR : {args.lr}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("\n  Chargement du dataset...")
    dataset = MedicalDataset(
        csv_file="data/ABreast-Classification.csv",
        image_dir="data/images",
        mask_dir="data/masks"
    )
    patient_ids, patient_labels, patient_to_indices = get_patient_level_data(dataset)
    print(f"  {len(dataset)} images | {len(patient_ids)} patients | "
          f"Prévalence lésion : {np.mean(patient_labels):.2%}")

    summaries = {}

    if args.mode in ('scratch', 'both'):
        t0 = time.time()
        summaries['scratch'] = run_pipeline(
            mode='scratch',
            dataset=dataset,
            patient_ids=patient_ids,
            patient_labels=patient_labels,
            patient_to_indices=patient_to_indices,
            args=args,
            device=device,
            save_dir=os.path.join(args.results_dir, 'scratch'),
        )
        print(f"\n  ⏱  Scratch terminé en {(time.time()-t0)/60:.1f} min")

    if args.mode in ('pretrained', 'both'):
        if not PRETRAINED_AVAILABLE:
            print("\n  [ERREUR] model_advanced.py introuvable. "
                  "Mode pretrained ignoré.")
        else:
            t0 = time.time()
            summaries['pretrained'] = run_pipeline(
                mode='pretrained',
                dataset=dataset,
                patient_ids=patient_ids,
                patient_labels=patient_labels,
                patient_to_indices=patient_to_indices,
                args=args,
                device=device,
                save_dir=os.path.join(args.results_dir, 'pretrained'),
            )
            print(f"\n  ⏱  Transfer learning terminé en {(time.time()-t0)/60:.1f} min")

    if args.mode == 'both' and 'scratch' in summaries and 'pretrained' in summaries:
        plot_comparison(
            scratch_summary=summaries['scratch'],
            pretrained_summary=summaries['pretrained'],
            save_dir=args.results_dir,
        )

    print(f"\n  ✓ Tout est sauvegardé dans : {args.results_dir}/")


if __name__ == '__main__':
    main()
