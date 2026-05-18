"""Dr. TABA CHABI - Evaluation avancée du modèle de classification d'images médicales (Pocus)
evaluate_advanced.py - Advanced Model Evaluation with Referral System and Grad-CAM Visualization"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from dataset import MedicalDataset
from model_advanced import EfficientNetLesionClassifier, LesionReferralSystem, generate_gradcam_heatmap, visualize_gradcam_comparison


def load_trained_model(model_path='models/best_model_cv.pth'):
    """
    Load the trained model from checkpoint
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = EfficientNetLesionClassifier()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model from fold {checkpoint['fold']+1} with AUC {checkpoint['auc']:.4f}")

    return model, device, checkpoint


def evaluate_with_referral(model, dataloader, referral_system, device):
    """
    Evaluate model with referral system
    """
    model.eval()
    all_probs = []
    all_labels = []
    all_decisions = []
    all_image_ids = []

    with torch.no_grad():
        for images, labels, patient_ids, image_ids in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            probs = model.get_probabilities(images)
            decisions, _ = referral_system.classify_and_refer(probs.cpu().numpy())

            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())
            all_decisions.extend(decisions)
            all_image_ids.extend(image_ids)

    return np.array(all_probs), np.array(all_labels), np.array(all_decisions), all_image_ids


def generate_gradcam_visualizations(model, dataset, device, num_samples=5, save_dir='gradcam_results'):
    """
    Generate Grad-CAM visualizations for sample images
    """
    os.makedirs(save_dir, exist_ok=True)

    # Get target layer for Grad-CAM
    target_layers = model.get_gradcam_target_layers()

    # Sample some images
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for idx in indices:
        image, label, patient_id, image_id = dataset[idx]

        # Add batch dimension
        image_batch = image.unsqueeze(0).to(device)

        # Generate Grad-CAM heatmap
        heatmap = generate_gradcam_heatmap(model, image_batch, target_layers)

        # Try to load corresponding mask if available
        mask_path = os.path.join("data/masks", f"{image_id.split('.')[0]}.tiff")
        mask = None
        if os.path.exists(mask_path):
            from PIL import Image
            mask = Image.open(mask_path).convert("L")
            import torchvision.transforms as transforms
            mask = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])(mask)

        # Visualize
        save_path = os.path.join(save_dir, f"gradcam_{image_id.split('.')[0]}.png")
        visualize_gradcam_comparison(image, heatmap, mask, save_path)

        print(f"Saved Grad-CAM visualization: {save_path}")


def main():
    print("Loading trained model...")
    model, device, checkpoint = load_trained_model()

    print("Loading dataset...")
    dataset = MedicalDataset(
        csv_file="data/ABreast-Classification.csv",
        image_dir="data/images",
        mask_dir="data/masks"
    )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Initialize referral system with calibrated thresholds
    referral_system = LesionReferralSystem()

    # For demonstration, use default thresholds (would be calibrated in real scenario)
    print(f"Using referral thresholds: min={referral_system.minimal_threshold:.3f}, "
          f"max={referral_system.acceptable_threshold:.3f}")

    print("\nEvaluating model with referral system...")
    probs, labels, decisions, image_ids = evaluate_with_referral(model, dataloader, referral_system, device)

    # Calculate metrics
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5

    # Referral statistics
    decision_counts = np.bincount(decisions, minlength=3)
    uncertain_rate = decision_counts[2] / len(decisions)

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {len(labels)}")
    print(f"AUC: {auc:.4f}")
    print(f"Uncertainty rate: {uncertain_rate:.2f}")
    print()
    print("Referral decisions:")
    print(f"  No lesion: {decision_counts[0]} samples ({decision_counts[0]/len(decisions):.1%})")
    print(f"  Uncertain: {decision_counts[2]} samples ({decision_counts[2]/len(decisions):.1%})")
    print(f"  Lesion: {decision_counts[1]} samples ({decision_counts[1]/len(decisions):.1%})")

    # Certain predictions analysis
    certain_mask = decisions != 2
    if np.sum(certain_mask) > 0:
        certain_labels = labels[certain_mask]
        certain_probs = probs[certain_mask]

        certain_auc = roc_auc_score(certain_labels, certain_probs) if len(np.unique(certain_labels)) > 1 else 0.5

        print(f"\nCertain predictions ({np.sum(certain_mask)}/{len(labels)}):")
        print(f"  AUC on certain: {certain_auc:.4f}")

        # Classification report for certain predictions
        certain_preds_binary = (certain_probs > 0.5).astype(int)
        print("\nClassification report (certain predictions):")
        print(classification_report(certain_labels, certain_preds_binary, target_names=['No Lesion', 'Lesion']))

    print(f"\n{'='*60}")
    print("GENERATING GRAD-CAM VISUALIZATIONS")
    print(f"{'='*60}")

    generate_gradcam_visualizations(model, dataset, device, num_samples=5)

    print("\nEvaluation complete!")
    print("Check 'gradcam_results/' for Grad-CAM visualizations")


if __name__ == "__main__":
    main()