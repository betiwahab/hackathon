"""Dr. TABA CHABI - Modèle de classification d'images médicales avancé (Pocus)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


class EfficientNetLesionClassifier(nn.Module):
    """
    EfficientNet-B0 based lesion classifier with temperature scaling
    """
    def __init__(self, num_classes=1, temperature=1.0):
        super(EfficientNetLesionClassifier, self).__init__()

        # Load EfficientNet-B0 pretrained on ImageNet
        # modèle de réseau de neurones convolutionnel pré-entraîné appelé EfficientNet-B0 comme 
        # backbone (extracteur de caractéristiques).
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')

        # Replace classifier head
        #Je remplace la dernière couche du modèle EfficientNet pour qu’il prédise mes classes à moi au lieu des classes ImageNet.
        #sert à modifier la dernière couche (fully connected layer) d’un modèle EfficientNet dans PyTorch afin de l’adapter à ton problème de classification.
        self.backbone._fc = nn.Linear(self.backbone._fc.in_features, num_classes)

        # Temperature scaling for calibration
        """La température est un hyperparamètre utilisé pour ajuster la confiance des prédictions d’un modèle de classification.
        La température est souvent utilisée pour :

            calibration des probabilités
            contrastive learning
            softmax scaling
            knowledge distillation
            attention mechanisms


        En ajustant la température, on peut rendre les prédictions du modèle plus ou moins confiantes, 
        ce qui peut être utile pour améliorer la performance sur des tâches spécifiques ou pour mieux gérer 
        les incertitudes dans les données.

        Effet de la température
         Température élevée
        T > 1

         probabilités plus “douces”

         Température faible
        T < 1

         prédictions plus “dures” et confiantes

        6. Intuition simple

        La température contrôle :

         le niveau de confiance des prédictions
        """

        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x):
        #le backbone extrait automatiquement :textures,contours,motifs,caractéristiques profondes. 
        # Ensuite, la tête de classification (la dernière couche) prend ces caractéristiques et produit une prédiction finale.
        #en retournant les logits non calibrés, on peut appliquer une fonction de perte appropriée (comme BCEWithLogitsLoss) qui intègre 
        # la sigmoid et gère la calibration de manière efficace.
        logits = self.backbone(x)
        # calibrer les prédictions du modèle en ajustant la température, ce qui peut aider à améliorer la performance 
        # du modèle sur des tâches spécifiques ou à mieux gérer les incertitudes dans les données.
        scaled_logits = logits / self.temperature
        return scaled_logits
    #Application de la fonction sigmoid pour obtenir des probabilités calibrées à partir des logits non calibrés produits par le modèle.
    def get_probabilities(self, x):
        """Return calibrated probabilities"""
        logits = self.forward(x)
        return torch.sigmoid(logits)
    #choisir la couche du réseau sur laquelle on va appliquer Grad-CAM pour visualiser les régions de l’image qui ont le plus contribué à la décision du modèle.
    def get_gradcam_target_layers(self):
        """Return target layers for Grad-CAM (last conv block)"""
        return [self.backbone._blocks[-1]]


class LesionReferralSystem:
    """
    Dual threshold referral system for lesion classification
    """
    def __init__(self, minimal_threshold=0.3, acceptable_threshold=0.7):
        self.minimal_threshold = minimal_threshold
        self.acceptable_threshold = acceptable_threshold

    def classify_and_refer(self, probabilities):
        """
        Apply dual threshold logic
        Returns: decision (0=no lesion, 1=lesion, 2=uncertain), confidence_score
        """
        probs = probabilities.flatten()

        decisions = []
        for p in probs:
            if p < self.minimal_threshold:
                decisions.append(0)  # No visible lesion
            elif p > self.acceptable_threshold:
                decisions.append(1)  # Visible lesion / refer
            else:
                decisions.append(2)  # Uncertain / repeat or expert review

        return np.array(decisions), probs

    def calibrate_thresholds(self, val_probs, val_labels, target_uncertainty_rate=0.2):
        """
        Calibrate thresholds using validation data to achieve target uncertainty rate
        """
        # Sort probabilities
        sorted_indices = np.argsort(val_probs)
        sorted_probs = val_probs[sorted_indices]
        sorted_labels = val_labels[sorted_indices]

        n_samples = len(sorted_probs)
        target_uncertain = int(n_samples * target_uncertainty_rate)

        # Find thresholds that maximize certain classifications while maintaining uncertainty rate
        best_score = 0
        best_min_thresh = 0.3
        best_max_thresh = 0.7

        for min_thresh in np.arange(0.1, 0.4, 0.05):
            for max_thresh in np.arange(0.6, 0.9, 0.05):
                if min_thresh >= max_thresh:
                    continue

                uncertain_mask = (sorted_probs >= min_thresh) & (sorted_probs <= max_thresh)
                uncertain_count = np.sum(uncertain_mask)

                if uncertain_count <= target_uncertain:
                    # Score based on certain classifications
                    certain_count = n_samples - uncertain_count
                    score = certain_count / n_samples
                    if score > best_score:
                        best_score = score
                        best_min_thresh = min_thresh
                        best_max_thresh = max_thresh

        self.minimal_threshold = best_min_thresh
        self.acceptable_threshold = best_max_thresh
        print(f"Calibrated thresholds: min={best_min_thresh:.3f}, max={best_max_thresh:.3f}")

#Cette fonction génère une carte de chaleur Grad-CAM pour visualiser où le modèle “regarde” dans l’image afin de prendre sa décision.
def generate_gradcam_heatmap(model, input_tensor, target_layer):
    """
    Generate Grad-CAM heatmap for lesion localization
    """
    # Create Grad-CAM explainer
    cam = GradCAM(model=model, target_layers=target_layer)

    # For binary classification with a single logit output, use index 0.
    targets = [ClassifierOutputTarget(0)]
    print(f"Generating Grad-CAM for target layer: {target_layer}")
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    return grayscale_cam[0]


def visualize_gradcam_comparison(image, heatmap, mask=None, save_path=None):
    """
    Visualize Grad-CAM heatmap with optional ground truth mask comparison
    """
    # Convert tensors to numpy
    if torch.is_tensor(image):
        image = image.permute(1, 2, 0).cpu().numpy()
    if torch.is_tensor(heatmap):
        heatmap = heatmap.cpu().numpy()

    # Denormalize image (reverse ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    # Create heatmap overlay
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image.astype(np.float32), 0.7, heatmap_colored.astype(np.float32)/255, 0.3, 0)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')

    if mask is not None:
        # Add mask comparison if available
        plt.figure(figsize=(5, 5))
        plt.imshow(mask.squeeze(), cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()

    plt.close('all')