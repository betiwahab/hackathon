"""
model_scratch.py — CNN from scratch pour comparaison avec EfficientNetV2 (transfer learning).

Architecture : CNN 4 blocs (Conv → BN → ReLU → MaxPool) + tête de classification FC.
Aucun poids pré-entraîné. Même interface que EfficientNetLesionClassifier pour être
un drop-in dans train_compare.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Bloc convolutif de base
# ─────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool (optionnel)."""

    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────
# CNN from scratch
# ─────────────────────────────────────────────
class ScratchCNN(nn.Module):
    """
    CNN entraîné from scratch pour la classification de lésions POCUS.

    Entrée  : images RGB 224×224
    Sortie  : logit scalaire (compatible BCEWithLogitsLoss)

    Architecture :
        ConvBlock(3  → 32,  pool)   → 112×112
        ConvBlock(32 → 64,  pool)   →  56×56
        ConvBlock(64 → 128, pool)   →  28×28
        ConvBlock(128→ 256, pool)   →  14×14
        AdaptiveAvgPool → 4×4      (256 × 4 × 4 = 4096 features)
        Dropout(0.5)
        FC(4096 → 256) → ReLU → Dropout(0.3)
        FC(256  → 1)   (logit)

    Choix de design justifiés pour le papier :
    - Taille des filtres modeste (3×3) : standard en vision, réduit les params
    - BatchNorm à chaque bloc : stabilise l'entraînement sur peu de données
    - AdaptiveAvgPool : tolère des images non exactement 224×224
    - Dropout double : régularisation explicite (pas de pré-entraînement disponible)
    - ~1.3M paramètres (vs ~7M EfficientNetB0) : adapté à <200 images
    """

    def __init__(self, dropout=0.5):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            ConvBlock(3,   32,  pool=True),   # → 112×112
            ConvBlock(32,  64,  pool=True),   # →  56×56
            ConvBlock(64,  128, pool=True),   # →  28×28
            ConvBlock(128, 256, pool=True),   # →  14×14
        )

        # Pooling global → vecteur fixe, quelle que soit la résolution d'entrée
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Tête de classification
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),   # dropout plus léger en couche intermédiaire
            nn.Linear(256, 1),
        )

        # Initialisation explicite (utile sans pré-entraînement)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)   # flatten
        x = self.classifier(x)
        return x

    def get_probabilities(self, x):
        """Retourne des probabilités [0, 1] — même interface que EfficientNetLesionClassifier."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


# ─────────────────────────────────────────────
# Système de référence (même interface)
# ─────────────────────────────────────────────
class ScratchLesionReferralSystem:
    """
    Système de décision identique à LesionReferralSystem du modèle pré-entraîné.
    Utilisé pour l'évaluation détaillée dans train_compare.py.
    """

    def __init__(self, low_threshold=0.3, high_threshold=0.7):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def calibrate_thresholds(self, probs, labels):
        """Calibration simple sur les données de validation (optionnel)."""
        pass   # pour l'instant même interface, seuils fixes comme dans le script original

    def classify_and_refer(self, probs):
        """
        Retourne (decisions, confidences).
        decisions : 0 = No Lesion, 1 = Lesion
        """
        probs = probs.flatten()
        decisions = (probs > 0.5).astype(int)
        confidences = probs
        return decisions, confidences


# ─────────────────────────────────────────────
# Résumé rapide du modèle
# ─────────────────────────────────────────────
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = ScratchCNN()
    total, trainable = count_parameters(model)
    print(f"ScratchCNN — Total params    : {total:,}")
    print(f"ScratchCNN — Trainable params: {trainable:,}")

    # Vérification d'un forward pass
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    probs = model.get_probabilities(x)
    print(f"Input shape  : {x.shape}")
    print(f"Logits shape : {logits.shape}")
    print(f"Probs shape  : {probs.shape}")
    print(f"Probs range  : [{probs.min():.4f}, {probs.max():.4f}]")
