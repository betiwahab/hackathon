# 📊 POCUS Lesion Classification Pipeline - Guide d'Utilisation

## 🚀 Démarrage Rapide

```bash
# 1. Explorer le dataset
python explore_data.py

# 2. Entraîner avec visualisations
python train_cv_with_plots.py

# 3. Évaluer le modèle
python evaluate_advanced.py
```

---

## 📁 Structure du Projet

```
hackaton_bon/
├── 📊 DATA EXPLORATION
│   └── explore_data.py                # Analyser le dataset
│
├── 🎯 TRAINING & EVALUATION
│   ├── train_cv_with_plots.py        # Entraînement 5-fold + graphes
│   ├── train_cv.py                   # Entraînement (version original)
│   ├── evaluate_advanced.py          # Évaluation + Grad-CAM
│
├── 🧠 CORE COMPONENTS
│   ├── dataset.py                    # PyTorch Dataset
│   ├── model_advanced.py             # EfficientNet + Grad-CAM
│
├── 📈 RESULTS & MODELS
│   ├── results/                      # Visualisations & graphes
│   │   ├── 01_loss_curves.png
│   │   ├── 02_auc_accuracy_curves.png
│   │   ├── 03_confusion_matrices.png
│   │   ├── 04_roc_curves.png
│   │   ├── 05_probability_distribution.png
│   │   ├── 06_cv_summary.png
│   │   ├── 07_metadata_distribution.png
│   │   ├── 08_sample_images.png
│   │   ├── 09_label_distribution.png
│   │   ├── 10_image_properties.png
│   │
│   ├── models/
│   │   └── best_model_cv.pth         # Meilleur modèle
│   │
│   └── gradcam_results/              # Visualisations Grad-CAM
│
├── 📚 DOCUMENTATION
│   ├── DOCUMENTATION.md              # Vue d'ensemble (FR)
│   ├── DOCUMENTATION_DETAILLEE.md    # Complète (FR)
│   └── README.md                     # Ce fichier
│
├── 📋 CONFIG
│   ├── requirements.txt
│   └── .gitignore
│
└── 📊 DATA
    ├── ABreast-Classification.csv
    ├── images/                       # Images POCUS
    ├── masks/                        # Masques segmentation (optionnel)
    └── ...
```

---

## 🔍 Étape 1: Explorer les Données

### Commande
```bash
python explore_data.py
```

### Génère
| Graphe | Description |
|--------|-------------|
| **07_metadata_distribution.png** | Distribution des métadonnées CSV |
| **08_sample_images.png** | 12 images échantillon du dataset |
| **09_label_distribution.png** | Distribution lésions vs pas lésions |
| **10_image_properties.png** | Propriétés des images (taille, format) |

### Exemple Output
```
============================================================
CSV METADATA ANALYSIS
============================================================

Shape: (20, 12) (images, columns)

Columns: ['Image ID', 'Image Quality', 'Lesion Presence', 
          'Breast Composition', 'BIRADS_Score', ...]

============================================================
LABEL DISTRIBUTION ANALYSIS
============================================================

Total Images: 20
Lesion Present: 20 (100.0%)
Lesion Absent: 0 (0.0%)

Total Patients: 13
Patients with Lesion: 13
Patients without Lesion: 0

⚠️ Class Balance: IMBALANCED ⚠️
```

### Interprétation

#### 07_metadata_distribution.png
```
┌─────────────────────────────────────┐
│ Lesion Presence: Pie chart          │
├─────────────────────────────────────┤
│ Image Quality: Distribution         │
├─────────────────────────────────────┤
│ BIRADS Score: Bar chart             │
├─────────────────────────────────────┤
│ Breast Composition: Pie chart       │
└─────────────────────────────────────┘
```

**À regarder:**
- Distribution équilibrée? (idéal: ~50/50 Yes/No)
- Qualité images acceptable? (Average/Good > Poor)
- Scores BIRADS couverts? (1, 2, 3, 4, 5)

#### 08_sample_images.png
```
┌──────────────────────────────────────┐
│ [Image 1]  [Image 2]  [Image 3] ...  │
│ Lesion ❌   Lesion ❌   No ✓      ...  │
│                                      │
│ Grille 3×4 = 12 images aléatoires   │
└──────────────────────────────────────┘
```

**À regarder:**
- Apparence raisonnablement variée?
- Contraste acceptable?
- Pas d'artefacts majeurs?

#### 09_label_distribution.png
```
Pie chart 1: Images (100% Lesion actuellement)
Pie chart 2: Patients (13 patients, tous avec lésions)
Bar chart: Images par patient (1-3 images)
Table: Statistiques complètes
```

**À regarder:**
- ⚠️ **IMPORTANT**: Actuellement 100% de lésions détectées!
- Solution: Obtenir des images sans lésions

#### 10_image_properties.png
```
Histogram: Taille fichiers images (KB)
Table: Propriétés preprocessing
  - Shape: [3, 224, 224] ✓
  - Normalized: Yes ✓
  - Range: [0, 1] ✓
```

---

## 🎯 Étape 2: Entraîner avec Visualisations

### Commande
```bash
python train_cv_with_plots.py
```

### Processus
```
1. Load dataset (20 images, 13 patients)
2. 5-fold stratified cross-validation
   └─ Fold 1-5: Train + Validate
3. Génère graphes après chaque epoch
4. Sauvegarde meilleur modèle
5. Crée visualisations finales
```

### Génère
| Graphe | Description |
|--------|-------------|
| **01_loss_curves.png** | Courbes d'apprentissage (train & val) |
| **02_auc_accuracy_curves.png** | AUC et Accuracy par fold |
| **03_confusion_matrices.png** | 5 matrices confusion |
| **04_roc_curves.png** | 5 courbes ROC |
| **05_probability_distribution.png** | Histogramme probabilités |
| **06_cv_summary.png** | Résumé CV (AUC, Acc, Loss) |

### Timeline Exécution
```
0:00 - Start
0:30 - Fold 1 training (20 epochs)
2:00 - Fold 2-5 training
7:00 - All folds done
7:05 - Generate 6 visualizations
7:10 - Save best model
      ✓ Complete!
```

### Graphes Détaillés

#### 01_loss_curves.png
```
LEFT: Training Loss per Fold         RIGHT: Validation Loss per Fold

Loss ↑                                Loss ↑
1.0 │ ╱╲╱╱╱╱                        1.0 │ ╱╱╱╱
    │╱  ╲ ╲ ╲ Fold 1                   │╱╱ Fold 1
0.5 │      ╲ ╲                       0.5 │  Fold 2
    │       ╲ ╲ Fold 2                  │   ...
0.0 └─────────────────→ Epoch       0.0 └─────────────→ Epoch
```

**À regarder:**
- ✓ Loss diminue au fil du temps
- ✗ Loss stagne → Learning rate trop faible
- ✗ Loss augmente → Overfitting
- ✓ Train < Val = bon équilibre

#### 02_auc_accuracy_curves.png
```
LEFT: Validation AUC                RIGHT: Validation Accuracy

AUC ↑                                Acc ↑
1.0 │ ╱╱╱                           1.0 │ ╱╱╱
0.95│╱  ╱ Fold 1                    0.95│╱ Fold 1
0.9 │   ╱ Fold 2                    0.9 │ Fold 2
    │  ╱  ...                           │  ...
0.5 └──────────→ Epoch              0.0 └──────────→ Epoch
```

**À regarder:**
- ✓ AUC > 0.90 = excellent
- ✓ Metrics converge = pas d'overfitting
- ⚠️ Variance élevée entre folds? → Plus de data

#### 03_confusion_matrices.png
```
┌─────────────────────────────────┐
│ Fold 1        │ Fold 2        │
│ ┌───────┐     │ ┌───────┐     │
│ │ TP FP │     │ │ TP FP │     │
│ │ FN TN │     │ │ FN TN │     │
│ └───────┘     │ └───────┘     │
├─────────────────────────────────┤
│ Metrics:                        │
│ Sens: 0.95  Spec: 0.92         │
│ (+ pour chaque fold)            │
└─────────────────────────────────┘

Legend:
TP = True Positive (correct lesion)
FP = False Positive (false alarm)
FN = False Negative (missed)
TN = True Negative (correct no-lesion)
```

**À regarder:**
- ✓ Diagonale principale remplie (TP & TN élevés)
- ✗ Hors-diagonale remplie (erreurs)
- ✗ FN élevé → Modèle rate lésions (dangereux!)
- ✗ FP élevé → Trop d'alarmes (frustrant)

#### 04_roc_curves.png
```
TPR ↑
1.0 │         ╱─────  Fold 1 (AUC=0.95)
    │        ╱
0.5 │       ╱ Random (AUC=0.50)
    │      ╱
0.0 └─────────────→ FPR
    0    0.5  1.0

Mean AUC: 0.936 ± 0.032
```

**À regarder:**
- ✓ Courbe proche du coin top-left = excellent
- ✓ AUC > 0.90 = bon modèle
- ⚠️ AUC ≈ 0.50 = pas mieux que aléatoire

#### 05_probability_distribution.png
```
Freq ↑
  5 │        🔴 Lesion
    │        █
  3 │    ██  █
    │    ██  █
  1 │🟢██████████ 🟢
    │    ██  █
    └──────────────→ Probability
      0   0.3  0.7  1.0
      └──┴──────┴──┘
        Zones seuils
```

**À regarder:**
- ✓ Séparation claire entre classes
- ✗ Chevauchement > 30% = modèle incertain
- ✓ Seuils (0.3, 0.7) séparent classes

#### 06_cv_summary.png
```
Bar charts:
│ Fold1 Fold2 Fold3 Fold4 Fold5  │  AUC        Accuracy     Loss
│  ┃    ┃     ┃     ┃     ┃     │  0.98       0.95         0.05
│  ┃    ┃     ┃     ┃     ┃     │  0.92       0.90         0.08
│  ┃    ┃     ┃     ┃     ┃     │  [Mean]
│  ┃    ┃     ┃     ┃     ┃     │
├────────────────────────────────┤
│ Mean ± Std:
│ AUC: 0.936 ± 0.032
│ Accuracy: 0.908 ± 0.027
│ Loss: 0.063 ± 0.015
```

**À regarder:**
- ✓ Écart-type petit → Résultats stables
- ✗ Écart-type large → Résultats variables
- ✓ Toutes les métriques cohérentes

---

## 📊 Étape 3: Évaluation & Grad-CAM

### Commande
```bash
python evaluate_advanced.py
```

### Génère
```
gradcam_results/
├── gradcam_AB019_lt_axilla.png      # [Original | Heatmap | Overlay | Mask]
├── gradcam_AB019_lt_breast.png
├── gradcam_AB027_lt_breast.png
├── gradcam_AB038_lt_breast.png
└── gradcam_AB050_lt_breast1.png
```

### Interprétation Grad-CAM

```
CAS 1: Détection Correcte ✓
┌─────────────┐
│  Original   │ → Région rouge (Grad-CAM)
│  ┌───────┐  │   coincide avec mass vraie ✓
│  │🔴Mass │  │
│  │visible│  │
│  └───────┘  │
└─────────────┘

CAS 2: Faux Positif ✗
┌─────────────┐
│  Original   │ → Région rouge mais pas d'anomalie
│  Normal img │   = Modèle trompé par artifact
│ 🔴 (artifact)
└─────────────┘

CAS 3: Localisation Imprécise ⚠
┌─────────────┐
│  Original   │ → Région rouge trop large
│  ┌─────────┐│   = Modèle détecte mais mal localisé
│  │🟡 masse ││
│  │ ↕ Large ││   (Peut indiquer petit objet)
│  │heatmap  ││
│  └─────────┘│
└─────────────┘
```

---

## 📈 Récapitulatif: Tous les Graphes

```
DATASET EXPLORATION (explore_data.py)
├─ 07: Métadonnées CSV (Lesion Presence, Quality, BIRADS)
├─ 08: Exemples images (12 aléatoires)
├─ 09: Distribution labels (100% lesions actuellement ⚠️)
└─ 10: Propriétés images (shape, taille fichier)

TRAINING & VALIDATION (train_cv_with_plots.py)
├─ 01: Courbes loss (train & val) - **Le plus important**
├─ 02: AUC & Accuracy par fold
├─ 03: Matrices confusion (5 folds) - **Diagnostic clé**
├─ 04: Courbes ROC (5 folds)
├─ 05: Distribution probabilités (montre seuils)
└─ 06: Résumé CV (barplots moyennes)

INTERPRETABILITY (evaluate_advanced.py)
└─ Grad-CAM: Explainabilité - **Pour cliniciens**
```

---

## 🎯 Checklist d'Interprétation

### Après explore_data.py
- [ ] Dataset bien équilibré? (idéal 50/50)
- [ ] Images de bonne qualité?
- [ ] Métadonnées cohérentes?
- [ ] Aucun fichier manquant?

### Après train_cv_with_plots.py
- [ ] Loss diminue? (pas de plateau)
- [ ] Pas d'overfitting? (train loss ≈ val loss)
- [ ] AUC > 0.90? (bon modèle)
- [ ] Stabilité entre folds? (faible std)
- [ ] Pas trop de FN? (ne rate pas lésions)
- [ ] Probabilities bien séparées?

### Après evaluate_advanced.py
- [ ] Grad-CAM localise bien?
- [ ] Heatmaps coïncident avec vraies lésions?
- [ ] Pas de biais évidents?

---

## ⚠️ Problèmes Courants & Solutions

### Problème: Loss reste élevée
```
Cause: Learning rate trop bas
Fix: Augmenter dans train_cv_with_plots.py:
  learning_rate = 1e-3  (au lieu de 1e-4)
```

### Problème: Overfitting (train loss << val loss)
```
Cause: Pas assez d'augmentation data
Fix: Augmenter dans dataset.py:
  transforms.RandomRotation(30)  (au lieu de 15)
  ColorJitter(brightness=0.2, ...)  (au lieu de 0.1)
```

### Problème: AUC oscille beaucoup
```
Cause: Dataset trop petit (20 images)
Fix: Collecter plus de données (300-500 minimum)
```

### Problème: Toutes les matrices pareilles
```
Cause: Modèle apprend à dire "toujours lesion"
Raison: Dataset 100% lesions (déséquilibré)
Fix: Collecte d'images sans lésions
```

---

## 📊 Recommandations par Métrique

| Métrique | ✓ Bon | ⚠️ Attention | ❌ Mauvais |
|----------|-------|-------------|---------|
| **AUC** | > 0.90 | 0.80-0.90 | < 0.80 |
| **Accuracy** | > 0.85 | 0.75-0.85 | < 0.75 |
| **Sensitivity** | > 0.90 | 0.80-0.90 | < 0.80 |
| **Specificity** | > 0.85 | 0.75-0.85 | < 0.75 |
| **Loss** | < 0.10 | 0.10-0.20 | > 0.20 |
| **Uncertainty %** | 15-25% | 10-30% | < 10% ou > 30% |

---

## 🚀 Workflow Complet (Étape par Étape)

```bash
# Étape 1: Voir ce que vous avez
python explore_data.py
# → Inspecter: results/07-10_*.png

# Étape 2: Entraîner avec monitoring
python train_cv_with_plots.py
# → Inspecter: results/01-06_*.png
# → Si bon AUC: continuer
# → Si mauvais: retour à Étape 1 (collecter plus de données)

# Étape 3: Évaluer et expliquer
python evaluate_advanced.py
# → Inspecter: gradcam_results/*.png
# → Montrer à radiologues
# → Collecter feedback

# Étape 4: Déployer
# → models/best_model_cv.pth prêt
# → Utiliser dans production
```

---

## 📚 Ressources Supplémentaires

- `DOCUMENTATION.md` - Vue d'ensemble du pipeline
- `DOCUMENTATION_DETAILLEE.md` - Toutes les formules & détails
- `model_advanced.py` - Architecture EfficientNet
- `evaluate_advanced.py` - Grad-CAM & explainabilité

---

## ✅ Checklist Projet Final

- [ ] Dataset exploré (07-10)
- [ ] Modèle entraîné (01-06)
- [ ] AUC > 0.90
- [ ] Pas de FN significatifs
- [ ] Grad-CAM explicable
- [ ] Documentation complète
- [ ] Modèle sauvegardé
- [ ] Prêt pour production

---

**Questions? Voir DOCUMENTATION_DETAILLEE.md ou troubleshooting section ci-dessus.**
