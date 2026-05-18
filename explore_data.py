"""
Dr. TABA CHABI - Data Exploration et Visualisation pour le projet de classification d'images médicales (Pocus)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter

from dataset import MedicalDataset


def explore_csv_metadata(csv_file='data/ABreast-Classification.csv', save_dir='results'):
    """Analyser et visualiser les métadonnées CSV"""
    os.makedirs(save_dir, exist_ok=True)
    
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    
    print("="*60)
    print("CSV METADATA ANALYSIS")
    print("="*60)
    print(f"\nShape: {df.shape} (images, columns)")
    print(f"\nColumns: {list(df.columns)}\n")
    print(df.head(10))
    
    # === Distribution des labels ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Lesion Presence
    #Ici, on visualise la distribution de la présence de lésions dans les images du dataset.
    if 'Lesion Presence' in df.columns:
        lesion_counts = df['Lesion Presence'].value_counts()
        #création de l'histogramme à barres pour la distribution de la présence de lésions, 
        # avec des couleurs personnalisées, des bordures noires, une transparence de 0.7 et une épaisseur de ligne de 2.
        axes[0, 0].bar(lesion_counts.index, lesion_counts.values, color=['#FF6B6B', '#4ECDC4'], 
                      edgecolor='black', alpha=0.7, linewidth=2)
        axes[0, 0].set_title('Lesion Presence Distribution', fontsize=12, fontweight='bold')
        #indique que l’axe vertical représente le nombre d’échantillons.
        axes[0, 0].set_ylabel('Count', fontsize=11)
        for i, v in enumerate(lesion_counts.values):
            axes[0, 0].text(i, v + 0.2, str(v), ha='center', fontsize=11, fontweight='bold')
    
    # 2. Image Quality
    if 'Image Quality' in df.columns:
        quality_counts = df['Image Quality'].value_counts()
        axes[0, 1].barh(quality_counts.index, quality_counts.values, color='steelblue', 
                       edgecolor='black', alpha=0.7, linewidth=2)
        axes[0, 1].set_title('Image Quality Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Count', fontsize=11)
        for i, v in enumerate(quality_counts.values):
            axes[0, 1].text(v + 0.1, i, str(v), va='center', fontsize=10)
    
    # 3. BIRADS Score
    #La colonne BIRADS_Score représente le score BI-RADS (Breast Imaging Reporting and Data System), un standard clinique utilisé en 
    # imagerie mammaire pour évaluer le niveau de suspicion de lésions cancéreuses.
    """
            Explication médicale simple
            == BI-RADS 1–2 : Normal ou sain
                Aucun cancer suspecté
                Suivi classique
                Pas d’intervention
                par xemple :  kystes simples, tissu normal

            == BI-RADS 3 : probablement sain
                Très faible risque (<2%)
                Surveillance à court terme (6 mois)
                  Objectif :  éviter biopsies inutiles

            == BI-RADS 4 : suspect
                Zone intermédiaire
                nécessite biopsie 
                     Sous-catégories (important en clinique) :    4A : faible suspicion, 4B : suspicion modérée, 4C : forte suspicion
            == BI-RADS 5 : très suspect
                Très forte probabilité de cancer: >95%                Biopsie urgente
            == BI-RADS 6 : cancer confirmé
                Résultat déjà validé par histologie: biopsie positive
                utilisé pour suivi traitement
    """
    if 'BIRADS_Score' in df.columns:
        birads_counts = df['BIRADS_Score'].value_counts().sort_index()
        axes[1, 0].bar(birads_counts.index.astype(str), birads_counts.values, 
                      color='coral', edgecolor='black', alpha=0.7, linewidth=2)
        axes[1, 0].set_title('BIRADS Score Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Count', fontsize=11)
        axes[1, 0].set_xlabel('BIRADS Score', fontsize=11)
        for i, v in enumerate(birads_counts.values):
            axes[1, 0].text(i, v + 0.1, str(v), ha='center', fontsize=10)
    
    # 4. Breast Composition
    #La breast composition : décrit la proportion de tissu glandulaire(dense) et graisseux “fatty” dans le sein.
    """
    #Sein graisseux (A-B): 
            meilleure visibilité des lésions
            diagnostic plus facile
            faible risque de masquage
    #Sein dense (C-D):
            les lésions peuvent être cachées
            risque de faux négatifs plus élevé
            nécessite parfois IRM ou échographie complémentaire
    #Le breast composition est un facteur important dans l’évaluation du risque de cancer du sein et la stratégie de dépistage.
    # Les femmes avec des seins denses peuvent bénéficier de méthodes d’imagerie supplémentaires pour une détection plus précoce."""
    """
    
    """
    if 'Breast Composition' in df.columns:
        composition_counts = df['Breast Composition'].value_counts()
        axes[1, 1].pie(composition_counts.values, labels=composition_counts.index, autopct='%1.1f%%',
                      colors=['#FFD93D', '#6BCB77', '#4D96FF', '#FF6B9D'], startangle=90)
        axes[1, 1].set_title('Breast Composition Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '07_metadata_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {save_dir}/07_metadata_distribution.png")


def visualize_sample_images(dataset, num_samples=12, save_dir='results'):
    """Visualiser des échantillons d'images du dataset"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("VISUALIZING SAMPLE IMAGES")
    print("="*60)
    
    # Sélectionner des indices aléatoires
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Calculer grille
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 14))
    axes = axes.flatten()
    
    # Denormalisation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for plot_idx, data_idx in enumerate(indices):
        image, label, patient_id, image_id = dataset[data_idx]
        
        # Denormalize
        image_np = image.permute(1, 2, 0).numpy()
        image_np = std * image_np + mean
        image_np = np.clip(image_np, 0, 1)
        
        # Label text
        label_text = "Lesion" if label.item() == 1 else "No Lesion"
        label_color = 'red' if label.item() == 1 else 'green'
        
        # Plot
        axes[plot_idx].imshow(image_np)
        axes[plot_idx].set_title(f'{image_id}\n{label_text}', 
                                fontsize=10, color=label_color, fontweight='bold')
        axes[plot_idx].axis('off')
    
    # Masquer subplots inutilisés
    for plot_idx in range(len(indices), len(axes)):
        axes[plot_idx].set_visible(False)
    
    plt.suptitle(f'Sample POCUS Images (n={len(indices)})', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '08_sample_images.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_dir}/08_sample_images.png")


def analyze_label_distribution(dataset, save_dir='results'):
    """Analyser distribution des labels"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("="*60)
    
    labels = []
    patient_labels = {}
    image_counts_per_patient = Counter()
    
    for idx in range(len(dataset)):
        _, label, patient_id, _ = dataset[idx]
        labels.append(label.item())
        patient_labels[patient_id] = label.item()
        image_counts_per_patient[patient_id] += 1
    
    labels = np.array(labels)
    
    # Statistiques
    lesion_count = np.sum(labels == 1)
    no_lesion_count = np.sum(labels == 0)
    lesion_rate = lesion_count / len(labels)
    
    print(f"\nTotal Images: {len(labels)}")
    print(f"Lesion Present: {lesion_count} ({lesion_rate*100:.1f}%)")
    print(f"Lesion Absent: {no_lesion_count} ({(1-lesion_rate)*100:.1f}%)")
    print(f"\nTotal Patients: {len(patient_labels)}")
    
    patient_lesion_counts = sum(1 for v in patient_labels.values() if v == 1)
    print(f"Patients with Lesion: {patient_lesion_counts}")
    print(f"Patients without Lesion: {len(patient_labels) - patient_lesion_counts}")
    
    # === VISUALIZATIONS ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Pie chart - Images
    axes[0, 0].pie([no_lesion_count, lesion_count], 
                   labels=[f'No Lesion\n({no_lesion_count})', f'Lesion\n({lesion_count})'],
                   autopct='%1.1f%%', colors=['#4ECDC4', '#FF6B6B'],
                   startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[0, 0].set_title('Image Label Distribution', fontsize=12, fontweight='bold')
    
    # 2. Pie chart - Patients
    axes[0, 1].pie([len(patient_labels) - patient_lesion_counts, patient_lesion_counts],
                   labels=[f'No Lesion\n({len(patient_labels) - patient_lesion_counts})', 
                          f'Lesion\n({patient_lesion_counts})'],
                   autopct='%1.1f%%', colors=['#4ECDC4', '#FF6B6B'],
                   startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[0, 1].set_title('Patient Distribution', fontsize=12, fontweight='bold')
    
    # 3. Images per patient
    patient_ids = list(image_counts_per_patient.keys())
    image_counts = list(image_counts_per_patient.values())
    colors = ['#FF6B6B' if patient_labels[p] == 1 else '#4ECDC4' for p in patient_ids]
    
    axes[1, 0].barh(patient_ids, image_counts, color=colors, edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Images per Patient', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Number of Images', fontsize=11)
    axes[1, 0].invert_yaxis()
    for i, v in enumerate(image_counts):
        axes[1, 0].text(v + 0.05, i, str(v), va='center', fontsize=9)
    
    # 4. Statistics box
    stats_text = f"""
    DATASET STATISTICS
    
    Total Images: {len(labels)}
    Total Patients: {len(patient_labels)}
    
    Lesion Images: {lesion_count} ({lesion_rate*100:.1f}%)
    No-Lesion Images: {no_lesion_count} ({(1-lesion_rate)*100:.1f}%)
    
    Avg Images/Patient: {np.mean(image_counts):.1f}
    Max Images/Patient: {np.max(image_counts)}
    Min Images/Patient: {np.min(image_counts)}
    
    Class Balance: {"IMBALANCED ⚠" if abs(lesion_rate - 0.5) > 0.2 else "BALANCED ✓"}
    """
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '09_label_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {save_dir}/09_label_distribution.png")


def analyze_image_properties(dataset, save_dir='results'):
    """Analyser propriétés des images (size, channels, etc.)"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("IMAGE PROPERTIES ANALYSIS")
    print("="*60)
    
    image_shapes = []
    image_sizes = []
    
    # Charger images originales pour statistiques
    for idx in range(len(dataset)):
        image, _, _, image_id = dataset[idx]
        image_shapes.append(image.shape)
        
        # Charger image originale pour taille fichier
        image_path = os.path.join(dataset.image_dir, image_id)
        if os.path.exists(image_path):
            size_kb = os.path.getsize(image_path) / 1024
            image_sizes.append(size_kb)
    
    image_shapes = np.array(image_shapes)
    
    print(f"\nImage Shapes (after preprocessing): {image_shapes[0]}")
    print(f"All shapes uniform: {np.all(image_shapes == image_shapes[0])}")
    print(f"\nImage Sizes (on disk):")
    print(f"  Mean: {np.mean(image_sizes):.1f} KB")
    print(f"  Min: {np.min(image_sizes):.1f} KB")
    print(f"  Max: {np.max(image_sizes):.1f} KB")
    print(f"  Total: {np.sum(image_sizes):.1f} KB")
    
    # === VISUALIZATION ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Image sizes
    axes[0].hist(image_sizes, bins=10, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(image_sizes), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(image_sizes):.1f} KB')
    axes[0].set_xlabel('File Size (KB)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Image File Size Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 2. Properties table
    props_text = f"""
    IMAGE PROPERTIES (After Preprocessing)
    
    Shape: {image_shapes[0]}
    Channels: {image_shapes[0][0]}
    Height: {image_shapes[0][1]}
    Width: {image_shapes[0][2]}
    
    Normalized: Yes (ImageNet stats)
    Range: [0, 1] per channel
    
    FILE SIZE STATISTICS
    Mean: {np.mean(image_sizes):.1f} KB
    Min: {np.min(image_sizes):.1f} KB
    Max: {np.max(image_sizes):.1f} KB
    Total: {np.sum(image_sizes):.1f} KB
    """
    
    axes[1].text(0.05, 0.95, props_text, transform=axes[1].transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '10_image_properties.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved: {save_dir}/10_image_properties.png")


def create_data_exploration_report(save_dir='results'):
    """Créer rapport d'exploration complète"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("DATA EXPLORATION COMPLETE")
    print("="*60)
    print(f"\nAll visualizations saved to: {save_dir}/")
    print("\nGenerated files:")
    print("  - 07_metadata_distribution.png")
    print("  - 08_sample_images.png")
    print("  - 09_label_distribution.png")
    print("  - 10_image_properties.png")


def main():
    print("="*60)
    print("STARTING DATA EXPLORATION")
    print("="*60)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = MedicalDataset(
        csv_file="data/ABreast-Classification.csv",
        image_dir="data/images",
        mask_dir="data/masks"
    )
    
    # Analyses
    explore_csv_metadata(save_dir='results')
    visualize_sample_images(dataset, num_samples=12, save_dir='results')
    analyze_label_distribution(dataset, save_dir='results')
    analyze_image_properties(dataset, save_dir='results')
    
    # Rapport
    create_data_exploration_report(save_dir='results')


if __name__ == "__main__":
    main()