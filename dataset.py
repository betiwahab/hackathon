"""Dr. TABA CHABI - Dataset personnalisé pour la classification d'images médicales (Pocus)
dataset.py - Custom Dataset Class for Medical Image Classification with PyTorch"""
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MedicalDataset(Dataset):
    def __init__(self, csv_file, image_dir, mask_dir=None):
        self.df = pd.read_csv(csv_file)
        #supprimer les espaces inutiles au début et à la fin des noms de colonnes d’un DataFrame pandas.
        self.df.columns = self.df.columns.str.strip()

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # 🔥 Nettoyage dataset : on garde uniquement les images existantes
        valid_rows = []

        for _, row in self.df.iterrows():
            image_id = str(row["Image ID"])
            img_path = os.path.join(self.image_dir, image_id)

            if os.path.exists(img_path):
                valid_rows.append(row)

        #convertit valid_rows en DataFrame et recréer les index propre
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        print(f"Dataset loaded: {len(self.df)} valid images")

        # 🔄 Transformations with ImageNet normalization (no augmentation for validation)
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #calcul de la normalisation des canaux d’une image en utilisant les valeurs de moyenne et d’écart type spécifiques à ImageNet.
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Separate transform with augmentation for training
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            #applique une transformation de retournement horizontal aléatoire à une image avec une probabilité de 50%.
            transforms.RandomHorizontalFlip(p=0.5),
            #applique une transformation de rotation aléatoire à une image dans une plage de -15 à 15 degrés.
            transforms.RandomRotation(15),
            #applique une transformation de changement de luminosité, de contraste et de saturation aléatoire à une image avec des facteurs de 0.1.
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            #calcul de la normalisation des canaux d’une image en utilisant les valeurs de moyenne et d’écart type spécifiques à ImageNet.
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        #Ici, on ne traite pas une image RGB classique, mais un mask de segmentation (annotation pixel par pixel).
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    #len() est une méthode spéciale qui doit être définie dans une classe pour permettre à l'objet de cette classe d'être utilisé 
    # avec la fonction len() intégrée de Python.
    def __len__(self):
        return len(self.df)
    #C’est elle qui définit comment récupérer un exemple individuel (image + label + métadonnées) à partir de l’index du dataset. 
    # C’est la méthode la plus importante à implémenter dans une classe de dataset personnalisée.
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_id = str(row["Image ID"])

        # 📌 Image
        img_path = os.path.join(self.image_dir, image_id)
        image = Image.open(img_path).convert("RGB")

        # je fais une conversion des Label de CSV (classification)te en label binaire (1 pour "yes" et 0 pour "no") en fonction de la présence ou 
        # de l’absence d’une lésion dans l’image médicale.
        label = 1 if row["Lesion Presence"].strip().lower() == "yes" else 0

        # � Extract patient ID from image name (assuming format: PATIENTID_...)
        patient_id = image_id.split('_')[0] if '_' in image_id else image_id.split('.')[0]

        # 🔄 Transform image (default to no augmentation, can be overridden)
        transform = getattr(self, 'current_transform', self.img_transform)
        image = transform(image)

        return image, torch.tensor(label, dtype=torch.float32), patient_id, image_id