import os
import glob
import numpy as np
import torch
import cv2
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, ToTensord
)
from monai.data import Dataset, DataLoader
from monai.data.image_reader import PILReader

def monai_pipeline(input_dir, output_dir):
    print("Ejecutando preprocesado con MONAI...")

    image_paths = [
        f for f in glob.glob(os.path.join(input_dir, "**", "*.png"), recursive=True)
        if os.path.isfile(f) and cv2.imread(f) is not None
    ]

    print("Ejemplos de imágenes encontradas:")
    for path in image_paths[:5]:
        print(" -", path)

    if not image_paths:
        print("No se encontraron imágenes válidas.")
        return

    data_dicts = [{"image": path} for path in image_paths]

    print("Primeros diccionarios del dataset:")
    for d in data_dicts[:5]:
        print(d)

    transforms = Compose([
        LoadImaged(keys="image", image_only=True, reader=PILReader()),
        EnsureChannelFirstd(keys="image"),
        ScaleIntensityd(keys="image"),
        Resized(keys="image", spatial_size=(256, 256)),
        ToTensord(keys="image")
    ])

    dataset = Dataset(data=data_dicts, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1)

    os.makedirs(output_dir, exist_ok=True)

    for i, batch in enumerate(dataloader):

        tensor_img = batch["image"][0]
        np_img = tensor_img.cpu().numpy()

        # ✅ Control total de dimensiones
        if np_img.ndim == 3 and np_img.shape[0] == 1:
            # (1, H, W) → (H, W)
            np_img = np_img[0]
        elif np_img.ndim == 3 and np_img.shape[0] in [3, 4]:
            # (3, H, W) o (4, H, W) → (H, W, C)
            np_img = np.transpose(np_img, (1, 2, 0))
        elif np_img.ndim != 2:
            print(f" Forma no compatible: {np_img.shape}")
            continue

        # Normalizamos a 0–255 y convertimos a uint8
        np_img = (np_img * 255).clip(0, 255).astype(np.uint8)

        output_path = os.path.join(output_dir, f"monai_{i}.png")
        print(f"Guardando: {output_path}")
        cv2.imwrite(output_path, np_img)


        # Normalizamos a 0–255 y convertimos a uint8
        np_img = (np_img * 255).clip(0, 255).astype(np.uint8)

        output_path = os.path.join(output_dir, f"monai_{i}.png")
        print(f"Guardando: {output_path}")
        cv2.imwrite(output_path, np_img)