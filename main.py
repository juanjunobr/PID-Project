import os
import sys
from preprocessing import process_image
from preprocessing_monai import monai_pipeline
from train_unet_monai import train_unet


def process_batch(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.png') or filename.endswith('.jpg'):
                img_path = os.path.join(root, filename)
                print(f"Procesando: {img_path}")
                process_image(img_path, output_dir)

if __name__ == "__main__":
    test_dir = "Dataset_BUSI_with_GT"
    output_dir = "preprocessed"

    mode = "monai"  # Cambiar a "monai" si se quiere usar MONAI

    if mode == "classic":
        process_batch(test_dir, output_dir)
    elif mode == "monai":
        monai_pipeline(test_dir, output_dir)
    elif mode == "train":
        train_unet()
    else:
        print("Modo no reconocido. Usa 'classic', 'monai' o 'train'.")
