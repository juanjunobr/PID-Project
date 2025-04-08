import os
import glob
import cv2
import numpy as np
import torch
from monai.networks.nets.unet import UNet
import matplotlib.pyplot as plt

MODEL_PATH = "unet_busi.pt"
IMAGE_DIR = "Dataset_BUSI_with_GT/test"
OUTPUT_DIR = "predictions_from_GT"
IMG_SIZE = (256, 256)
USE_CUDA = torch.cuda.is_available()

def predict_with_gt():

    # === PREPARAR DISPOSITIVO Y MODELO ===
    device = torch.device("cuda" if USE_CUDA else "cpu")

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # === PREDECIR TODAS LAS IMÁGENES DEL DIRECTORIO ===
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths = glob.glob(os.path.join(IMAGE_DIR, "**", "*.png"), recursive=True)
    print(f"Encontradas {len(image_paths)} imágenes para predecir.")

    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error cargando: {img_path}")
                continue

            img_resized = cv2.resize(img, IMG_SIZE)
            img_norm = img_resized.astype(np.float32) / 255.0
            img_tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

            pred = torch.sigmoid(model(img_tensor))
            pred_mask = (pred[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255  # binaria

            # Guardar predicción
            filename = os.path.basename(img_path)
            output_path = os.path.join(OUTPUT_DIR, f"mask_{filename}")
            cv2.imwrite(output_path, pred_mask)

            # Guardar comparación visual
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(img_resized, cmap="gray")
            axs[0].set_title("Imagen original")
            axs[1].imshow(pred_mask, cmap="gray")
            axs[1].set_title("Predicción máscara")
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"comp_{filename}"))
            plt.close()


            print(f"Predicción guardada para: {filename}")

    print("\n Predicciones completas.")
