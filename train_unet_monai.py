import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from monai.data import DataLoader
from monai.networks.nets.unet import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from torch.utils.data import Dataset as TorchDataset

# === CONFIGURACIÓN ===
PREPROC_TAG = "_butterworth"  # Cambia por "_median", "_butterworth", etc.
PREPROCESSED_DIR = "preprocessed"  # Carpeta donde están las imágenes preprocesadas

GT_MASK_DIR = "Dataset_BUSI_train"  # Ground truth original
OUTPUT_DIR = f"predicted_masks{PREPROC_TAG}"
MODEL_PATH = f"unet{PREPROC_TAG}.pt"

IMG_SIZE = (256, 256)
EPOCHS = 30
BATCH_SIZE = 2
USE_CUDA = torch.cuda.is_available()

set_determinism(42)

def get_image_mask_pairs():
    images, masks = [], []
    for filename in os.listdir(PREPROCESSED_DIR):
        if PREPROC_TAG not in filename or not filename.endswith(".png"):
            continue
        base = filename.replace(PREPROC_TAG, "").replace(".png", "")  # quitar sufijo y extensión

        # Buscar máscara en Dataset_BUSI_train
        for cls in ["benign", "malignant", "normal"]:
            mask_path = os.path.join(GT_MASK_DIR, cls, f"{base}_mask.png")
            if os.path.exists(mask_path):
                images.append(os.path.join(PREPROCESSED_DIR, filename))
                masks.append(mask_path)
                break
    return images, masks

class BUSIDataset(TorchDataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        print(f"[DEBUG] Usando imagen: {self.image_paths[idx]} — Máscara: {self.mask_paths[idx]}")  # 👈 AÑADIDO

        img = cv2.resize(img, IMG_SIZE)
        mask = cv2.resize(mask, IMG_SIZE)

        img = img.astype(np.float32) / 255.0
        mask = (mask.astype(np.float32) > 0).astype(np.float32)

        return {
            "img": torch.tensor(img).unsqueeze(0),    # (1, H, W)
            "seg": torch.tensor(mask).unsqueeze(0)
        }

def train_unet():
    images, masks = get_image_mask_pairs()
    if len(images) == 0:
        raise ValueError("No se encontraron imágenes preprocesadas con el sufijo especificado.")

    dataset = BUSIDataset(images, masks)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if USE_CUDA else "cpu")

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2
    ).to(device)

    loss_fn = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch in loader:
            x = batch["img"].to(device)
            y = batch["seg"].to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✅ Modelo UNet guardado como '{MODEL_PATH}'")

    # Guardar predicciones de muestra
    model.eval()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch["img"].to(device)
            y = batch["seg"].to(device)
            pred = torch.sigmoid(model(x)) > 0.5

            for j in range(x.shape[0]):
                img_np = x[j][0].cpu().numpy()
                mask_np = y[j][0].cpu().numpy()
                pred_np = pred[j][0].cpu().numpy()

                fig, axs = plt.subplots(1, 3, figsize=(10, 3))
                axs[0].imshow(img_np, cmap="gray")
                axs[0].set_title("Imagen")
                axs[1].imshow(mask_np, cmap="gray")
                axs[1].set_title("Máscara Real")
                axs[2].imshow(pred_np, cmap="gray")
                axs[2].set_title("Predicción")

                for ax in axs:
                    ax.axis('off')

                plt.tight_layout()
                plt.savefig(f"{OUTPUT_DIR}/comparison_{i}_{j}.png")
                plt.close()

            if i >= 4:  # solo guardar algunas muestras
                break
