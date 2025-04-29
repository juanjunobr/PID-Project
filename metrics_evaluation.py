import os
import cv2
import numpy as np
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)

def iou_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-7)

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=[0, 1]).ravel()
    return tn / (tn + fp + 1e-7)

#Mapeo de nombres de predicciones a GT reales
def get_gt_filename(pred_mask_name):
    match = re.match(r"mask_(benign|malignant|normal)_(\d+)\.png", pred_mask_name)
    if match:
        clase, numero = match.groups()
        return f"{clase} ({int(numero)})_mask.png"
    return None

def evaluate_all(gt_dir, pred_dir):
    pred_files = [f for f in os.listdir(pred_dir) if f.startswith("mask_") and f.endswith(".png")]

    dice_list, iou_list, acc_list, prec_list, rec_list, spec_list = [], [], [], [], [], []

    for pred_file in pred_files:
        try:
            gt_filename = get_gt_filename(pred_file)
            if not gt_filename:
                print(f"Nombre de predicción no válido: {pred_file}")
                continue

            class_name = gt_filename.split()[0]
            gt_path = os.path.join(gt_dir, class_name, gt_filename)
            pred_path = os.path.join(pred_dir, pred_file)

            if not os.path.exists(gt_path):
                print(f"No se encontró GT para {pred_file}")
                continue

            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

            if gt_mask is None or pred_mask is None:
                print(f"Error leyendo: {gt_path} o {pred_path}")
                continue

            # Redimensionar predicción al tamaño del GT si es necesario
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            gt_mask = (gt_mask > 127).astype(np.uint8)
            pred_mask = (pred_mask > 127).astype(np.uint8)

            dice = dice_score(gt_mask, pred_mask)
            iou = iou_score(gt_mask, pred_mask)
            acc = accuracy_score(gt_mask.flatten(), pred_mask.flatten())
            prec = precision_score(gt_mask.flatten(), pred_mask.flatten(), zero_division=0)
            rec = recall_score(gt_mask.flatten(), pred_mask.flatten(), zero_division=0)
            spec = specificity_score(gt_mask, pred_mask)

            dice_list.append(dice)
            iou_list.append(iou)
            acc_list.append(acc)
            prec_list.append(prec)
            rec_list.append(rec)
            spec_list.append(spec)

        except Exception as e:
            print(f"Error procesando {pred_file}: {e}")

    print("\nEvaluación final (predicciones válidas):")
    print(f"Dice Coefficient (DSC): {np.mean(dice_list):.4f}")
    print(f"IoU: {np.mean(iou_list):.4f}")
    print(f"Accuracy: {np.mean(acc_list):.4f}")
    print(f"Precision: {np.mean(prec_list):.4f}")
    print(f"Recall (Sensitivity): {np.mean(rec_list):.4f}")
    print(f"Specificity: {np.mean(spec_list):.4f}")

if __name__ == "__main__":
    gt_dir = "Dataset_BUSI_train"
    pred_dir = "predictions_from_GT"
    evaluate_all(gt_dir, pred_dir)
