import os
from preprocessing import process_image

def process_batch(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):  # Recorre subdirectorios también
        for filename in files:
            if filename.endswith('.png') or filename.endswith('.jpg'):
                img_path = os.path.join(root, filename)
                print(f"Procesando: {img_path}")
                process_image(img_path, output_dir)  # Guarda TODO en una carpeta de salida

if __name__ == "__main__":
    test_dir = "Dataset_BUSI_with_GT"  # carpeta de entrada con imágenes
    output_dir = "preprocessed"          # carpeta de salida
    process_batch(test_dir, output_dir)
