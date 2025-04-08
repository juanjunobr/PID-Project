import os
import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy.fftpack import fft2, ifft2
import matplotlib.pyplot as plt

def butterworth_high_pass(img, cutoff=30, order=4):
    rows, cols = img.shape
    u = np.array(range(rows)) - rows / 2
    v = np.array(range(cols)) - cols / 2
    u, v = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt(u ** 2 + v ** 2)
    H = 1 / (1 + (cutoff / (D + 1e-5)) ** (2 * order))
    img_fft = fft2(img)
    img_fft_shift = np.fft.fftshift(img_fft)
    img_filtered = img_fft_shift * H
    img_ifft = ifft2(np.fft.ifftshift(img_filtered))
    return np.abs(img_ifft).astype(np.uint8)

def adaptive_median_filter(img, kernel_size=5):
    return cv2.medianBlur(img, kernel_size)

def first_order_features(img):
    mean = cv2.blur(img, (5, 5))
    ent = entropy(img, disk(5))
    ent = ((ent - ent.min()) / (ent.max() - ent.min()) * 255).astype(np.uint8)
    return mean, ent

def second_order_features_fallback(img):
    # Fallback usando operadores espaciales básicos en lugar de greycomatrix
    edges = cv2.Canny(img, 50, 150)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    return edges, laplacian

def fuse_channels(ch1, ch2, ch3):
    return cv2.merge((ch1, ch2, ch3))


def process_image(img_path, output_dir):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    base = os.path.splitext(os.path.basename(img_path))[0]

    # Traditional filters
    butter = butterworth_high_pass(img)
    median = adaptive_median_filter(img)

    # First-order features
    mean, ent = first_order_features(img)
    first_order_fused = fuse_channels(img, mean, ent)

    # Second-order features (simulada sin greycomatrix)
    auto, homog = second_order_features_fallback(img)
    second_order_fused = fuse_channels(img, auto, homog)

    # Save all results
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f"{output_dir}/{base}_butterworth.png", butter)
    cv2.imwrite(f"{output_dir}/{base}_median.png", median)
    cv2.imwrite(f"{output_dir}/{base}_first_order.png", first_order_fused)
    cv2.imwrite(f"{output_dir}/{base}_second_order.png", second_order_fused)

    # Plot an example
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    axs[0].imshow(img, cmap='gray'); axs[0].set_title("Original")
    axs[1].imshow(butter, cmap='gray'); axs[1].set_title("Butterworth")
    axs[2].imshow(median, cmap='gray'); axs[2].set_title("Median")
    axs[3].imshow(first_order_fused[..., ::-1]); axs[3].set_title("First-order")
    axs[4].imshow(second_order_fused[..., ::-1]); axs[4].set_title("Second-order")
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{base}_comparison.png")
    plt.close()
