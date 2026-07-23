"""
Generowanie syntetycznych, czarno-białych odcisków podeszwy.

WERSJA 3 - dodaje "depth_source": "model" / "shading" / "combined"

Dlaczego: Depth Anything (i podobne sieci) są trenowane na głębi w SKALI SCENY
(metry - odległość do obiektów w kadrze), nie na milimetrowym reliefie płaskiej
powierzchni z bliska. Dla takiej sieci cały ślad często wygląda jak jedna,
niemal płaska płaszczyzna -> stąd bardzo mały kontrast na wyjściu, którego
żaden CLAHE ani próg nie naprawi, bo sygnału tam po prostu nie ma.

"shading" to klasyczne podejście (shape-from-shading / lokalny high-pass):
zakłada, że lokalne różnice jasności na niemal płaskiej, jednolicie
oświetlonej powierzchni SĄ bezpośrednim śladem reliefu (krawędzie rowków
łapią/tracą światło). To o wiele silniejszy sygnał dla tego konkretnego
przypadku niż nauczona sieć do głębi scenowej.

"combined" łączy oba - ogólny kształt z modelu + drobny detal z cieniowania.

Wymagania:
    pip install torch transformers pillow opencv-python numpy

Wejście:
    - obrazy: data\\moje_slady\\image_2_prep
    - maski:  data\\moje_slady\\mask_3_prep
"""

import os
import numpy as np
import torch
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

print(f"Ładowanie modelu {MODEL_NAME} na urządzeniu: {DEVICE} ...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


def estimate_depth_model(image: Image.Image) -> np.ndarray:
    """Głębia z Depth Anything - dobra do ogólnego kształtu, słaba do drobnego reliefu."""
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    return depth.astype(np.float32)


def inpaint_background(gray_u8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Wypełnia obszar POZA maską wartościami wywnioskowanymi z wnętrza maski,
    żeby rozmycie/CLAHE liczone później nie łapały sztucznego skoku jasności
    na granicy ślad->tło (co wygląda jak fałszywy, ostry "relief" na obwodzie).
    """
    background_mask = (~mask).astype(np.uint8) * 255
    if background_mask.sum() == 0:
        return gray_u8  # cała klatka to maska, nic do wypełnienia
    return cv2.inpaint(gray_u8, background_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)


def estimate_relief_from_shading(image: Image.Image, mask: np.ndarray, blur_sigma: float = 15.0) -> np.ndarray:
    """
    Shape-from-shading (uproszczone): lokalna różnica jasności między pikselem
    a jego mocno rozmytym otoczeniem = lokalny relief (krawędzie rowków
    łapią/tracą światło na niemal płaskiej powierzchni).

    To jest w praktyce high-pass filter na luminancji - bardzo czuły na
    subtelne różnice jasności, dokładnie to, czego brakuje modelom depth
    trenowanym na scenach.

    WAŻNE: tło poza maską jest najpierw "inpaintowane" (patrz inpaint_background),
    żeby rozmycie blisko krawędzi maski nie łapało tła jako fałszywego reliefu.
    """
    gray = np.array(image.convert("L"))
    gray_filled = inpaint_background(gray, mask).astype(np.float32)

    ksize = int(blur_sigma * 3) | 1  # nieparzyste
    low_freq = cv2.GaussianBlur(gray_filled, (ksize, ksize), blur_sigma)
    relief = gray_filled - low_freq  # dodatnie = jaśniejsze od otoczenia, ujemne = ciemniejsze
    return relief


def normalize_percentile(values_2d: np.ndarray, mask: np.ndarray,
                          low_p: float = 2.0, high_p: float = 98.0) -> np.ndarray:
    vals = values_2d[mask]
    if vals.size == 0:
        return np.zeros_like(values_2d, dtype=np.float32)
    lo, hi = np.percentile(vals, low_p), np.percentile(vals, high_p)
    norm = (values_2d - lo) / (hi - lo + 1e-8)
    norm = np.clip(norm, 0.0, 1.0)
    norm[~mask] = 0.0
    return norm


def boost_details(depth_norm: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img_u8 = (depth_norm * 255).astype(np.uint8)
    # wypełniamy tło (twarde zera poza maską) zanim policzymy CLAHE, żeby kafelki
    # na granicy maski nie mieszały prawdziwych wartości z zerami tła
    img_filled = inpaint_background(img_u8, mask)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_filled)
    enhanced[~mask] = 0
    return enhanced


def get_relief_map(
    image: Image.Image,
    mask: np.ndarray,
    depth_source: str = "combined",
    shading_blur_sigma: float = 15.0,
    combined_weight_shading: float = 0.6,
) -> np.ndarray:
    """
    Zwraca finalną mapę reliefu (uint8, 0-255) wg wybranego źródła.

    depth_source:
        "model"    - tylko Depth Anything (ogólny kształt, mało detalu)
        "shading"  - tylko lokalne cieniowanie (dużo detalu, brak globalnego kształtu)
        "combined" - połączenie obu (rekomendowane, jeśli "model" wychodzi płasko)
    """
    if depth_source in ("model", "combined"):
        depth_raw = estimate_depth_model(image)
        depth_norm = normalize_percentile(depth_raw, mask)

    if depth_source in ("shading", "combined"):
        shading_raw = estimate_relief_from_shading(image, mask, blur_sigma=shading_blur_sigma)
        shading_norm = normalize_percentile(shading_raw, mask)

    if depth_source == "model":
        combined = depth_norm
    elif depth_source == "shading":
        combined = shading_norm
    else:  # combined
        w = combined_weight_shading
        combined = (1 - w) * depth_norm + w * shading_norm
        combined = np.clip(combined, 0.0, 1.0)

    return boost_details(combined, mask)


def make_synthetic_print(
    depth_img_u8: np.ndarray,
    mask: np.ndarray,
    invert: bool = True,
    method: str = "adaptive",
    fixed_threshold: float = 0.5,
    adaptive_block_size: int = 35,
    adaptive_C: int = 5,
    morph_kernel: int = 1,
) -> np.ndarray:
    img = depth_img_u8.copy()

    if method == "adaptive":
        bs = adaptive_block_size if adaptive_block_size % 2 == 1 else adaptive_block_size + 1
        binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=bs, C=adaptive_C,
        )
    elif method == "otsu":
        masked_pixels = img[mask]
        if masked_pixels.size == 0:
            return np.full_like(img, 255)
        thresh_val, _ = cv2.threshold(masked_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = (img > thresh_val).astype(np.uint8) * 255
    else:  # fixed
        thresh_val = fixed_threshold * 255
        binary = (img > thresh_val).astype(np.uint8) * 255

    if invert:
        binary = 255 - binary
    binary[~mask] = 255

    if morph_kernel and morph_kernel > 1:
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


def process_folder(
    image_folder: str,
    mask_folder: str,
    output_relief_folder: str,
    output_print_folder: str,
    depth_source: str = "combined",
    invert: bool = True,
    threshold_method: str = "adaptive",
    adaptive_block_size: int = 35,
    adaptive_C: int = 5,
    shading_blur_sigma: float = 15.0,
    combined_weight_shading: float = 0.6,
):
    os.makedirs(output_relief_folder, exist_ok=True)
    os.makedirs(output_print_folder, exist_ok=True)

    valid_ext = (".png", ".jpg", ".jpeg")
    filenames = sorted(f for f in os.listdir(image_folder) if f.lower().endswith(valid_ext))

    if not filenames:
        print("Brak obrazów w folderze:", image_folder)
        return

    for filename in filenames:
        image_path = os.path.join(image_folder, filename)
        mask_path = os.path.join(mask_folder, filename)

        if not os.path.exists(mask_path):
            print(f"Pominięto {filename}: brak maski w {mask_folder}")
            continue

        print("Przetwarzanie:", filename)

        image = Image.open(image_path).convert("RGB")
        mask = np.array(Image.open(mask_path).convert("L").resize(image.size, Image.NEAREST)) > 127

        relief_u8 = get_relief_map(
            image, mask,
            depth_source=depth_source,
            shading_blur_sigma=shading_blur_sigma,
            combined_weight_shading=combined_weight_shading,
        )

        print(f"  relief std w masce: {relief_u8[mask].std():.2f} (0-255)")

        Image.fromarray(relief_u8).save(os.path.join(output_relief_folder, filename))

        synthetic_print = make_synthetic_print(
            relief_u8, mask, invert=invert, method=threshold_method,
            adaptive_block_size=adaptive_block_size, adaptive_C=adaptive_C,
        )
        Image.fromarray(synthetic_print).save(os.path.join(output_print_folder, filename))

    print("\nGotowe.")
    print("Mapy reliefu:", output_relief_folder)
    print("Syntetyczne odciski:", output_print_folder)


if __name__ == "__main__":
    image_folder = r"data\moje_slady\image_2_prep"
    mask_folder = r"data\moje_slady\mask_3_prep"

    output_relief_folder = r"data\moje_slady\relief_4"
    output_print_folder = r"data\moje_slady\print_5"

    process_folder(
        image_folder=image_folder,
        mask_folder=mask_folder,
        output_relief_folder=output_relief_folder,
        output_print_folder=output_print_folder,
        depth_source="combined",        # "model" / "shading" / "combined"
        invert=True,
        threshold_method="adaptive",
        adaptive_block_size=35,
        adaptive_C=5,
        shading_blur_sigma=15.0,        # mniejsze = wyłapuje drobniejsze detale
        combined_weight_shading=0.6,    # 0 = tylko model, 1 = tylko shading
    )