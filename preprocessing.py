"""
Preprocessing zdjęć śladów przed estymacją głębi (Depth Anything V2 itp.).

Cel: 256x256 to mało pikseli na subtelne cieniowanie reliefu odcisku.
Ten skrypt:
  1. przycina obraz do bounding boxa maski (+ margines) - nie marnujemy
     rozdzielczości na tło,
  2. powiększa wynik (upscaling) do większej "kanwy" przed depth estimation,
  3. spłaszcza globalne, nierówne oświetlenie, zachowując lokalne cienie
     (to one niosą informację o głębokości wgłębień),
  4. odszumia z zachowaniem krawędzi (bilateral filter),
  5. wzmacnia lokalny kontrast (CLAHE) i ostrość (unsharp mask).

Wymagania:
    pip install opencv-python pillow numpy

Wejście:
    - obrazy: data\\moje_slady\\image_2
    - maski:  data\\moje_slady\\mask_3

Wyjście (podajcie te foldery jako image_folder/mask_folder
do skryptu generate_synthetic_prints.py):
    - przetworzone obrazy: data\\moje_slady\\image_2_prep
    - dopasowane maski:    data\\moje_slady\\mask_3_prep
"""

import os
import numpy as np
import cv2
from PIL import Image


def crop_to_mask_bbox(image: np.ndarray, mask: np.ndarray, margin_ratio: float = 0.12):
    """
    Przycina obraz i maskę do bounding boxa maski, z marginesem.
    margin_ratio: procent szerokości/wysokości bboxa dodawany jako margines.
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return image, mask  # pusta maska - nic nie robimy

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    h, w = mask.shape
    box_h, box_w = (y1 - y0), (x1 - x0)
    margin_y = int(box_h * margin_ratio) + 1
    margin_x = int(box_w * margin_ratio) + 1

    y0 = max(0, y0 - margin_y)
    y1 = min(h, y1 + margin_y)
    x0 = max(0, x0 - margin_x)
    x1 = min(w, x1 + margin_x)

    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def upscale(image: np.ndarray, target_size: int = 768) -> np.ndarray:
    """
    Powiększa obraz tak, by dłuższy bok miał target_size px.
    Lanczos radzi sobie lepiej niż bicubic przy dużym powiększeniu drobnej faktury.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    if scale <= 1.0:
        # nie pomniejszamy - jeśli obraz już jest większy, zostawiamy bez zmian
        return image
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def upscale_mask(mask: np.ndarray, target_shape) -> np.ndarray:
    """Skaluje maskę do docelowego rozmiaru z interpolacją nearest (bez rozmywania krawędzi)."""
    h, w = target_shape[:2]
    mask_u8 = (mask.astype(np.uint8)) * 255
    resized = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
    return resized > 127


def flatten_illumination(gray: np.ndarray, blur_sigma_ratio: float = 0.15) -> np.ndarray:
    """
    Usuwa duży, wolnozmienny gradient oświetlenia (np. cień z jednej strony kadru),
    zachowując lokalne cienie na krawędziach reliefu.

    Działanie: dzielimy obraz przez jego mocno rozmytą wersję (illumination map),
    co jest standardowym trikiem do korekcji nierównego oświetlenia.
    """
    h, w = gray.shape[:2]
    sigma = max(h, w) * blur_sigma_ratio
    ksize = int(sigma) | 1  # musi być nieparzyste
    illumination = cv2.GaussianBlur(gray, (ksize, ksize), 0).astype(np.float32) + 1e-3

    flat = gray.astype(np.float32) / illumination
    flat = flat / flat.max() * 255.0
    return np.clip(flat, 0, 255).astype(np.uint8)


def denoise_preserve_edges(gray: np.ndarray) -> np.ndarray:
    """Bilateral filter - odszumia, ale nie rozmywa krawędzi reliefu."""
    return cv2.bilateralFilter(gray, d=7, sigmaColor=35, sigmaSpace=35)


def enhance_local_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE - adaptacyjne wyrównanie histogramu w małych kafelkach, wydobywa lokalny kontrast."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def unsharp_mask(gray: np.ndarray, amount: float = 1.0, radius: int = 3) -> np.ndarray:
    """Wyostrzenie - podkreśla drobne krawędzie cieniowania."""
    blurred = cv2.GaussianBlur(gray, (0, 0), radius)
    sharpened = cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess_image(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    target_size: int = 768,
    do_flatten: bool = True,
    do_denoise: bool = True,
    do_clahe: bool = True,
    do_sharpen: bool = True,
):
    """
    Pełny pipeline preprocessingu. Zwraca (obraz_RGB_3-kanałowy, maska_bool)
    gotowe do wrzucenia do depth estimation.
    """
    cropped_img, cropped_mask = crop_to_mask_bbox(image_bgr, mask)

    up_img = upscale(cropped_img, target_size=target_size)
    up_mask = upscale_mask(cropped_mask, up_img.shape)

    gray = cv2.cvtColor(up_img, cv2.COLOR_BGR2GRAY)

    if do_flatten:
        gray = flatten_illumination(gray)
    if do_denoise:
        gray = denoise_preserve_edges(gray)
    if do_clahe:
        gray = enhance_local_contrast(gray)
    if do_sharpen:
        gray = unsharp_mask(gray)

    # Depth Anything (i większość modeli monocular depth) przyjmuje obraz RGB
    # 3-kanałowy - konwertujemy przetworzony szary z powrotem do 3 kanałów,
    # zachowując wzmocniony kontrast na wszystkich kanałach.
    result_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return result_rgb, up_mask


def process_folder(
    image_folder: str,
    mask_folder: str,
    output_image_folder: str,
    output_mask_folder: str,
    target_size: int = 768,
):
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

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

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"  Nie udało się wczytać obrazu: {image_path}")
            continue

        mask = np.array(Image.open(mask_path).convert("L")) > 127
        if mask.shape != image_bgr.shape[:2]:
            mask = cv2.resize(
                (mask.astype(np.uint8)) * 255,
                (image_bgr.shape[1], image_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 127

        result_rgb, result_mask = preprocess_image(image_bgr, mask, target_size=target_size)

        out_name = os.path.splitext(filename)[0] + ".png"  # PNG - bez strat po tylu operacjach

        Image.fromarray(result_rgb).save(os.path.join(output_image_folder, out_name))
        Image.fromarray((result_mask.astype(np.uint8)) * 255).save(
            os.path.join(output_mask_folder, out_name)
        )

    print("\nGotowe.")
    print("Przetworzone obrazy:", output_image_folder)
    print("Dopasowane maski:", output_mask_folder)


if __name__ == "__main__":
    image_folder = r"data\moje_slady\image"
    mask_folder = r"data\moje_slady\mask"

    output_image_folder = r"data\moje_slady\image_2_prep"
    output_mask_folder = r"data\moje_slady\mask_3_prep"

    process_folder(
        image_folder=image_folder,
        mask_folder=mask_folder,
        output_image_folder=output_image_folder,
        output_mask_folder=output_mask_folder,
        target_size=768,  
    )