import cv2
import numpy as np
from PIL import Image
import os

from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained(
    "facebook/sam2.1-hiera-tiny",
    device="cpu"
)

fg_points = []
bg_points = []

def mouse_callback(event, x, y, flags, param):
    global fg_points, bg_points

    if event == cv2.EVENT_LBUTTONDOWN:
        fg_points.append((x, y))
        print("FG:", (x, y))

    if event == cv2.EVENT_RBUTTONDOWN:
        bg_points.append((x, y))
        print("BG:", (x, y))


def segment_with_points(image_path, save_path):
    global fg_points, bg_points
    fg_points = []
    bg_points = []

    # Wczytaj obraz
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Wyświetl obraz
    cv2.namedWindow("Wybierz punkty (LPM=FG, PPM=BG, ENTER=segmentacja)")
    cv2.setMouseCallback("Wybierz punkty (LPM=FG, PPM=BG, ENTER=segmentacja)", mouse_callback)

    while True:
        img_show = image.copy()

        # Rysowanie punktów FG
        for (x, y) in fg_points:
            cv2.circle(img_show, (x, y), 5, (0, 255, 0), -1)

        # Rysowanie punktów BG
        for (x, y) in bg_points:
            cv2.circle(img_show, (x, y), 5, (0, 0, 255), -1)

        cv2.imshow("Wybierz punkty (LPM=FG, PPM=BG, ENTER=segmentacja)", img_show)
        key = cv2.waitKey(1)

        # ENTER segmentacja
        if key == 13:
            break

        # ESC → wyjście
        if key == 27:
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    # Sprawdzenie minimalnej liczby punktów FG
    if len(fg_points) < 2:
        print("Musisz wybrać minimum 2 punkty FG!")
        return

    # Przygotowanie punktów
    all_points = np.array(fg_points + bg_points)
    all_labels = np.array([1] * len(fg_points) + [0] * len(bg_points))

    # Segmentacja SAM2
    predictor.set_image(image_rgb)

    masks, scores, _ = predictor.predict(
        point_coords=all_points,
        point_labels=all_labels,
        multimask_output=True,
    )

    best_mask = masks[np.argmax(scores)].astype(np.uint8) * 255

    
    mask_img = Image.fromarray(best_mask)
    mask_img.save(save_path)

    print("✔ Zapisano maskę:", save_path)


def segment_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            in_path = os.path.join(input_folder, filename)
            out_path = os.path.join(output_folder, filename)

            print("\n➡ Segmentuję:", filename)
            print("LPM = foreground, PPM = background, ENTER = segmentacja")

            segment_with_points(in_path, out_path)

    print("\n Wszystkie maski zapisane!")


input_folder = r"data\moje_slady\image"
output_folder = r"data\moje_slady\mask_2"

segment_folder(input_folder, output_folder)
