from PIL import Image
import numpy as np

# Wczytaj oryginał przez PIL (obsługuje wszystko)
img = Image.open(r"data\moje_slady\image\converse.png").convert("L")
img_np = np.array(img)

# Prosta maska Otsu
threshold = 128
mask_np = (img_np > threshold).astype(np.uint8) * 255

# Zapisz maskę
mask_img = Image.fromarray(mask_np)
mask_img.save(r"data\moje_slady\mask\converse.png")
print("Maska zapisana!")
print("Rozmiar:", mask_img.size)