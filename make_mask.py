from PIL import Image
import numpy as np

# Wczytaj obraz
img = Image.open(r"data\moje_slady\image\converse_01.png").convert("L")
img_np = np.array(img)

# Próg
threshold = 128

# Odwrócona maska
mask_np = (img_np <= threshold).astype(np.uint8) * 255

# Zapisz maskę
mask_img = Image.fromarray(mask_np)
mask_img.save(r"data\moje_slady\mask\converse_01.png")

print("Maska zapisana!")
print("Rozmiar:", mask_img.size)