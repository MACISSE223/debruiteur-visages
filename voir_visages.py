# voir_visages.py : affiche 8 visages pris au hasard dans notre dossier

from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

# 1. L'endroit où sont rangées les images
dossier = Path("data/images128x128")

# 2. On fait la liste de toutes les images (même dans les sous-dossiers)
images = sorted(p for p in dossier.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
print("Nombre d'images trouvées :", len(images))

# 3. On en choisit 8 au hasard
exemples = random.sample(images, 8)

# 4. On les affiche dans une grille de 2 lignes et 4 colonnes
fig, cases = plt.subplots(2, 4, figsize=(10, 5))
for case, chemin in zip(cases.flat, exemples):
    image = Image.open(chemin)
    case.imshow(image)
    case.set_title(f"{chemin.name}\n{image.size[0]}x{image.size[1]} pixels", fontsize=8)
    case.axis("off")  # on cache les graduations autour de l'image

plt.tight_layout()
plt.show()
