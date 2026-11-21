# bruit.py : on prend un visage et on l'abîme de plus en plus

from pathlib import Path
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_tensor

# 1. On choisit un visage au hasard
dossier = Path("data/images128x128")
images = sorted(dossier.rglob("*.png"))
chemin = random.choice(images)

# 2. On transforme l'image en tableau de nombres entre 0 et 1
#    (PyTorch appelle ce genre de tableau un "tenseur")
propre = to_tensor(Image.open(chemin).convert("RGB"))
print("Forme du tableau :", tuple(propre.shape), "-> 3 couleurs x 128 x 128 pixels")
print("Plus petit nombre :", round(propre.min().item(), 2))
print("Plus grand nombre :", round(propre.max().item(), 2))

# 3. On essaie plusieurs forces de bruit
forces = [0.0, 0.1, 0.2, 0.3, 0.5]

fig, cases = plt.subplots(1, len(forces), figsize=(15, 3.5))
for case, force in zip(cases, forces):
    bruit = torch.randn_like(propre) * force    # des petits nombres au hasard
    abime = (propre + bruit).clamp(0, 1)        # on les ajoute, et on reste entre 0 et 1
    case.imshow(abime.permute(1, 2, 0))         # on remet les couleurs à la fin pour l'affichage
    case.set_title("Image propre" if force == 0 else f"Force du bruit = {force}")
    case.axis("off")

plt.suptitle(f"Le visage {chemin.name}, de plus en plus abîmé")
plt.tight_layout()
plt.show()