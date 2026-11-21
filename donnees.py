# donnees.py : le "distributeur" de paires (visage abîmé, visage propre)

from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms.functional import to_tensor

DOSSIER = Path("data/images128x128")
FORCE_BRUIT = 0.3


class VisagesBruites(Dataset):
    """Donne une paire : (visage abîmé, visage propre)."""

    def __init__(self, chemins, force=FORCE_BRUIT):
        self.chemins = chemins
        self.force = force

    def __len__(self):
        # combien de visages dans ce distributeur
        return len(self.chemins)

    def __getitem__(self, i):
        # on ouvre le visage numéro i et on le transforme en nombres entre 0 et 1
        propre = to_tensor(Image.open(self.chemins[i]).convert("RGB"))
        # on fabrique un bruit tout neuf et on l'ajoute
        bruit = torch.randn_like(propre) * self.force
        abime = (propre + bruit).clamp(0, 1)
        return abime, propre


def creer_chargeurs(taille_lot=64, part_controle=0.2):
    """Sépare les visages en 2 groupes et prépare un distributeur pour chacun."""
    chemins = sorted(DOSSIER.rglob("*.png"))
    random.Random(42).shuffle(chemins)  # on mélange, toujours de la même façon (graine 42)

    nb_controle = int(len(chemins) * part_controle)  # 20 % de 70 000 = 14 000
    controle = chemins[:nb_controle]       # 20 % mis de côté
    entrainement = chemins[nb_controle:]   # 80 % pour apprendre

    chargeur_entrainement = DataLoader(
        VisagesBruites(entrainement), batch_size=taille_lot,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    chargeur_controle = DataLoader(
        VisagesBruites(controle), batch_size=taille_lot,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    return chargeur_entrainement, chargeur_controle


# ---- Petit test : ce bloc ne s'exécute que si on lance "python donnees.py" ----
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    chargeur_entrainement, chargeur_controle = creer_chargeurs()
    print("Visages pour l'entraînement :", len(chargeur_entrainement.dataset))
    print("Visages pour le contrôle    :", len(chargeur_controle.dataset))

    # on demande un lot au distributeur
    abimes, propres = next(iter(chargeur_entrainement))
    print("Forme d'un lot :", tuple(abimes.shape), "-> 64 images x 3 couleurs x 128 x 128")

    # on affiche 4 paires : abîmé en haut, propre en bas
    fig, cases = plt.subplots(2, 4, figsize=(12, 6))
    for k in range(4):
        cases[0, k].imshow(abimes[k].permute(1, 2, 0))
        cases[0, k].set_title("Question (abîmé)")
        cases[1, k].imshow(propres[k].permute(1, 2, 0))
        cases[1, k].set_title("Réponse (propre)")
        cases[0, k].axis("off")
        cases[1, k].axis("off")
    plt.tight_layout()
    plt.show()