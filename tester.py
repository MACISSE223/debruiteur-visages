# tester.py : on regarde l'IA réparer des visages qu'elle n'a jamais vus

import math
import random
import torch
import matplotlib.pyplot as plt

from donnees import creer_chargeurs
from modele import AutoEncodeur


def note(image, original):
    """La note de qualité (PSNR, en dB) d'une image par rapport à l'originale."""
    erreur = torch.mean((image - original) ** 2).item()
    return 10 * math.log10(1 / erreur)


if __name__ == "__main__":
    appareil = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. On recrée un cerveau vide, puis on y recharge les réglages appris
    modele = AutoEncodeur().to(appareil)
    modele.load_state_dict(torch.load("debruiteur.pth", map_location=appareil))
    modele.eval()  # mode "contrôle" : on ne touche plus aux réglages

    # 2. On récupère les visages du groupe de contrôle (jamais vus par l'IA)
    _, chargeur_controle = creer_chargeurs()
    visages_controle = chargeur_controle.dataset
    choix = random.sample(range(len(visages_controle)), 5)

    # 3. Pour chaque visage choisi : abîmé / réparé / original
    fig, cases = plt.subplots(3, 5, figsize=(15, 9))
    for colonne, i in enumerate(choix):
        abime, propre = visages_controle[i]

        with torch.no_grad():
            # unsqueeze(0) : on fait un "lot" d'une seule image, car le modèle attend des lots
            repare = modele(abime.unsqueeze(0).to(appareil)).squeeze(0).cpu()

        lignes = [
            (abime, f"Abîmé\n{note(abime, propre):.1f} dB"),
            (repare, f"Réparé par l'IA\n{note(repare, propre):.1f} dB"),
            (propre, "Original"),
        ]
        for ligne, (image, titre) in enumerate(lignes):
            cases[ligne, colonne].imshow(image.permute(1, 2, 0))
            cases[ligne, colonne].set_title(titre)
            cases[ligne, colonne].axis("off")

    plt.tight_layout()
    plt.savefig("resultats.png", dpi=150)  # on garde aussi une copie en image
    plt.show()