# match.py : le grand match entre l'auto-encodeur et le U-Net

import math
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from donnees import creer_chargeurs
from modele import AutoEncodeur, UNet


def note(erreur):
    """Transforme une erreur en note de qualité (PSNR, en dB)."""
    return 10 * math.log10(1 / erreur)


def charger(Modele, fichier, appareil):
    """Crée un cerveau vide, puis y recharge les réglages appris."""
    modele = Modele().to(appareil)
    modele.load_state_dict(torch.load(fichier, map_location=appareil))
    modele.eval()
    return modele


if __name__ == "__main__":
    appareil = "cuda" if torch.cuda.is_available() else "cpu"

    # Les deux champions
    champions = {
        "Auto-encodeur": charger(AutoEncodeur, "debruiteur.pth", appareil),
        "U-Net": charger(UNet, "debruiteur_unet.pth", appareil),
    }

    _, chargeur_controle = creer_chargeurs()

    # ---------- 1. Les notes moyennes sur les 14 000 visages de contrôle ----------
    print("Calcul des notes sur les visages de contrôle, patiente un peu...")
    total_abimees = 0.0
    totaux = {nom: 0.0 for nom in champions}

    with torch.no_grad():
        for abimes, propres in chargeur_controle:
            abimes = abimes.to(appareil)
            propres = propres.to(appareil)
            total_abimees += F.mse_loss(abimes, propres).item()
            # les deux modèles reçoivent EXACTEMENT les mêmes images abîmées
            for nom, modele in champions.items():
                totaux[nom] += F.mse_loss(modele(abimes), propres).item()

    nb_lots = len(chargeur_controle)
    print("\n========== RÉSULTATS DU MATCH ==========")
    print(f"Image abîmée    : {note(total_abimees / nb_lots):.2f} dB")
    for nom in champions:
        print(f"{nom:<15} : {note(totaux[nom] / nb_lots):.2f} dB")
    print("========================================")

    # ---------- 2. Le match en images ----------
    visages = chargeur_controle.dataset
    choix = random.sample(range(len(visages)), 5)

    fig, cases = plt.subplots(4, 5, figsize=(15, 12))
    for colonne, i in enumerate(choix):
        abime, propre = visages[i]
        lignes = [(abime, f"Abîmé\n{note(F.mse_loss(abime, propre).item()):.1f} dB")]

        with torch.no_grad():
            for nom, modele in champions.items():
                repare = modele(abime.unsqueeze(0).to(appareil)).squeeze(0).cpu()
                lignes.append((repare, f"{nom}\n{note(F.mse_loss(repare, propre).item()):.1f} dB"))

        lignes.append((propre, "Original"))

        for ligne, (image, titre) in enumerate(lignes):
            cases[ligne, colonne].imshow(image.permute(1, 2, 0))
            cases[ligne, colonne].set_title(titre)
            cases[ligne, colonne].axis("off")

    plt.tight_layout()
    plt.savefig("match.png", dpi=150)
    plt.show()
    