# entrainer.py : on apprend à l'IA à débruiter les visages

import math
import time
import torch
import torch.nn as nn

from donnees import creer_chargeurs
from modele import AutoEncodeur, UNet

NB_EPOQUES = 10          # combien de fois l'IA verra tous les visages d'entraînement
TAUX_APPRENTISSAGE = 1e-3  # la taille des petits coups de tournevis sur les réglages
FICHIER_MODELE = "debruiteur_unet.pth"


def psnr(erreur):
    """Transforme une erreur en note de qualité (en dB). Plus c'est haut, mieux c'est."""
    return 10 * math.log10(1 / erreur)


if __name__ == "__main__":
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    print("Appareil utilisé :", appareil)

    chargeur_entrainement, chargeur_controle = creer_chargeurs()
    modele = UNet().to(appareil)
    mesure_erreur = nn.MSELoss()  # erreur = moyenne des (pixel dessiné - vrai pixel)²
    optimiseur = torch.optim.Adam(modele.parameters(), lr=TAUX_APPRENTISSAGE)

    record = float("inf")
    nb_lots = len(chargeur_entrainement)

    for epoque in range(1, NB_EPOQUES + 1):
        debut = time.time()

        # ---------- 1. Entraînement ----------
        modele.train()
        total = 0.0
        for i, (abimes, propres) in enumerate(chargeur_entrainement, start=1):
            abimes = abimes.to(appareil, non_blocking=True)
            propres = propres.to(appareil, non_blocking=True)

            reparees = modele(abimes)                 # 👀 elle essaie de réparer
            erreur = mesure_erreur(reparees, propres) # 📏 on mesure son erreur

            optimiseur.zero_grad()
            erreur.backward()                         # 🔧 on calcule dans quel sens tourner les réglages
            optimiseur.step()                         # 🔧 on les tourne un petit peu

            total += erreur.item()
            if i % 100 == 0:
                print(f"   Époque {epoque} - lot {i}/{nb_lots} - erreur {erreur.item():.4f}")

        erreur_entrainement = total / nb_lots

        # ---------- 2. Contrôle surprise ----------
        modele.eval()
        total_reparees = 0.0
        total_abimees = 0.0
        with torch.no_grad():  # pendant le contrôle, on ne touche pas aux réglages
            for abimes, propres in chargeur_controle:
                abimes = abimes.to(appareil, non_blocking=True)
                propres = propres.to(appareil, non_blocking=True)
                total_reparees += mesure_erreur(modele(abimes), propres).item()
                total_abimees += mesure_erreur(abimes, propres).item()

        erreur_controle = total_reparees / len(chargeur_controle)
        erreur_abimees = total_abimees / len(chargeur_controle)
        duree = time.time() - debut

        print(f"\n=== Époque {epoque}/{NB_EPOQUES} terminée en {duree:.0f} s ===")
        print(f"   Erreur entraînement : {erreur_entrainement:.4f}")
        print(f"   Erreur contrôle     : {erreur_controle:.4f}")
        print(f"   Qualité : abîmée {psnr(erreur_abimees):.1f} dB  ->  réparée {psnr(erreur_controle):.1f} dB")

        # ---------- 3. Sauvegarde si nouveau record ----------
        if erreur_controle < record:
            record = erreur_controle
            torch.save(modele.state_dict(), FICHIER_MODELE)
            print(f"   Nouveau record ! Cerveau sauvegardé dans {FICHIER_MODELE}")
        print()

    print("Entraînement terminé ! Meilleure qualité :", f"{psnr(record):.1f} dB")