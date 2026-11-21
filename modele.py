# modele.py : le cerveau de notre débruiteur (un auto-encodeur)

import torch
import torch.nn as nn


def bloc(entree, sortie):
    """Une petite brique de base : elle repère des motifs dans l'image."""
    return nn.Sequential(
        nn.Conv2d(entree, sortie, kernel_size=3, padding=1),  # regarde chaque pixel et ses voisins
        nn.BatchNorm2d(sortie),                               # garde les nombres à une taille raisonnable
        nn.ReLU(inplace=True),                                # ne garde que les motifs utiles
    )


class AutoEncodeur(nn.Module):
    def __init__(self):
        super().__init__()

        # 🕵️ L'encodeur : il résume l'image en la faisant rétrécir
        self.encodeur = nn.Sequential(
            bloc(3, 32),   nn.MaxPool2d(2),   # 128x128 -> 64x64
            bloc(32, 64),  nn.MaxPool2d(2),   # 64x64   -> 32x32
            bloc(64, 128), nn.MaxPool2d(2),   # 32x32   -> 16x16
            bloc(128, 64),                    # le résumé final : 64 x 16 x 16
        )

        # 🎨 Le décodeur : il redessine l'image en la faisant regrandir
        self.decodeur = nn.Sequential(
            nn.Upsample(scale_factor=2), bloc(64, 64),   # 16x16 -> 32x32
            nn.Upsample(scale_factor=2), bloc(64, 32),   # 32x32 -> 64x64
            nn.Upsample(scale_factor=2), bloc(32, 32),   # 64x64 -> 128x128
            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # on revient à 3 couleurs
            nn.Sigmoid(),                                # on garde les nombres entre 0 et 1
        )

    def forward(self, image_abimee):
        resume = self.encodeur(image_abimee)
        image_reparee = self.decodeur(resume)
        return image_reparee


class UNet(nn.Module):
    """Comme l'auto-encodeur, mais avec des raccourcis entre la descente et la remontée."""

    def __init__(self):
        super().__init__()
        self.retrecir = nn.MaxPool2d(2)
        self.agrandir = nn.Upsample(scale_factor=2)

        # ⬇️ La descente (comme l'encodeur)
        self.descente1 = bloc(3, 32)     # 128x128
        self.descente2 = bloc(32, 64)    # 64x64
        self.descente3 = bloc(64, 128)   # 32x32
        self.fond = bloc(128, 128)       # 16x16 : le résumé

        # ⬆️ La remontée : chaque bloc reçoit "ce qui remonte" + "le raccourci"
        self.remontee3 = bloc(128 + 128, 64)  # 32x32
        self.remontee2 = bloc(64 + 64, 32)    # 64x64
        self.remontee1 = bloc(32 + 32, 32)    # 128x128

        self.sortie = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # on revient à 3 couleurs
            nn.Sigmoid(),                                # entre 0 et 1
        )

    def forward(self, x):
        # ⬇️ On descend, en gardant une copie de chaque étage (les raccourcis)
        d1 = self.descente1(x)                     # 32 x 128 x 128
        d2 = self.descente2(self.retrecir(d1))     # 64 x 64 x 64
        d3 = self.descente3(self.retrecir(d2))     # 128 x 32 x 32
        fond = self.fond(self.retrecir(d3))        # 128 x 16 x 16

        # ⬆️ On remonte, en recollant le raccourci du même étage (torch.cat = coller côte à côte)
        r3 = self.remontee3(torch.cat([self.agrandir(fond), d3], dim=1))  # 64 x 32 x 32
        r2 = self.remontee2(torch.cat([self.agrandir(r3), d2], dim=1))    # 32 x 64 x 64
        r1 = self.remontee1(torch.cat([self.agrandir(r2), d1], dim=1))    # 32 x 128 x 128
        return self.sortie(r1)


# ---- Petit test : ce bloc ne s'exécute que si on lance "python modele.py" ----
if __name__ == "__main__":
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    print("Appareil utilisé :", appareil)

    fausses_images = torch.rand(4, 3, 128, 128).to(appareil)

    for nom, Modele in [("AutoEncodeur", AutoEncodeur), ("UNet", UNet)]:
        modele = Modele().to(appareil)
        nb_reglages = sum(p.numel() for p in modele.parameters())
        sortie = modele(fausses_images)
        print(f"\n--- {nom} ---")
        print(f"Nombre de réglages : {nb_reglages:,}".replace(",", " "))
        print("Taille à l'entrée  :", tuple(fausses_images.shape))
        print("Taille à la sortie :", tuple(sortie.shape))