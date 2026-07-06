import numpy as np
import pickle
import math as mt
import matplotlib.pyplot as plt
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Découpe le spectrogramme complet selon les limites du fichier texte')
    parser.add_argument('file_path', help='Chemin vers csi_matrix_processed.txt (fichier de données binaire)')
    args = parser.parse_args()

    chemin_limites = 'limites_spectogramme_doppler.txt'
    
    # 1. Lecture automatique des limites depuis le fichier texte
    print(f"🚀 Lecture des limites depuis le fichier '{chemin_limites}'...")
    limites_experiences = {}
    
    if not os.path.exists(chemin_limites):
        print(f"⚠️ Erreur : Le fichier {chemin_limites} est introuvable dans ce dossier.")
        exit()
        
    with open(chemin_limites, 'r') as f:
        for ligne in f:
            ligne = ligne.strip()
            # On cherche le bon format de texte
            if ':' in ligne and '-->' in ligne:
                partie_gauche, partie_droite = ligne.split(':')
                nb_personnes = int(partie_gauche.strip())
                
                debut_str, fin_str = partie_droite.split('-->')
                debut = int(debut_str.strip())
                fin = int(fin_str.strip())
                
                limites_experiences[nb_personnes] = (debut, fin)

    print(f"✅ Limites chargées : {limites_experiences}")

    # 2. Chargement des données CSI
    print(f"\n📡 Chargement de la matrice complète : {args.file_path}")
    if not os.path.exists(args.file_path):
        print("⚠️ Fichier de données introuvable.")
        exit()

    with open(args.file_path, "rb") as fp:
        stft_sum = pickle.load(fp)

    # 3. Traitement mathématique et Contraste
    seuil = mt.pow(10, -2.5)
    stft_sum[stft_sum < seuil] = seuil
    stft_log = 10 * np.log10(stft_sum)

    val_max = np.max(stft_log)
    stft_log = np.clip(stft_log, val_max - 30, val_max)

    # 4. Paramètres physiques pour la conversion des axes
    sliding = 32
    Tc = 6e-3
    fc = 5e9
    v_light = 3e8
    num_bins = stft_log.shape[1] 

    # Création de l'axe des vitesses (Y) en m/s
    delta_v = v_light / (Tc * fc * num_bins)
    axe_vitesses = (np.arange(num_bins) - num_bins / 2) * delta_v

    dossier_sortie = './plots'
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)

    # 5. Boucle de découpage
    print("\nLancement de la génération des images...")
    for nb_personnes, (debut, fin) in limites_experiences.items():
        print(f"✂️  Extraction pour {nb_personnes} personne(s) (Fenêtres {debut} à {fin})...")
        
        # Sécurité : si la matrice est plus courte que prévue
        if fin > stft_log.shape[0]:
            print(f"   ⚠️ Attention, la limite {fin} dépasse la taille des données ({stft_log.shape[0]}).")
            fin = stft_log.shape[0]

        segment = stft_log[debut:fin, :]
        
        # Axe temporel absolu (conserve le vrai temps de l'expérience)
        axe_temps_absolu = np.arange(debut, fin) * sliding * Tc

        plt.figure(figsize=(10, 5))
        plt.pcolormesh(axe_temps_absolu, axe_vitesses, segment.T, cmap='viridis', shading='auto')
        plt.ylim(-4.5, 4.5) 
        
        plt.title(f'Activité : {nb_personnes} personne(s)')
        plt.xlabel('Temps absolu de l\'expérience (secondes)')
        plt.ylabel('Vitesse (m/s)')
        plt.colorbar(label='Puissance [dB]')
        
        nom_fichier = os.path.join(dossier_sortie, f"spectrogramme_{nb_personnes}_personne.png")
        plt.savefig(nom_fichier, bbox_inches='tight', dpi=300)
        plt.close() 
        
    print(f"\n🎉 Terminé ! Les images sont dans le dossier {dossier_sortie}/")