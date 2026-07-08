import argparse
import numpy as np
import pickle
import math as mt
import os
import glob
from plots_utility import plt_doppler_antennas
import matplotlib.pyplot as plt

def decouper_spectrogramme(stft_log, sliding, chemin_txt='limites_spectogramme_doppler.txt', out_dir='plots'):
    print(f"\nReading limits (in seconds) from the file '{chemin_txt}'...")
    
    # 1. Physical parameters (placed at the beginning for conversion)
    sliding = 32
    Tc = 6e-3
    fc = 5e9
    v_light = 3e8
    
    # Real time elapsed for each computed window (0.192 seconds)
    temps_par_fenetre = sliding * Tc 

    # 2. Automatic reading and conversion
    limites_experiences = {}
    
    if not os.path.exists(chemin_txt):
        print(f"Error: The file '{chemin_txt}' was not found.")
        return

    with open(chemin_txt, 'r') as f:
        for ligne in f:
            ligne = ligne.strip()
            if ':' in ligne and '-->' in ligne:
                try:
                    partie_gauche, partie_droite = ligne.split(':')
                    nb_personnes = int(partie_gauche.strip())
                    
                    debut_str, fin_str = partie_droite.split('-->')
                    
                    # Read the values as floating-point numbers (seconds)
                    debut_sec = float(debut_str.strip())
                    fin_sec = float(fin_str.strip())
                    
                    # --- CONVERSION: Seconds -> Matrix indices ---
                    debut_idx = int(debut_sec / temps_par_fenetre)
                    fin_idx = int(fin_sec / temps_par_fenetre)
                    
                    limites_experiences[nb_personnes] = (debut_idx, fin_idx)
                    print(f"   Activity {nb_personnes} : read {debut_sec}s to {fin_sec}s -> Converted to windows {debut_idx} to {fin_idx}")
                    
                except ValueError:
                    print(f"Read error on line: {ligne}")

    if not limites_experiences:
        print("No valid limits found. Operation cancelled.")
        return

    # 3. Plotting parameters
    num_bins = stft_log.shape[1] 
    delta_v = v_light / (Tc * fc * num_bins)
    axe_vitesses = (np.arange(num_bins) - num_bins / 2) * delta_v

    dossier_sortie = out_dir
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)

    vmin_global = np.min(stft_log)
    vmax_global = np.max(stft_log)

    # 4. Slicing loop (now using integer indices)
    print("\nStarting image generation...")
    for nb_personnes, (debut_idx, fin_idx) in limites_experiences.items():
        
        if fin_idx > stft_log.shape[0]:
            print(f"   Warning: the limit {fin_idx} exceeds the data size.")
            fin_idx = stft_log.shape[0]

        segment = stft_log[debut_idx:fin_idx, :]
        axe_temps_absolu = np.arange(debut_idx, fin_idx) * temps_par_fenetre

        plt.figure(figsize=(10, 5))
        
        mesh = plt.pcolormesh(axe_temps_absolu, axe_vitesses, segment.T, cmap='viridis',
                              shading='auto', vmin=-12.0, vmax=0.0)
        
        plt.ylim(-4.5, 4.5) 
        plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        
        plt.title(f'Activity: {nb_personnes} person(s)')
        plt.xlabel('Absolute time (seconds)')
        plt.ylabel('Velocity (m/s)')
        
        plt.colorbar(mesh, label='Power [dB]')
        
        nom_fichier = os.path.join(dossier_sortie, f"spectrogram_{nb_personnes}_persons.png")
        plt.savefig(nom_fichier, bbox_inches='tight', dpi=300)
        plt.close() 
        
    print(f"\nFinished! The images are located in the {dossier_sortie}/ folder")

if __name__ == '__main__':
    # 1. Addition of the new 'out_dir' argument
    parser = argparse.ArgumentParser(description='Plots Doppler profiles for all .txt files in a given folder')
    parser.add_argument('dir_path', help='Path to the folder containing the input .txt files')
    parser.add_argument('out_dir', help='Path to the folder where the .png plots will be saved')
    parser.add_argument('feature_length', help='FFT height/bins (e.g., 100)', type=int)
    parser.add_argument('sliding', help='Sliding step size (e.g., 5)', type=int)
    parser.add_argument('start_plt', help='Start index for plotting (e.g., 0)', type=int)
    parser.add_argument('end_plt', help='End index for plotting (e.g., 1000)', type=int)

    args = parser.parse_args()

    # 2. Physical constants
    Tc = 6e-3
    fc = 5e9
    v_light = 3e8
    delta_v = round(v_light / (Tc * fc * args.feature_length), 3)

    # 3. Verification and creation of the custom output folder
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"Output folder created: {args.out_dir}")

    # 4. Search for all .txt files in the input folder
    chemin_recherche = os.path.join(args.dir_path, '*.txt')
    fichiers_txt = glob.glob(chemin_recherche)

    if not fichiers_txt:
        print(f"No .txt files found in the input folder: {args.dir_path}")
        exit()

    print(f"{len(fichiers_txt)} file(s) found. Starting processing...")

    # 5. The processing loop
    for file_path in fichiers_txt:
        print(f"Processing: {os.path.basename(file_path)}")
        
        with open(file_path, "rb") as fp:
            stft_sum_1 = pickle.load(fp)

        # Mathematical processing
        seuil = mt.pow(10, -2.5)
        stft_sum_1[stft_sum_1 < seuil] = seuil
        stft_sum_1_log = 10 * np.log10(stft_sum_1)

        decouper_spectrogramme(stft_sum_1_log, args.sliding, out_dir=args.out_dir)

        # Temporal slicing (Zoom)
        stft_sum_1_log = stft_sum_1_log[args.start_plt:min(stft_sum_1_log.shape[0], args.end_plt), :]

        stft_antennas = [stft_sum_1_log]

        # Creation of the save filename in the new target folder
        nom_base = os.path.basename(file_path).replace('.txt', '')
        # name_p = os.path.join(args.out_dir, f'csi_doppler_{nom_base}.png')
        name_p = os.path.join(args.out_dir, f'spectogram_complete.png')

        plt_doppler_antennas(stft_antennas, args.sliding, delta_v, name_p)

    print(f"Batch processing finished! All plots are located in: {args.out_dir}")