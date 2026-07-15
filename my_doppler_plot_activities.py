import argparse
import numpy as np
import pickle
import math as mt
import os
import glob
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter

# --- Dictionnaires de mapping basés sur l'article EHUNAM ---
ACTIVITIES = {
    'W': 'Walking', 'S': 'Standing still', 'J': 'Jumping',
    'T': 'Sitting still', 'G': 'Sitting down/Getting up', 'F': 'Falling',
    'E': 'Empty', '#': 'None'
}

APPLICATIONS = {
    'HAR': 'Human Activity Recognition', 'PC': 'People Counting',
    'MR': 'Machine Recognition', 'MAR': 'Machine Activity Recognition',
    'PCMAR': 'People + Machine', 'E': 'Empty'
}

MACHINES = {
    '1': 'Hair removal', '2': 'Hair dryer', '3': 'Jigsaw', '4': 'Drill',
    '5': 'Horizontal lathe (1)', '6': 'Horizontal lathe (2)', 
    '7': 'Horizontal lathe (3)', '8': 'Horizontal lathe (4)',
    '9': 'Manual milling cutter', '#': 'None'
}
# -----------------------------------------------------------

def parse_ehunam_filename(filename):
    """
    Extrait les métadonnées du nom de fichier selon la norme du dataset EHUNAM.
    Format attendu : Campaign_Set_Rx_Application_People_Activity_Machine_Status_Seq
    """
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')
    
    # Si le nom ne respecte pas les 9 champs, on renvoie le nom brut
    if len(parts) < 9:
        return f"File: {name_without_ext}"
        
    campaign = parts[0]
    set_name = parts[1]
    rx = parts[2]
    app = parts[3]
    people = parts[4]
    activity_code = parts[5]
    machine_code = parts[6]
    status = parts[7]
    
    # Décodage des champs
    app_desc = APPLICATIONS.get(app, app)
    act_desc = ACTIVITIES.get(activity_code, activity_code)
    
    # Le nombre de personnes correspond au nombre de lettres dans le champ 5 (ex: 'abf' = 3)
    num_people = 0 if people == '#' else len(people)
    
    # Déduction de la bande passante selon la lettre du Set (A=20MHz, B=80MHz)
    if 'A' in set_name:
        bw = "20 MHz"
    elif 'B' in set_name:
        bw = "80 MHz"
    else:
        bw = "80 MHz (default)"
        
    # Formatage du titre
    title_line1 = f"{app_desc} | {campaign} Set {set_name} | BW: {bw} | Rx: {rx}"
    
    title_line2 = f"Activity: {act_desc} | People: {num_people}"
    if machine_code != '#':
        mach_desc = MACHINES.get(machine_code, machine_code)
        stat_desc = "Running" if status == 'R' else "On"
        title_line2 += f" | Machine: {mach_desc} ({stat_desc})"
        
    return f"{title_line1}\n{title_line2}"

def plot_spectrograms(stft_log, feature_length, sliding, start_plt, end_plt, out_dir, custom_title):
    """Plot Doppler spectrograms directly without splitting"""
    
    # Physical parameters
    Tc = 6e-3
    fc = 5e9
    v_light = 3e8
    delta_v = round(v_light / (Tc * fc * feature_length), 3)
    temps_par_fenetre = sliding * Tc
    
    # Slice temporal range
    stft_sliced = stft_log[start_plt:min(stft_log.shape[0], end_plt), :]
    
    # Create time and velocity axes
    axe_temps = np.arange(start_plt, start_plt + stft_sliced.shape[0]) * temps_par_fenetre
    axe_vitesses = (np.arange(feature_length) - feature_length / 2) * delta_v
    
    # Plot the spectrogram
    plt.figure(figsize=(12, 7)) # Hauteur légèrement augmentée pour le titre sur 2 lignes
    mesh = plt.pcolormesh(axe_temps, axe_vitesses, stft_sliced.T, cmap='viridis',
                          shading='auto', vmin=-12.0, vmax=0.0)
    
    plt.ylim(-4.5, 4.5)
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    # Injection du titre dynamique ici
    plt.title(custom_title, fontsize=11, pad=15, fontweight='bold')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity (m/s)')
    plt.colorbar(mesh, label='Power [dB]')
    
    return plt, axe_temps, axe_vitesses, stft_sliced

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Doppler spectrograms from .txt files')
    parser.add_argument('dir_path', help='Path to the folder containing the input .txt files')
    parser.add_argument('out_dir', help='Path to the folder where the .png plots will be saved')
    parser.add_argument('feature_length', help='FFT height/bins (e.g., 100)', type=int)
    parser.add_argument('sliding', help='Sliding step size (e.g., 32)', type=int)
    parser.add_argument('start_plt', help='Start index for plotting (e.g., 0)', type=int)
    parser.add_argument('end_plt', help='End index for plotting (e.g., 256)', type=int)

    args = parser.parse_args()

    # Verification and creation of the custom output folder
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"Output folder created: {args.out_dir}")

    # Search for all .txt files in the input folder
    chemin_recherche = os.path.join(args.dir_path, '*.txt')
    fichiers_txt = glob.glob(chemin_recherche)

    if not fichiers_txt:
        print(f"No .txt files found in the input folder: {args.dir_path}")
        exit()

    print("\n" + "#"*60)
    print("#  DOPPLER PLOTTING START")
    print("#"*60)
    print(f"\n[Found] {len(fichiers_txt)} file(s)")
    print(f"[Output] Directory: {args.out_dir}")
    print("-"*60)

    # Processing loop
    for file_path in fichiers_txt:
        file_start = time.time()
        file_name = os.path.basename(file_path)
        
        with open(file_path, "rb") as fp:
            stft_data = pickle.load(fp)

        # Log conversion
        seuil = mt.pow(10, -2.5)
        stft_data[stft_data < seuil] = seuil
        stft_log = 10 * np.log10(stft_data)

        stft_log = gaussian_filter(stft_log, sigma=(1.0, 0.5)) ###

        # 1. Extraction des métadonnées du nom du fichier
        dynamic_title = parse_ehunam_filename(file_name)

        # 2. Passage du titre à la fonction de tracé
        plt_obj, axe_temps, axe_vitesses, stft_sliced = plot_spectrograms(
            stft_log, args.feature_length, args.sliding, 
            args.start_plt, args.end_plt, args.out_dir, dynamic_title
        )
        
        # Save plot
        output_name = os.path.splitext(file_name)[0] + '.png'
        output_path = os.path.join(args.out_dir, output_name)
        plt_obj.savefig(output_path, bbox_inches='tight', dpi=300)
        plt_obj.close()
        
        file_elapsed = time.time() - file_start
        print(f"  ✓ {file_name} -> {output_name} ({file_elapsed:.2f}s)")

    print("\n" + "#"*60)
    print(f"#  DOPPLER PLOTTING COMPLETED")
    print("#"*60 + "\n")