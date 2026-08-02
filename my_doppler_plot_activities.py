import argparse
import numpy as np
import pickle
import math as mt
import os
import glob
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter

# --- Mapping dictionaries based on the EHUNAM paper ---
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
    Extract metadata from the filename according to the EHUNAM dataset naming convention.
    Expected format: Campaign_Set_Rx_Application_People_Activity_Machine_Status_Seq
    """
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')

    # If the name does not follow the 9 fields, return the raw name
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
    
    # Decode the fields
    app_desc = APPLICATIONS.get(app, app)
    act_desc = ACTIVITIES.get(activity_code, activity_code)

    # The number of people corresponds to the number of letters in field 5 (e.g. 'abf' = 3)
    num_people = 0 if people == '#' else len(people)

    # Infer the bandwidth from the Set letter (A=20MHz, B=80MHz)
    if 'A' in set_name:
        bw = "20 MHz"
    elif 'B' in set_name:
        bw = "80 MHz"
    else:
        bw = "80 MHz (default)"
        
    # Format the title
    title_line1 = f"{app_desc} | {campaign} Set {set_name} | BW: {bw} | Rx: {rx}"
    
    title_line2 = f"Activity: {act_desc} | People: {num_people}"
    if machine_code != '#':
        mach_desc = MACHINES.get(machine_code, machine_code)
        stat_desc = "Running" if status == 'R' else "On"
        title_line2 += f" | Machine: {mach_desc} ({stat_desc})"
        
    return f"{title_line1}\n{title_line2}"

# New 'Tc' parameter added to the function
def plot_spectrograms(stft_log, feature_length, sliding, start_plt, end_plt, out_dir, custom_title, Tc):
    """Plot Doppler spectrograms directly without splitting"""
    
    # Physical parameters
    fc = 5e9
    v_light = 3e8
    delta_v = round(v_light / (Tc * fc * feature_length), 3)
    time_per_window = sliding * Tc
    
    # Slice temporal range
    stft_sliced = stft_log[start_plt:min(stft_log.shape[0], end_plt), :]
    
    # Create time and velocity axes
    time_axis = np.arange(start_plt, start_plt + stft_sliced.shape[0]) * time_per_window
    velocity_axis = (np.arange(feature_length) - feature_length / 2) * delta_v
    
    # Plot the spectrogram
    plt.figure(figsize=(12, 7)) 
    
    mesh = plt.pcolormesh(time_axis, velocity_axis, stft_sliced.T, cmap='viridis',
                          shading='auto', vmin=np.min(stft_sliced), vmax=0.0)
    
    plt.ylim(-4.5, 4.5)
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    # Inject the dynamic title here
    plt.title(custom_title, fontsize=11, pad=15, fontweight='bold')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity (m/s)')
    plt.colorbar(mesh, label='Power [dB]')
    
    return plt, time_axis, velocity_axis, stft_sliced

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Doppler spectrograms from .txt files')
    parser.add_argument('dir_path', help='Path to the folder containing the input .txt files')
    parser.add_argument('out_dir', help='Path to the folder where the .png plots will be saved')
    parser.add_argument('feature_length', help='FFT height/bins (e.g., 100)', type=int)
    parser.add_argument('sliding', help='Sliding step size (e.g., 32)', type=int)
    parser.add_argument('start_plt', help='Start index for plotting (e.g., 0)', type=int)
    parser.add_argument('end_plt', help='End index for plotting (e.g., 256)', type=int)
    # New argument added here
    parser.add_argument('--tc', help='Time parameter Tc in seconds (default 6e-3)', default=6e-3, required=False, type=float)

    args = parser.parse_args()

    # Verify and create the custom output folder
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"Output folder created: {args.out_dir}")

    # Search for all .txt files in the input folder
    search_path = os.path.join(args.dir_path, '*.txt')
    txt_files = glob.glob(search_path)

    if not txt_files:
        print(f"No .txt files found in the input folder: {args.dir_path}")
        exit()

    print("\n" + "#"*60)
    print("#  DOPPLER PLOTTING START")
    print("#"*60)
    print(f"\n[Found] {len(txt_files)} file(s)")
    print(f"[Output] Directory: {args.out_dir}")
    print(f"[Parameters] FFT size: {args.feature_length}, Sliding: {args.sliding}, Tc: {args.tc}s")
    print("-"*60)

    # Processing loop
    for file_path in txt_files:
        file_start = time.time()
        file_name = os.path.basename(file_path)
        
        with open(file_path, "rb") as fp:
            stft_data = pickle.load(fp)

        # Log conversion
        threshold = mt.pow(10, -2.5)
        stft_data[stft_data < threshold] = threshold
        stft_log = 10 * np.log10(stft_data)

        stft_log = gaussian_filter(stft_log, sigma=(1.0, 0.5)) ###

        # 1. Extract metadata from the filename
        dynamic_title = parse_ehunam_filename(file_name)

        # 2. Pass the title to the plotting function and the Tc argument
        plt_obj, time_axis, velocity_axis, stft_sliced = plot_spectrograms(
            stft_log, args.feature_length, args.sliding, 
            args.start_plt, args.end_plt, args.out_dir, dynamic_title, args.tc
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