import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import os
import pickle
import torch
import torch.nn.functional as F
import glob 

def generate_people_signal(fs=1000, duration=5.0, num_people=0):
    """
    Generates a synthetic time-domain signal for a specific number of people (0 to 4).
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sig = np.zeros_like(t, dtype=complex)
    
    if num_people == 0:
        return t, sig
        
    f_modulations = [1.8, 2.2, 1.5, 2.6]
    f_deviations = [60.0, -75.0, 45.0, -90.0] 
    
    for i in range(min(num_people, len(f_modulations))):
        phase = -(f_deviations[i] / f_modulations[i]) * np.cos(2 * np.pi * f_modulations[i] * t)
        sig += np.exp(1j * phase)
        
    return t, sig

def process_to_32x32_spectrogram(sig, fs):
    """
    Computes STFT and forces the output magnitude to exactly 32x32 pixels using interpolation.
    """
    nperseg = 256
    noverlap = nperseg // 2
    f, t_spec, Zxx = stft(sig, fs, window='hann', nperseg=nperseg, noverlap=noverlap, return_onesided=False)
    
    Zxx = np.fft.fftshift(Zxx, axes=0)
    magnitude = np.abs(Zxx).astype(np.float32)
    
    tensor_img = torch.tensor(magnitude).unsqueeze(0).unsqueeze(0) 
    resized_tensor = F.interpolate(tensor_img, size=(32, 32), mode='bilinear', align_corners=False)
    matrix_32x32 = resized_tensor.squeeze().numpy()
    
    max_val = matrix_32x32.max()
    if max_val > 0:
        matrix_32x32 /= max_val
        
    return matrix_32x32

def generate_dataset(output_dir, num_samples_per_class=50):
    """
    Creates synthetic pickled .txt datasets matching the naming convention of dataset.py.
    """
    os.makedirs(output_dir, exist_ok=True)
    fs = 1000  
    duration = 5.0
    
    label_to_filename = {
        0: "SYNTH_01B_1_E_empty.txt",            
        1: "SYNTH_01B_1_PC_a_1person.txt",       
        2: "SYNTH_01B_1_PC_ab_2people.txt",      
        3: "SYNTH_01B_1_PC_abc_3people.txt",     
        4: "SYNTH_01B_1_PC_abcd_4people.txt"     
    }

    for people_count in range(5):
        samples = []
        for _ in range(num_samples_per_class):
            _, sig = generate_people_signal(fs, duration, num_people=people_count)
            
            matrix_32x32 = process_to_32x32_spectrogram(sig, fs)
            flat_profile = matrix_32x32.flatten()
            samples.append(flat_profile)
            
        dataset_array = np.vstack(samples).astype(np.float32)
        
        filepath = os.path.join(output_dir, label_to_filename[people_count])
        with open(filepath, 'wb') as f:
            pickle.dump(dataset_array, f)
            
        print(f"Saved {filepath} with shape {dataset_array.shape}")

def plot_and_save_synthetic_dataset(dataset_folder="doppler_output_synthetic"):
    """
    Reads the generated .txt datasets, plots the first sample of each class,
    and saves the plots as separate PNG files in a specific subfolder.
    """
    files = sorted(glob.glob(os.path.join(dataset_folder, "*.txt")))
    
    if not files:
        print(f"No files found in {dataset_folder}")
        return

    output_dir = os.path.join("plots", "plots_synthetic")
    os.makedirs(output_dir, exist_ok=True)

    # MODIFIED: Removed the single large figure creation before the loop.
    
    for i, file in enumerate(files):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        
        sample_1024 = data[0]
        image_32x32 = sample_1024.reshape(32, 32)
        
        # ADDED: Create a new distinct figure for each plot
        plt.figure(figsize=(4, 4))
        
        # MODIFIED: Removed plt.subplot() since we are saving separate files
        plt.imshow(image_32x32, origin='lower', cmap='viridis', aspect='auto')
        
        filename = os.path.basename(file)
        plt.title(f"Class File: {filename[:15]}...\nShape: 32x32")
        plt.axis('off') 
        plt.tight_layout()
        
        # ADDED: Generate a specific save path for the current individual plot
        plot_filename = f"plot_{os.path.splitext(filename)[0]}.png"
        save_path = os.path.join(output_dir, plot_filename)
        
        # MODIFIED: Save the individual figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # ADDED: Close the figure to avoid overlapping images in memory
        plt.close()
        
        print(f"Plot successfully saved to: {save_path}")

def main():
    dataset_folder = "doppler_output_synthetic"
    print(f"Generating perfect synthetic dataset in '{dataset_folder}'...")
    generate_dataset(output_dir=dataset_folder, num_samples_per_class=100)
    
    print("\nGenerating and saving individual plots...")
    plot_and_save_synthetic_dataset(dataset_folder=dataset_folder)
    
    print("\nGeneration and plotting complete. You can now run train.py using --train_env doppler_output_synthetic")

if __name__ == "__main__":
    main()
