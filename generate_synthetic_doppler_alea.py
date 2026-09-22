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
    Adds random variations so every sample is unique for proper CNN training.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sig = np.zeros_like(t, dtype=complex)
    
    if num_people == 0:
        # ADDED: Add a tiny bit of background white noise even for the empty class
        # This prevents the matrix from being absolute zero, which can sometimes cause NaN losses
        noise = np.random.normal(0, 0.01, size=t.shape) + 1j * np.random.normal(0, 0.01, size=t.shape)
        return t, sig + noise
        
    # MODIFIED: Renamed base parameters
    base_f_modulations = [1.8, 2.2, 1.5, 2.6]
    base_f_deviations = [60.0, -75.0, 45.0, -90.0] 
    
    for i in range(min(num_people, len(base_f_modulations))):
        # ADDED: Random variations for each sample to create dataset diversity (Train vs Valid)
        # Random walking speed variation (+/- 0.3 Hz)
        f_mod = base_f_modulations[i] + np.random.uniform(-0.3, 0.3)
        # Random direction/velocity variation (+/- 10 Hz)
        f_dev = base_f_deviations[i] + np.random.uniform(-10.0, 10.0)
        # Random phase shift (simulate people starting their steps at different times)
        phase_shift = np.random.uniform(0, 2 * np.pi)
        
        # MODIFIED: Included phase_shift in the cosine calculation
        phase = -(f_dev / f_mod) * np.cos(2 * np.pi * f_mod * t + phase_shift)
        sig += np.exp(1j * phase)
        
    # ADDED: Add slight background noise to make it slightly more realistic
    noise = np.random.normal(0, 0.05, size=t.shape) + 1j * np.random.normal(0, 0.05, size=t.shape)
    sig += noise
        
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

def generate_dataset(output_dir, num_samples_per_class=100):
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
    
    for i, file in enumerate(files):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        
        sample_1024 = data[0]
        image_32x32 = sample_1024.reshape(32, 32)
        
        plt.figure(figsize=(4, 4))
        plt.imshow(image_32x32, origin='lower', cmap='viridis', aspect='auto')
        
        filename = os.path.basename(file)
        plt.title(f"Class File: {filename[:15]}...\nShape: 32x32")
        plt.axis('off') 
        plt.tight_layout()
        
        plot_filename = f"plot_{os.path.splitext(filename)[0]}.png"
        save_path = os.path.join(output_dir, plot_filename)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot successfully saved to: {save_path}")

def main():
    dataset_folder = "doppler_output_synthetic"
    # MODIFIED: Ensuring 100 samples per class are generated for a decent dataset size
    print(f"Generating perfect synthetic dataset with variance in '{dataset_folder}'...")
    generate_dataset(output_dir=dataset_folder, num_samples_per_class=100)
    
    print("\nGenerating and saving individual plots...")
    plot_and_save_synthetic_dataset(dataset_folder=dataset_folder)
    
    print("\nGeneration and plotting complete.")
    print("To train your CNN, move 'doppler_output_synthetic' into your ROOT_DIR and run:")
    print("python train.py --train_env doppler_output_synthetic")

if __name__ == "__main__":
    main()