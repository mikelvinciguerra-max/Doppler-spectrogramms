import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
import gc

def process_and_save_csi_clutter_removal(file_path, var_name, sample_rate, cutoff_freq=1.5, filter_order=4, overwrite=False):
    try:
        mat_data = sio.loadmat(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_path}' not found.")
    
    if var_name not in mat_data:
        raise KeyError(f"Variable '{var_name}' not found in the .mat file.")
    
    csi_matrix = mat_data[var_name]
    print(f"-> Loaded CSI matrix dimensions: {csi_matrix.shape}")
    
    time_axis = np.argmax(csi_matrix.shape)
    if time_axis != 0:
        print("-> Auto-transposing to set time as the primary axis (axis 0)...")
        csi_matrix = np.swapaxes(csi_matrix, 0, time_axis)
    
    nyquist_freq = 0.5 * sample_rate
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(filter_order, normalized_cutoff, btype='highpass', analog=False)
    
    print("-> Starting high-pass filtering. This may take a moment for large matrices...")
    filtered_csi = np.zeros_like(csi_matrix)
    
    if len(csi_matrix.shape) == 3:
        num_subcarriers = csi_matrix.shape[1]
        num_antennas = csi_matrix.shape[2]
        
        for ant_idx in range(num_antennas):
            print(f"   [+] Processing antenna {ant_idx + 1}/{num_antennas}...")
            for sub_idx in range(num_subcarriers):
                filtered_csi[:, sub_idx, ant_idx] = filtfilt(b, a, csi_matrix[:, sub_idx, ant_idx])
                
    elif len(csi_matrix.shape) == 2:
        num_subcarriers = csi_matrix.shape[1]
        print("   [+] Processing subcarriers...")
        for sub_idx in range(num_subcarriers):
            filtered_csi[:, sub_idx] = filtfilt(b, a, csi_matrix[:, sub_idx])
    else:
        raise ValueError("Unsupported format: matrix has more than 3 dimensions.")

    del csi_matrix
    gc.collect()
    
    if time_axis != 0:
        filtered_csi = np.swapaxes(filtered_csi, 0, time_axis)
    
    if overwrite:
        mat_data[var_name] = filtered_csi
        print(f"\n-> Variable '{var_name}' successfully overwritten.")
    else:
        new_var_name = f"{var_name}_filtered"
        mat_data[new_var_name] = filtered_csi
        print(f"\n-> Filtered data appended under new variable '{new_var_name}'.")
    
    print("-> Writing file to disk...")
    sio.savemat(file_path, mat_data)
    print("-> Done!")

process_and_save_csi_clutter_removal(
    file_path="data/csi_matrix_processed.mat", 
    var_name="csi_matrix_processed", 
    sample_rate=1000, 
    overwrite=True
)
