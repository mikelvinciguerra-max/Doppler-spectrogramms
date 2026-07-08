import numpy as np
import os
import scipy.io as sio
import gc

def process_and_save_csi_mean_subtraction(file_path, var_name, overwrite=False):
    # 1. Loading the file
    try:
        mat_data = sio.loadmat(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_path}' not found.")
    
    if var_name not in mat_data:
        raise KeyError(f"Variable '{var_name}' not found in the .mat file.")
    
    csi_matrix = mat_data[var_name]
    print(f"-> Loaded CSI matrix dimensions: {csi_matrix.shape}")
    
    # 2. Dynamic time axis detection
    time_axis = np.argmax(csi_matrix.shape)
    if time_axis != 0:
        print("-> Auto-transposing to set time as the primary axis (axis 0)...")
        csi_matrix = np.swapaxes(csi_matrix, 0, time_axis)
    
    # 3. Mean Subtraction (Static Clutter Removal)
    print("-> Starting Mean Subtraction filtering...")
    
    # Calculate the mean along the time axis (axis 0)
    # keepdims=True ensures the shape stays compatible for vectorized subtraction
    static_mean = np.mean(csi_matrix, axis=0, keepdims=True)
    
    # Subtract the static environment signature from the entire matrix
    filtered_csi = csi_matrix - static_mean
    
    # 4. RAM Cleanup
    del csi_matrix
    del static_mean
    gc.collect()
    
    # Revert transposition if axes were swapped
    if time_axis != 0:
        filtered_csi = np.swapaxes(filtered_csi, 0, time_axis)
    
    # 5. Saving process
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

dossier = "data_preprocessed"

for file in os.listdir(dossier):
    if file.endswith(".mat"):
        file_path = os.path.join(dossier, file)
        process_and_save_csi_mean_subtraction(file_path, var_name='CSI', overwrite=False)