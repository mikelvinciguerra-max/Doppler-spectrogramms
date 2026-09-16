import glob
import numpy as np
import scipy.io as sio
import shutil
import os
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

REMOVE_20_DEFAULT = [0, 1, 2, 3, 32, 61, 62, 63]
REMOVE_80A = [*range(6), *range(127, 131), *range(251, 256)]
REMOVE_80B = [*range(6), 32, *range(59, 70), 96, *range(123, 134), 160, *range(187, 198), 224, *range(251, 256)]

def clear_output_folder(output_folder):
    """Create an output directory and remove all of its existing contents."""
    os.makedirs(output_folder, exist_ok=True)
    for entry in os.scandir(output_folder):
        if entry.is_dir(follow_symlinks=False):
            shutil.rmtree(entry.path)
        else:
            os.unlink(entry.path)

def _remove_subcarriers(CSI, BW, Subcarriers, file_name):
    """Remove unused CSI subcarriers according to bandwidth and file variant.

    Raises:
        ValueError: If ``BW`` is not a supported bandwidth.
    """
    variant = file_name[6] if len(file_name) > 6 else 'A'
    if BW == 20:
        indices_to_remove = [] if Subcarriers == 56 else REMOVE_20_DEFAULT
    elif BW == 80:
        indices_to_remove = REMOVE_80B if variant == 'B' else REMOVE_80A
    else:
        raise ValueError(f"Unsupported BW ({BW})")

    if indices_to_remove:
        CSI = np.delete(CSI, indices_to_remove, axis=1)
    return CSI


def linear_phase_transformation(csi_phase):
    """Remove the linear phase component caused by temporal offsets.

    The correction estimates a slope and an offset for each packet across
    the subcarriers, mitigating carrier- and sampling-frequency offsets.
    """
    unwrapped_phase = np.unwrap(csi_phase, axis=1)
    K = unwrapped_phase.shape[1]

    # Centred indices: avoids bias -epsilon_s*(K-1)/2 per packet
    m = np.arange(K) - (K - 1) / 2

    # Regression slope on all subcarriers (robust to impulsive noise), instead of just the 2 ends
    epsilon_s = (unwrapped_phase @ m) / (m @ m)
    tau_s = np.mean(unwrapped_phase, axis=1)
    
    epsilon_s = epsilon_s[:, np.newaxis]
    tau_s = tau_s[:, np.newaxis]
    m_matrix = m[np.newaxis, :]
    
    calibrated_phase = unwrapped_phase - (epsilon_s * m_matrix) - tau_s
    return calibrated_phase


def _mean_subtraction(csi_array):
    """Remove the temporal mean to suppress the static CSI component."""
    time_axis = 0

    static_mean = np.mean(csi_array, axis=time_axis, keepdims=True)
    filtered = csi_array - static_mean
    
    return filtered


def process_one_file(file_path, output_folder):
    """Preprocess one MATLAB CSI file and save its processed representation.

    The processing removes unused subcarriers, calibrates phase, normalizes
    amplitude, removes the static mean, and writes a ``CSI`` variable to the
    output MATLAB file. Loading and unsupported-input failures are returned as
    status strings for the multiprocessing caller.
    """
    file_name = os.path.basename(file_path)
    t0 = time.time()

    try:
        mat_data = sio.loadmat(file_path)
    except Exception as e:
        return f"[ERROR] {file_name}: loading failed ({e})"

    if not {'BW', 'CSI', 'Subcarriers'}.issubset(mat_data.keys()):
        return f"[SKIP] {file_name}: missing variables"

    BW = mat_data['BW'].item()
    Subcarriers = mat_data['Subcarriers'].item()

    # Phase 1: Subcarrier Removal
    try:
        CSI = _remove_subcarriers(mat_data['CSI'], BW, Subcarriers, file_name)
    except ValueError as e:
        return f"[SKIP] {file_name}: {e}"

    # Phase 2: Linear Phase Transform + Amplitude-Phase 
    csi_amplitude = np.abs(CSI)
    csi_amplitude /= np.mean(csi_amplitude, axis=1, keepdims=True)
    csi_phase = np.angle(CSI)
    
    csi_phase_calibrated = linear_phase_transformation(csi_phase)
    
    csi_processed = csi_amplitude * np.exp(1j * csi_phase_calibrated)
    csi_a = np.abs(csi_processed).astype(np.float32)
    csi_p = np.angle(csi_processed).astype(np.float32)
    csi_output = np.stack((csi_a, csi_p), axis=-1)

    # Phase 3: Mean Subtraction (static component removal)
    csi_output = _mean_subtraction(csi_output)

    output_path = os.path.join(output_folder, file_name)
    sio.savemat(output_path, {'CSI': csi_output})

    return f"OK {file_name} ({time.time() - t0:.2f}s)"


def _class_from_file_name(file_path):
    """Extract the target class encoded in an EHUNAM filename.

    Returns ``0`` for the empty class, the number of people for a ``PC``
    recording, or ``None`` when the filename does not match the expected
    format.
    """
    parts = os.path.splitext(os.path.basename(file_path))[0].split('_')
    if len(parts) >= 4 and parts[3] == 'E':
        return 0
    if len(parts) >= 5 and parts[3] == 'PC':
        return len(parts[4])
    return None


def _limit_files_per_class(file_paths, max_files_per_class):
    """Keep at most a requested number of files for each target class.

    When ``max_files_per_class`` is ``None``, the input list is returned
    unchanged. Only classes 0 through 4 are retained when a limit is used.

    Raises:
        ValueError: If ``max_files_per_class`` is less than one.
    """
    if max_files_per_class is None:
        return file_paths
    if max_files_per_class < 1:
        raise ValueError('max_files_per_class must be at least 1')

    files_by_class = {}
    for file_path in sorted(file_paths):
        class_id = _class_from_file_name(file_path)
        
        if class_id is not None and class_id in [0, 1, 2, 3, 4]:
            files_by_class.setdefault(class_id, []).append(file_path)

    limited_files = [
        file_path
        for class_id in sorted(files_by_class)
        for file_path in files_by_class[class_id][:max_files_per_class]
    ]
    return limited_files


def run_pipeline(input_folder, output_folder, workers=None, max_files_per_class=None):
    """Run the parallel preprocessing pipeline for all matching MAT files.

    The output directory is cleared before processing. Files can optionally be
    limited per class; by default, every file matching the target classes is
    processed.
    """
    clear_output_folder(output_folder)
    mat_files = glob.glob(os.path.join(input_folder, '*.mat'))
    if not mat_files:
        print(f'No .mat files found in "{input_folder}".')
        return

    mat_files = _limit_files_per_class(mat_files, max_files_per_class)
    if not mat_files:
        print('No files matched a known target class.')
        return

    workers = workers or os.cpu_count()
    print(f"Processing {len(mat_files)} files using {workers} parallel worker(s)...")

    total_start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_one_file, fp, output_folder): fp
            for fp in mat_files
        }
        progress = tqdm(as_completed(futures), total=len(mat_files), desc='Preprocessing', unit='file')
        for future in progress:
            future.result()

    print(f"\nPipeline completed in {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing pipeline for CSI files (EHUNAM).")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing the raw .mat files")
    parser.add_argument("output_folder", type=str, help="Path to the output folder for the processed files")
    parser.add_argument("--workers", type=int, default=2,
                         help="Number of parallel processes (default: 2)")
    parser.add_argument("--max_files_per_class", type=int, default=None,
                         help="Maximum number of files to process per class (default: all files)")

    args = parser.parse_args()
    
    print("\n" + "#"*60)
    print("#  PREPROCESSING PIPELINE START")
    print(f"#  Input folder  : {args.input_folder}")
    print(f"#  Output folder : {args.output_folder}")
    print("#"*60)

    run_pipeline(
        args.input_folder, 
        args.output_folder,
        workers=args.workers,
        max_files_per_class=args.max_files_per_class
    )