import argparse
import numpy as np
import scipy.io as sio
import math as mt
import shutil
from scipy.fftpack import fft
from scipy.fftpack import fftshift
from scipy.signal.windows import hann
import pickle
import os
import time
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

EPS = 1e-10

def v1_log_dc_removal(spectrogram, dc_bins=1, clip_db=40.0):
    """Remove central DC bins and normalize spectrogram values in dB.

    Args:
        spectrogram: Spectrogram arranged as frequency by time.
        dc_bins: Number of bins removed on each side of the central bin.
        clip_db: Dynamic range retained below the maximum value.

    Returns:
        A ``float32`` spectrogram normalized to the range ``[0, 1]``.
    """
    spec = spectrogram.copy().astype(np.float64)
    center = spec.shape[0] // 2
    spec[max(0, center - dc_bins): center + dc_bins + 1, :] = 0.0
    spec_db = 20 * np.log10(spec + EPS)
    floor = spec_db.max() - clip_db
    spec_db = np.clip(spec_db, floor, spec_db.max())
    spec_norm = (spec_db - floor) / (clip_db + EPS)
    return spec_norm.astype(np.float32)

def v3_percentile_normalization(spectrogram, low_pct=1, high_pct=99):
    """Clip a spectrogram to percentiles and normalize it to ``[0, 1]``.

    Args:
        spectrogram: Spectrogram to normalize.
        low_pct: Lower percentile used as the clipping boundary.
        high_pct: Upper percentile used as the clipping boundary.

    Returns:
        A clipped and normalized ``float32`` spectrogram.
    """
    lo, hi = np.percentile(spectrogram, [low_pct, high_pct])
    spec = np.clip(spectrogram, lo, hi)
    spec = (spec - lo) / (hi - lo + EPS)
    return spec.astype(np.float32)

def v4_noise_floor_masking(spectrogram, noise_floor_pct=20, per_bin=True):
    """Mask values below a percentile noise floor and standardize the result.

    Args:
        spectrogram: Spectrogram to process.
        noise_floor_pct: Percentile used to estimate the noise floor.
        per_bin: Standardize each frequency bin independently when true;
            otherwise standardize the complete spectrogram.

    Returns:
        A noise-floor-masked and standardized ``float32`` spectrogram.
    """
    noise_floor = np.percentile(spectrogram, noise_floor_pct)
    spec = np.where(spectrogram > noise_floor, spectrogram - noise_floor, 0.0)
    if per_bin:
        mu = spec.mean(axis=1, keepdims=True)
        sigma = spec.std(axis=1, keepdims=True) + EPS
        spec = (spec - mu) / sigma
    else:
        spec = (spec - spec.mean()) / (spec.std() + EPS)
    return spec.astype(np.float32)


def clear_output_folder(output_folder):
    """Create an output directory and remove all existing contents."""
    os.makedirs(output_folder, exist_ok=True)
    for entry in os.scandir(output_folder):
        if entry.is_dir(follow_symlinks=False):
            shutil.rmtree(entry.path)
        else:
            os.unlink(entry.path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('subdirs', help='Sub-directories')
    parser.add_argument('dir_doppler', help='Directory to save the Doppler data')
    parser.add_argument('start', help='Start processing', type=int)
    parser.add_argument('end', help='End processing', type=int)
    parser.add_argument('sample_length', help='Number of packet in a sample', type=int)
    parser.add_argument('sliding', help='Number of packet for sliding operations', type=int)
    parser.add_argument('noise_level', help='Level for the noise to be removed', type=float)
    parser.add_argument('--bandwidth', help='Bandwidth in [MHz]', default=80, required=False, type=int)
    parser.add_argument('--sub_band', help='Sub_band idx', default=1, required=False, type=int)
    parser.add_argument('--tc', help='Time parameter Tc in seconds', default=6e-3, required=False, type=float)
    parser.add_argument('--fft', help='Number of FFT values', default=1024, required=False, type=int)
    parser.add_argument('--prep_version', choices=['default', 'v1', 'v3', 'v4'], default='v4', 
                        help='Choose the spectrogram preprocessing method')
    
    args = parser.parse_args()

    num_symbols = args.sample_length 
    middle = int(mt.floor(num_symbols / 2))

    Tc = args.tc
    fc = 5e9
    v_light = 3e8
    delta_v = round(v_light / (Tc * fc * num_symbols), 3)

    sliding = args.sliding
    noise_lev = args.noise_level
    bandwidth = args.bandwidth
    sub_band = args.sub_band

    list_subdir = args.subdirs

    print("\n" + "#"*60)
    print(f"#  DOPPLER COMPUTATION PIPELINE START (PREP: {args.prep_version.upper()})")
    print("#"*60)

    for subdir in list_subdir.split(','):
        path_doppler = args.dir_doppler + subdir
        clear_output_folder(path_doppler)

        exp_dir = args.dir + subdir + '/'

        names = []
        all_files = os.listdir(exp_dir)
        for i in range(len(all_files)):
            if (all_files[i][-4:] == '.mat'):
                names.append(all_files[i][:-4])

        for name in tqdm(names, desc=f"Processing {subdir or '/'}", unit="file"):
            file_start = time.time()
            path_doppler_name = path_doppler + '/' + name + '.txt'
            
            name_file = exp_dir + name + '.mat'
            mdic = sio.loadmat(name_file)
            csi_matrix_processed = mdic['CSI']

            csi_matrix_processed = csi_matrix_processed[args.start:args.end, :, :]
            
            csi_matrix_complete = csi_matrix_processed[:, :, 0]*np.exp(1j*csi_matrix_processed[:, :, 1])

            csi_d_profile_list = []
            num_iterations = (csi_matrix_complete.shape[0] - num_symbols) // sliding
            
            for i in range(0, csi_matrix_complete.shape[0]-num_symbols, sliding):
                csi_matrix_cut = csi_matrix_complete[i:i+num_symbols, :]
                csi_matrix_cut = np.nan_to_num(csi_matrix_cut)

                hann_window = np.expand_dims(hann(num_symbols), axis=-1)
                csi_matrix_wind = np.multiply(csi_matrix_cut, hann_window)
                
                csi_doppler_prof = fft(csi_matrix_wind, n=args.fft, axis=0)
                csi_doppler_prof = fftshift(csi_doppler_prof, axes=0)

                csi_d_map = np.abs(csi_doppler_prof * np.conj(csi_doppler_prof))
                csi_d_map = np.sum(csi_d_map, axis=1)
                
                csi_d_map = gaussian_filter1d(csi_d_map, sigma=1.0) 
                
                csi_d_profile_list.append(csi_d_map)
                
            csi_d_profile_array = np.asarray(csi_d_profile_list)
            
            if args.prep_version == 'default':
                csi_d_profile_array_max = np.max(csi_d_profile_array, axis=1, keepdims=True)
                csi_d_profile_array = csi_d_profile_array / csi_d_profile_array_max
                csi_d_profile_array[csi_d_profile_array < mt.pow(10, noise_lev)] = mt.pow(10, noise_lev)
            else:
                spec_freq_time = csi_d_profile_array.T

                if args.prep_version == 'v1':
                    spec_freq_time = v1_log_dc_removal(spec_freq_time)
                elif args.prep_version == 'v3':
                    spec_freq_time = v3_percentile_normalization(spec_freq_time)
                elif args.prep_version == 'v4':
                    spec_freq_time = v4_noise_floor_masking(spec_freq_time)

                csi_d_profile_array = spec_freq_time.T

            with open(path_doppler_name, "wb") as fp:  
                pickle.dump(csi_d_profile_array, fp)
            
    print("\n" + "#"*60)
    print("#  DOPPLER COMPUTATION COMPLETED")
    print("#"*60 + "\n")