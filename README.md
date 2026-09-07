# Doppler Spectrograms for CSI-Based Activity Recognition

This repository implements a full pipeline for processing Wi‑Fi CSI (Channel State Information) recordings, converting them into Doppler spectrograms, and classifying human activities with a convolutional neural network (CNN).

The project is designed for activity recognition and people-counting scenarios based on Doppler signatures extracted from CSI data. It includes preprocessing, spectrogram generation, dataset loading, model training, evaluation, confusion matrices, and plotting utilities.

## Overview

The workflow is:

1. Load raw `.mat` CSI files
2. Preprocess subcarriers and phase/amplitude information
3. Generate Doppler spectrograms from temporal CSI sequences
4. Save each spectrogram as a sample in a dataset
5. Train a CNN on those samples
6. Evaluate the model on unseen environments and classes
7. Plot and compare results

This project is especially tailored to EHUNAM-style experiments where each recording is associated with an activity or people-count label.

---

## Repository structure

Important folders and files:

- `data_ehunam/` — raw experiment files in `.mat` format
- `data_preprocessed/` — processed CSI files after preprocessing
- `doppler_output/` — generated Doppler spectrograms saved as `.txt` pickled arrays
- `experiments/` — experiment outputs, model checkpoints, and generated plots
- `models/` — trained CNN checkpoints
- `matrix/` — confusion matrix and performance outputs
- `plots/` — visual outputs from plotting scripts
- `preprocessing.py` — CSI sanitization and pre-processing pipeline
- `my_doppler_computation.py` — main Doppler extraction pipeline
- `dataset.py` — dataset loader and cache management
- `model.py` — CNN architecture
- `train.py` — training script for the CNN
- `eval.py` — evaluation across environments
- `confusion_matrix.py` — confusion matrix generation
- `CSI_doppler_computation.py` — alternative Doppler computation utility
- `my_doppler_plot_activities.py` — activity spectrogram plotting
- `run_experiments.sh` — batch pipeline for multiple parameter configurations
- `run_cnn.sh` — training/evaluation helper for CNN runs
- `prepare_data_for_cnn.sh` — full preparation workflow example

---

## Dependencies

This project requires Python 3.8+ and commonly used scientific and ML libraries.

Example installation:

```bash
pip install numpy scipy matplotlib h5py scikit-learn torch seaborn pillow
```

Depending on your environment, you may also want:

```bash
pip install tqdm pandas
```

If you are using a CUDA-enabled machine, PyTorch should be installed with the appropriate CUDA version for your system.

---

## Data format

The raw data is expected to be stored as MATLAB `.mat` files containing at least:

- `CSI`
- `BW`
- `Subcarriers`

The project assumes recordings are organized by environment or acquisition group, for example:

- `a/`, `b/`, `c/`, `d/`
- or custom directories passed as arguments to the scripts

The file names encode labels such as:

- `MC1_01B_1_E_#_#_#_#_01.mat` → empty / no-people scenario
- `MC1_01B_1_PC_ab_#_#_#_01.mat` → people-counting task with 2 people (`ab`)
- `MC1_01B_1_PC_abc_#_#_#_01.mat` → 3 people

The dataset loader infers the class by parsing the filename fields.

---

## Preprocessing pipeline

The preprocessing step is implemented in `preprocessing.py`.

It performs three main stages:

### 1) Subcarrier removal

Bad or unused subcarriers are removed depending on the bandwidth and file variant.

### 2) Phase normalization and amplitude-phase reconstruction

The code applies a linear phase transformation to reduce temporal offsets and reconstruct a processed complex CSI signal from amplitude and phase components.

### 3) Mean subtraction filtering

Static components are removed to make the signal more sensitive to motion and body movement.

Example usage:

```bash
python3 preprocessing.py /path/to/raw_data /path/to/output_processed_data
```

This creates processed `.mat` files in the output folder, which are then used as input to Doppler computation.

---

## Doppler spectrogram generation

The main Doppler extraction routine is in `my_doppler_computation.py`.

This script:

- loads processed CSI files
- slices a time window from the signal
- converts CSI to complex values
- applies a Hann window
- computes FFT along the temporal axis
- obtains Doppler power maps
- normalizes and filters the output
- saves the resulting spectrograms as pickled NumPy arrays

### Core parameters

The script accepts the following key arguments:

```bash
python3 my_doppler_computation.py <dir> <subdirs> <dir_doppler> <start> <end> <sample_length> <sliding> <noise_level> [--tc ...] [--fft ...]
```

Where:

- `dir` — input directory containing processed CSI `.mat` files
- `subdirs` — comma-separated subdirectories to process (empty string allowed)
- `dir_doppler` — destination folder for Doppler output
- `start`, `end` — time-range indices used for slicing
- `sample_length` — number of symbols in one sample window
- `sliding` — sliding step between consecutive windows
- `noise_level` — threshold level used for noise suppression
- `--tc` — coherence time parameter, e.g. `8.5e-4`
- `--fft` — FFT size, e.g. `100`, `256`, `1024`

Example:

```bash
python3 my_doppler_computation.py data_preprocessed/ "" doppler_output/ 0 2000000 256 220 -0.7 --tc 8.5e-4 --fft 1024
```

The output is stored as `.txt` files: each file contains a pickled array shaped approximately as:

- `(num_windows, fft_size)`

Each row is one Doppler profile sample.

---

## Dataset loading and labels

The dataset logic is implemented in `dataset.py`.

The loader:

- scans all `.txt` files in a Doppler folder
- parses the file name to infer the class label
- converts each Doppler profile row to a training sample
- reshapes each profile to `(1, 32, 32)`
- creates a PyTorch dataset object

Label conventions used by the project:

- `E` → no person → label `0`
- `PC_ab` → 2 people → label `2`
- `PC_abc` → 3 people → label `3`
- ...

The loader also includes a cache mode for RAM-friendly loading when the dataset fits in memory.

---

## CNN model

The neural network architecture is defined in `model.py`.

It uses a compact CNN with:

- 3 convolution blocks
- Mish activation
- Batch normalization
- Max pooling
- Dropout
- fully-connected layers for classification

The model expects input shape:

```python
(batch_size, 1, 32, 32)
```

and outputs logits for a selected number of target classes.

---

## Training

The training pipeline is implemented in `train.py`.

Main features:

- loads a Doppler dataset from a directory
- filters samples by class
- optionally performs stratified k-fold validation
- trains the CNN with Adam optimizer
- saves the trained checkpoint into `models/`
- supports train/test split and robust evaluation

Typical training command:

```bash
python3 train.py --train_env doppler_output_a --epochs 20 --root_dir /path/to/root --classes 0 1 2 3 4
```

The script also supports:

```bash
--k-folds 3
```

for cross-validation diagnostics.

### Training output

A model is saved as a `.pth` checkpoint, for example:

```text
models/model_doppler_d_classes_0-1-2-3-4_epochs_20_kfolds_1.pth
```

The checkpoint contains:

- `model_state_dict`
- `train_env`
- `num_classes`
- `target_classes`
- `root_dir`
- `epochs`
- `k_folds`
- test indices and validation statistics

---

## Evaluation and confusion matrices

### Evaluation

The script `eval.py` loads a saved model and evaluates it on different environments.

Example:

```bash
python3 eval.py --env a --epochs 20 --classes 0 1 2 3 4
```

This computes cross-environment accuracy, e.g. training on environment `a` and testing on `b`, `c`, `d`.

### Confusion matrix

The repository also contains `confusion_matrix.py` to compare actual vs predicted classes and produce classification metrics.

This is useful to assess which activities are confused with each other.

---

## Plotting utilities

Several scripts produce visual outputs:

- `my_doppler_plot_activities.py` — plots Doppler spectrograms for activity classes
- `CSI_doppler_plot_activities.py` — older plotting utility for activity comparisons
- `CSI_doppler_plots_antennas.py` — antenna-wise representations
- `plots_utility.py` — plotting backend functions

The scripts generate heatmaps and spectrogram figures used in experiments and reports.

---

## Shell helper scripts

### `prepare_data_for_cnn.sh`

This script shows a typical preparation pipeline for several environment folders:

- preprocess raw CSI files
- generate Doppler outputs
- prepare the dataset for training

### `run_experiments.sh`

This script iterates over multiple parameter combinations such as:

- `Tc` values
- FFT lengths
- noise thresholds

Example configuration values in the repo:

```bash
TC_VALUES=("6e-3" "8.5e-4" "9.5e-4" "1e-3")
FFT_VALUES=(100 256 1024)
NOISE_VALUES=("-0.7" "-2" "-3")
```

This makes it possible to compare experiment variants systematically.

### `run_cnn.sh`

This script automates training and evaluation for the CNN using a chosen environment and class set.

Example:

```bash
./run_cnn.sh 20 1 /path/to/root 0 1 2 3 4
```

Arguments are:

1. number of epochs
2. number of k-folds
3. dataset root directory
4. class IDs to include

---

## Typical full workflow

A standard end-to-end workflow looks like this:

### 1. Preprocess raw CSI data

```bash
python3 preprocessing.py /path/to/raw_data /path/to/data_preprocessed
```

### 2. Generate Doppler spectrograms

```bash
python3 my_doppler_computation.py /path/to/data_preprocessed/ "" /path/to/doppler_output/ 0 2000000 256 220 -0.7 --tc 8.5e-4 --fft 1024
```

### 3. Train the CNN

```bash
python3 train.py --train_env doppler_output_a --epochs 20 --root_dir /path/to/root --classes 0 1 2 3 4
```

### 4. Evaluate on unseen environments

```bash
python3 eval.py --env a --epochs 20 --classes 0 1 2 3 4
```

### 5. Generate confusion visualizations

```bash
python3 confusion_matrix.py --epochs 20 --classes 0 1 2 3 4
```

---

## Notes on practical usage

- Many scripts assume rigid folder structures and environment names such as `a`, `b`, `c`, `d`.
- Shell scripts may include hard-coded absolute paths; update them for your local machine.
- The generated Doppler outputs are usually large; ensure enough disk space and RAM for dataset caching.
- The project is experimental and parameter-sensitive; `Tc`, FFT size, SLIDING, and noise thresholds strongly influence the resulting spectrograms.

---

## Quick start summary

```bash
# 1. preprocess raw CSI files
python3 preprocessing.py /path/to/raw_data /path/to/data_preprocessed

# 2. create Doppler profiles
python3 my_doppler_computation.py /path/to/data_preprocessed/ "" /path/to/doppler_output/ 0 2000000 256 220 -0.7 --tc 8.5e-4 --fft 1024

# 3. train the CNN
python3 train.py --train_env doppler_output_a --epochs 20 --root_dir /path/to/root --classes 0 1 2 3 4

# 4. evaluate the model
python3 eval.py --env a --epochs 20 --classes 0 1 2 3 4
```

---

## Project goal

The goal of this repo is to transform CSI measurements into motion-dependent Doppler signatures and use them for activity recognition with deep learning. It sits at the intersection of wireless sensing, signal processing, and computer vision-style classification.

This makes it a useful baseline for research on human activity recognition, occupancy estimation, and environment-independent CSI classification.

---

## License and usage

This project is intended for research and internal experimentation. If you reuse it, please adapt the paths and parameters to your dataset and environment.

If you are using this repository as a starting point for a new study, be sure to review the preprocessing parameters carefully because even small changes in `Tc`, FFT size, or noise threshold can significantly affect model performance.
