# CSI-Based Doppler Spectrograms for Human Activity Recognition

## Abstract

This repository implements a signal-processing and deep-learning pipeline for human activity recognition from Wi‑Fi Channel State Information (CSI). The proposed approach converts raw CSI measurements into Doppler spectrograms, which encode motion-induced frequency shifts associated with human movement. These spectrograms are then used as inputs to a convolutional neural network (CNN) for supervised classification.

The project is designed for experimental activity recognition and people-counting tasks in indoor environments, where CSI variations caused by human motion can be analyzed as time-varying Doppler signatures. The implementation includes preprocessing, Doppler extraction, dataset construction, model training, cross-environment evaluation, and visualization utilities.

## Keywords

CSI, Doppler spectrograms, Wi‑Fi sensing, human activity recognition, deep learning, convolutional neural networks, wireless signal processing.

---

## 1. Introduction

Wi‑Fi-based sensing has emerged as a non-intrusive and low-cost alternative to traditional sensing modalities such as cameras, wearable sensors, or infrared systems. In particular, CSI provides fine-grained amplitude and phase information across multiple subcarriers, enabling the observation of propagation changes induced by human motion.

The underlying hypothesis of this project is that motion patterns generate distinctive Doppler signatures, which can be represented as spectrogram-like energy distributions. These signatures can then be exploited by machine-learning models for classification tasks such as activity recognition or occupancy estimation.

This repository provides a complete experimental pipeline from raw CSI recordings to class prediction. It includes the main steps required for reproducible research in this domain: preprocessing, feature extraction, dataset construction, model training, and evaluation.

---

## 2. Problem Statement

The objective is to classify human activities or people-count states from CSI-derived Doppler features. Given a sequence of CSI measurements collected under controlled acquisition conditions, the system must extract motion-sensitive information and predict the associated activity class.

The project addresses several challenges inherent to CSI analysis:

- signal instability due to phase offsets and temporal drift
- interference and noise in wireless measurements
- variability across environments and acquisition conditions
- limited and imbalanced labelled data
- dependence on carefully tuned signal-processing parameters

---

## 3. Methodology

The methodology follows a standard CSI-to-spectrogram recognition pipeline:

1. Acquisition of raw CSI data from MATLAB files
2. Removal of invalid or non-informative subcarriers
3. Amplitude/phase reconstruction with phase calibration
4. Mean-subtraction filtering to suppress static components
5. Doppler extraction using time-windowed FFT analysis
6. Normalization and noise thresholding
7. Dataset generation from spectrogram profiles
8. CNN-based classification
9. Evaluation under cross-environment settings

---

## 4. Data and Experimental Setup

### 4.1 Input data

The raw dataset is expected to be stored as MATLAB files containing the following variables:

- CSI
- BW
- Subcarriers

Each acquisition file corresponds to a recorded wireless observation associated with a particular activity or occupancy condition. The file naming convention encodes the class or state, such as empty scenes or people-count situations.

Examples of the naming scheme include:

- E: empty or no-person scenario
- PC_ab: two-person case
- PC_abc: three-person case

The parsing logic used in the dataset loader infers labels directly from these file names.

### 4.2 Acquisition structure

The repository supports multiple data environments, typically represented by folders such as:

- a/
- b/
- c/
- d/

This organization is useful for evaluating environment generalization, where a model is trained on one environment and tested on others.

---

## 5. Preprocessing Pipeline

The preprocessing stage is implemented in `preprocessing.py` and is essential for making CSI data suitable for Doppler analysis.

### 5.1 Subcarrier filtering

The code removes unreliable subcarriers depending on the bandwidth and acquisition configuration. This step reduces the impact of contaminated frequency bins that may otherwise distort the extracted motion signatures.

### 5.2 Phase calibration

Temporal drift and phase offsets are corrected using a linear phase transformation. This operation reduces slow-varying phase distortions that otherwise mask motion-related variations in CSI.

### 5.3 Amplitude-phase reconstruction

The signal is reconstructed from its amplitude and calibrated phase components, yielding a more consistent representation for the subsequent Doppler analysis.

### 5.4 Mean subtraction

A static mean subtraction is applied to suppress constant or quasi-static channel terms that do not contain relevant motion information. This improves the sensitivity of the final representation to dynamic movement patterns.

---

## 6. Doppler Spectrogram Extraction

The core feature extraction stage is implemented in `my_doppler_computation.py`.

The Doppler pipeline performs the following operations:

- loads processed CSI matrices
- selects the relevant temporal window
- converts CSI samples into complex-valued signals
- applies a Hann window to reduce spectral leakage
- computes a Fourier transform along the temporal axis
- obtains the Doppler energy map
- normalizes the resulting feature vector
- applies a noise threshold
- stores the result as a serialized feature representation

The extraction is parameterized by several important variables:

- sample length
- sliding step
- noise threshold
- coherence time parameter Tc
- FFT size

These parameters significantly affect the resulting spectrogram resolution and classification performance. In particular, the FFT size and time-window size influence the frequency resolution and temporal localization of the Doppler signature.

Each saved output file contains a set of Doppler profiles, where each row corresponds to one spectrogram-like sample extracted from the original CSI sequence.

---

## 7. Dataset Construction

The dataset logic is implemented in `dataset.py`.

The loader scans all generated Doppler files and parses the filename to infer the associated class label. Each row of a Doppler output file becomes a distinct sample in the dataset. These samples are reshaped into a tensor of size `(1, 32, 32)`, which becomes the input to the CNN.

The dataset supports:

- lazy loading
- cached loading for RAM-efficient execution
- class filtering based on selected labels

This allows the system to work with datasets of varying size while preserving reproducibility and computational efficiency.

---

## 8. Convolutional Neural Network Architecture

The CNN architecture is defined in `model.py`.

The model consists of stacked convolutional layers with:

- Mish activation
- batch normalization
- max-pooling
- dropout regularization
- flattening
- fully connected layers for classification

This design is appropriate for spectrogram-like inputs, as it captures localized spatial patterns and frequency-energy distributions in the Doppler map.

The network output is a set of class logits, which are interpreted as predicted activity or occupancy classes. The training process uses the cross-entropy loss function and the Adam optimizer.

---

## 9. Training Procedure

The training pipeline is implemented in `train.py`.

The procedure includes:

- loading the Doppler dataset
- selecting a subset of target classes
- performing a train/validation split
- optionally running stratified cross-validation
- training the CNN over a fixed number of epochs
- saving the resulting model checkpoint

The script also computes class weights and supports evaluation on a held-out test subset. This helps mitigate class imbalance and provides a more reliable estimate of generalization performance.

The trained model is saved in the `models/` directory under a filename encoding the environment, class set, epoch count, and fold configuration.

---

## 10. Evaluation Strategy

The repository includes evaluation tools for cross-environment generalization, which is a critical requirement in real wireless sensing applications.

The evaluation process includes:

- loading a previously trained model
- testing on environments different from the training environment
- measuring classification accuracy
- producing confusion matrices to analyze class-level errors

This type of evaluation is necessary because CSI features are sensitive to propagation conditions, device position, and room geometry. A strong model should therefore be evaluated beyond a single acquisition environment.

---

## 11. Visualization and Analysis

Several utilities are provided to visualize the extracted Doppler signatures and classification outputs:

- activity-level Doppler plots
- antenna-wise comparisons
- confusion matrices
- spectrogram summaries across experiments

These visualizations are useful for both qualitative inspection and quantitative validation of the signal-processing pipeline.

---

## 12. Repository Structure

The repository contains the following main components:

- `preprocessing.py` — CSI calibration and denoising pipeline
- `my_doppler_computation.py` — Doppler feature extraction
- `dataset.py` — dataset creation and caching
- `model.py` — CNN architecture
- `train.py` — training routine
- `eval.py` — cross-environment evaluation
- `confusion_matrix.py` — confusion matrix generation
- `run_experiments.sh` — automated experiment runner
- `run_cnn.sh` — training and evaluation helper
- `prepare_data_for_cnn.sh` — example full preprocessing workflow
- `data_preprocessed/` — processed CSI files
- `doppler_output/` — extracted Doppler profiles
- `models/` — saved trained checkpoints
- `matrix/` — performance plots and confusion outputs
- `plots/` — visualization outputs

---

## 13. Usage

### 13.1 Preprocessing

```bash
python3 preprocessing.py /path/to/raw_data /path/to/preprocessed_data
```

### 13.2 Doppler extraction

```bash
python3 my_doppler_computation.py /path/to/preprocessed_data/ "" /path/to/doppler_output/ 0 2000000 256 220 -0.7 --tc 8.5e-4 --fft 1024
```

### 13.3 Training

```bash
python3 train.py --train_env doppler_output_a --epochs 20 --root_dir /path/to/root --classes 0 1 2 3 4
```

### 13.4 Training with k-fold cross-validation

```bash
python3 train.py --train_env doppler_output_a --epochs 20 --k-folds 3 --root_dir /path/to/root --classes 0 1 2 3 4
```

This performs a stratified k-fold validation on the training pool and reports the mean validation score across folds.

### 13.5 Evaluation

```bash
python3 eval.py --env a --epochs 20 --classes 0 1 2 3 4
```

### 13.6 Confusion matrix generation

```bash
python3 confusion_matrix.py --epochs 20 --classes 0 1 2 3 4
```

### 13.7 Batch training script example

```bash
./run_cnn.sh 20 3 /path/to/root 0 1 2 3 4
```

This script runs the training pipeline for a selected environment with the specified number of epochs and k-folds, then evaluates the resulting model.

---

## 14. Experimental Parameters

The project includes multiple configurable parameters that strongly influence performance:

- FFT size
- sliding window length
- selected CSI time interval
- noise threshold
- coherence-time parameter Tc
- class selection

These parameters must be tuned carefully, as variations in them may lead to different spectral resolutions and therefore materially different classification outcomes.

---

## 15. Limitations and Perspectives

Although the approach is effective in controlled experimental settings, several limitations remain:

- sensitivity to environment changes and propagation conditions
- dependence on labelled datasets and data quality
- possible degradation in performance under noisy or heterogeneous conditions
- need for careful parameter tuning for each dataset

Future work may include:

- domain adaptation to improve cross-environment robustness
- augmentation strategies for CSI data
- integration of temporal models for sequence-aware classification
- extension to more complex activity taxonomies and occupancy estimation tasks

---

## 16. Conclusion

This repository presents a practical and reproducible pipeline for CSI-based activity recognition using Doppler spectrograms and convolutional neural networks. The combination of signal processing and deep learning provides a robust foundation for motion sensing in Wi‑Fi environments.

The implementation is particularly relevant for research on wireless sensing, passive human monitoring, and non-invasive activity recognition, where CSI offers a rich but complex representation of motion-induced propagation changes.

---

## 17. License and Usage Note

This project is intended for research and experimental use. The scripts contain hard-coded paths and parameter choices that may need to be adjusted to a specific dataset or hardware setup. It is therefore recommended to review the configuration files before running large-scale experiments.
