#!/bin/bash

HANN_SIZE=256
SLIDING=220
START_COMP=0
END_COMP=2000000
START_PLT=0
END_PLT=2000000

DIR_DATA="data_preprocessed/"
SUBDIRS="" 

TC_VALUES=("6e-3" "8.5e-4" "9.5e-4" "1e-3")
FFT_VALUES=(100 256 1024)
NOISE_VALUES=("-0.7" "-2" "-3")

for tc in "${TC_VALUES[@]}"; do
    for fft in "${FFT_VALUES[@]}"; do
        for noise in "${NOISE_VALUES[@]}"; do
            CONFIG_NAME="Tc_${tc}_FFT_${fft}_Noise_${noise}"
            
            echo "=========================================================="
            echo "Starting configuration: $CONFIG_NAME"
            echo "=========================================================="
            
            OUT_DOPPLER="doppler_output_${CONFIG_NAME}/"
            OUT_PLOTS="plots_${CONFIG_NAME}/"
            
            mkdir -p "$OUT_DOPPLER"
            
            python3 my_doppler_computation.py "$DIR_DATA" "$SUBDIRS" "$OUT_DOPPLER" \
                $START_COMP $END_COMP $HANN_SIZE $SLIDING "$noise" \
                --tc "$tc" --fft "$fft"
            
            python3 my_doppler_plot_activities.py "${OUT_DOPPLER}${SUBDIRS}" "$OUT_PLOTS" \
                "$fft" $SLIDING $START_PLT $END_PLT \
                --tc "$tc"
                
            echo "Configuration $CONFIG_NAME completed."
            echo "----------------------------------------------------------"
            
        done
    done
done
echo "All experiments have been successfully generated!"