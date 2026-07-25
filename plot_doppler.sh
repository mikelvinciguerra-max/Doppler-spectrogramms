#!/bin/bash

python3 my_doppler_computation.py data_preprocessed/ "" doppler_output 0 2000000 128 32 -0.9
python3 my_doppler_plot_activities.py doppler_output plots 100 8 0 2000000