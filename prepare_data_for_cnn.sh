#!/bin/bash

python3 preprocessing.py /media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/a /media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/data_preprocessed_a/
python3 preprocessing.py /media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/b /media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/data_preprocessed_b/
python3 preprocessing.py /media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/c /media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/data_preprocessed_c/
python3 preprocessing.py /media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/d /media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/data_preprocessed_d/

python3 my_doppler_computation.py "/media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/data_preprocessed_a/" "" "/media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/doppler_output_a/" 0 2000000 256 220 -0.7 --tc 8.5e-4 --fft 1024 
python3 my_doppler_computation.py "/media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/data_preprocessed_b/" "" "/media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/doppler_output_b/" 0 2000000 256 220 -0.7 --tc 8.5e-4 --fft 1024 
python3 my_doppler_computation.py "/media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/data_preprocessed_c/" "" "/media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/doppler_output_c/" 0 2000000 256 220 -0.7 --tc 8.5e-4 --fft 1024 
python3 my_doppler_computation.py "/media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/data_preprocessed_d/" "" "/media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/doppler_output_d/" 0 2000000 256 220 -0.7 --tc 8.5e-4 --fft 1024 