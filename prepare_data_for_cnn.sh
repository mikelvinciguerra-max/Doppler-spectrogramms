#!/bin/bash

python3 preprocessing.py /media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/01 /media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/data_preprocessed01/
python3 preprocessing.py /media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/02 /media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/data_preprocessed02/
python3 preprocessing.py /media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/04 /media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/data_preprocessed04/
python3 preprocessing.py /media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/05 /media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/data_preprocessed05/
python3 preprocessing.py /media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/06 /media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/data_preprocessed06/

python3 my_doppler_computation.py "/media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/data_preprocessed01/" "" "/media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/doppler_output01/" 0 2000000 256 220 -0.7 --tc 8.5e-4 --fft 1024 
python3 my_doppler_computation.py "/media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/data_preprocessed02/" "" "/media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/doppler_output02/" 0 2000000 256 220 -0.7 --tc 8.5e-4 --fft 1024 
python3 my_doppler_computation.py "/media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/data_preprocessed04/" "" "/media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/doppler_output04/" 0 2000000 256 220 -0.7 --tc 8.5e-4 --fft 1024 
python3 my_doppler_computation.py "/media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/data_preprocessed05/" "" "/media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/doppler_output05/" 0 2000000 256 220 -0.7 --tc 8.5e-4 --fft 1024 
python3 my_doppler_computation.py "/media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/data_preprocessed06/" "" "/media/mikel/Elements/MikelVinciguerra/dataset_PC_ehunam/doppler_output06/" 0 2000000 256 220 -0.7 --tc 8.5e-4 --fft 1024 