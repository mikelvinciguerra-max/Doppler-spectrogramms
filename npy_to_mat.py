import numpy as np
from scipy.io import savemat

mat = np.load("data/csi_matrix_processed.npy")
savemat("data/csi_matrix_processed.mat", {"csi_matrix_processed": mat})

print("shape :", mat.shape) 
