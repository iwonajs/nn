
import json
import numpy as np
from numpy import linalg as LA
import os

pathx = r'C:Users\\evona\\Downloads\\Data_Large\\msr_input\\vectorized_trainedGloVe_300_paired\\vectorized\\gh-6109.npy'
print(pathx)
dir = os.path.join("C:/", "Users", "evona", "Downloads", "Data_Large", "msr_input", "vectorized_trainedGloVe_300_paired")
print(dir)
#path = r"C:\Users\evona\Downloads\Data_Large\msr_input\vectorized_trainedGloVe_300_paired\vectorized\gh-6109.npy"
path = os.path.join(dir, "vectorized", "gh-6109.npy")
print(path)
print(pathx==path)
np.load(path)
#bug_embedded = np.load(path)