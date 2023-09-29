import os

import numpy as np
import math
import matplotlib.pyplot as plt
import numba as nb
import time
from tqdm import tqdm
from utils import read_txt_file,write_xyz_file

Temperature = 100
dt = 0.002
timestep = 100
box_length = 6.8
cutoff = 2.5
box = np.array([box_length,box_length,box_length])

@nb.njit
def calculate_distance(r1,r2):
    return r1-r2-box*np.round((r1-r2)/box)





if __name__ == '__main__':  
    # atoms = read_txt_file('./liquid256.txt')
    r1 = np.array([0.0,0.0,0.0])
    r2 = np.array([1.0,5.0,2.0])
    print(calculate_distance(r1,r2))
    # print(atoms)