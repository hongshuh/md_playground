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
@nb.njit
def LJ(dist):
    return 4  * (np.pow(1/dist, 12) - np.pow(dist, -6))

@nb.njit
def dLJ_dr(dist):
    return -48  * np.pow(1/dist, 13) + 24 * np.pow(1/dist, 7)

@nb.njit
def LJ_potential(dist,cutoff):
    if dist > cutoff:
        return 0
    else:
        return LJ(dist) - LJ(cutoff) - (dist - cutoff) * dLJ_dr(cutoff)




if __name__ == '__main__':  
    # atoms = read_txt_file('./liquid256.txt')
    r1 = np.array([0.0,0.0,0.0])
    r2 = np.array([1.0,5.0,2.0])
    print(calculate_distance(r1,r2))
    # print(atoms)