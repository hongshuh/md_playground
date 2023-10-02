import os

import numpy as np
import math
import matplotlib.pyplot as plt
import numba as nb
import time
from tqdm import tqdm
from utils import read_txt_file,write_xyz_file,Kb,Epsilon_unit

temperature = 100*Kb/Epsilon_unit
dt = 0.004
timestep = 50000
box_length = 6.8
cutoff = 2.5
box = np.array([box_length,box_length,box_length])
SIGMA = 1       # file have already non-dimensionalize
EPS = 1         # energy constant in LJ

@nb.njit
def calculate_r(r1,r2):
    dr = r1-r2
    r_vec = dr-box*np.round((dr)/box)
    return r_vec
@nb.njit
def calculate_dist(r1,r2):
    return np.sqrt(np.sum((r1-r2)**2)) 

@nb.njit
def LJ(r_mag):
    r_mag = r_mag/SIGMA
    return 4  * EPS* (pow(1/r_mag, 12) - pow(r_mag, -6))

@nb.njit
def dLJ_dr(r_mag):
    r_mag = r_mag/SIGMA
    return -48  * EPS*pow(1/r_mag, 13) + 24 * pow(1/r_mag, 7)

@nb.njit
def LJ_potential(r_mag,cutoff=2.5):
    if r_mag > cutoff:
        return 0
    else:
        return LJ(r_mag) - LJ(cutoff) - (r_mag - cutoff) * dLJ_dr(cutoff)

@nb.njit
def lj_force(r_vec,cutoff=2.5):
    r_mag = np.sqrt(np.sum(r_vec**2))
    if r_mag <= 1e-8:
        print('Atoms too close, check')
    r_hat = r_vec/r_mag
    force_mag = -dLJ_dr(r_mag) + dLJ_dr(cutoff)
    force = r_hat * force_mag
    return force

def calculate_kinetic_energy(momentum):
    return np.sum(momentum**2 / 2.)

def calculate_temperature(ke, n):
    return 2*ke / (3*(n-1))

def calculate_pressure(atoms,force_array):
    ##TODO 
    return 0 

def apply_PBC(pos, box_length = 6.8):
    pos = np.mod(pos, box_length)
    return pos

def set_initial_momentum(atoms,temperature = 100.0):
    half_momentum = np.random.normal(0, np.sqrt(1.0*temperature), size=(len(atoms)//2, 3))
    momentum = np.concatenate([half_momentum,-half_momentum],axis=0)
    assert momentum.shape == atoms.shape ,'Shape of momentum should equal to position'
    return momentum

def calculate_system_state(atoms,momentum,force_array):
    kinetic_energy = calculate_kinetic_energy(momentum)
    temperature = calculate_temperature(kinetic_energy,n=len(atoms))
    pressure = calculate_pressure(atoms,force_array)
    return kinetic_energy,temperature,pressure

@nb.njit
def calculate_potential_force(atoms):
    '''
    Input: Atom Position Array [N,3]

    Return: Force vector array [N,3]
            Potential Energy   [1]
    '''
    force_array = np.zeros_like(atoms)
    total_potential = 0.0
    for i in range(len(atoms)):
        for j in range(len(atoms)):
            if j > i:
                
                r_vec = calculate_r(atoms[i],atoms[j])
                r_mag = calculate_dist(atoms[i],atoms[j])
                pair_potential = LJ_potential(r_mag)
                total_potential += pair_potential
                force_ij = lj_force(r_vec,cutoff)
                force_array[i] += force_ij
                force_array[j] -= force_ij
    return force_array,total_potential

def vv_forward(atoms,momentum,force_array,dt = 0.002,cutoff=2.5,box_length = 6.8):
    momentum = momentum + dt / 2 * force_array
    atoms += dt * momentum
    atoms = apply_PBC(atoms)
    force_array,total_potential = calculate_potential_force(atoms)
    momentum += dt / 2 * force_array
    kinetic_energy,temperature,pressure = calculate_system_state(atoms,momentum,force_array)

    return atoms,momentum,force_array,kinetic_energy,temperature,pressure,total_potential

def run_md(atoms,initial_momentum,initial_force_array):
    momentum,force_array = initial_momentum,initial_force_array
    momentum_list = []
    potential_list=[]
    kinetic_list = []
    temperature_list = []
    pressure_list = []
    atoms_list = [atoms]
    pbar = tqdm(range(timestep))
    for i in pbar:
        atoms,momentum,force_array,kinetic_energy,temperature,pressure,total_potential = vv_forward(atoms,momentum,force_array)
        momentum_list.append(momentum)
        potential_list.append(total_potential)
        kinetic_list.append(kinetic_energy)
        temperature_list.append(temperature)
        pressure_list.append(pressure)
        atoms_list.append(atoms)
        pbar.set_postfix_str({f'T : {np.round(temperature,2)}'})
    return atoms_list,momentum_list,potential_list,kinetic_list,temperature_list,pressure_list



def export_file(p, box_length=6.8):
    with open('output.xyz', 'w') as out:
        for time in p:
            out.write(str(atom_count) + '\n')
            out.write(f'Lattice="{box_length} 0.0 0.0 0.0 {box_length} 0.0 0.0 0.0 {box_length} 90.0 90.0 90.0" Properties=species:S:1:pos:R:3 Time={dt}\n')
            for atom in time:
                atom_string = atom_type
                for dir in range(0, 3):
                    atom_string += ' ' + str(atom[dir])
                out.write(atom_string)
                out.write('\n')

if __name__ == '__main__':  
    # atoms = read_txt_file('./10.txt')
    atoms = read_txt_file('./hw3/liquid256.txt')
    plt.close('all')
    # r1 = np.array([0.,2.4,0.])
    # r2 = np.array([0.0,0.,0.])
    # r_vec,r_mag = calculate_r(r1,r2)
    atom_count = len(atoms)
    atom_type = 'Z'
    initial_force,initial_potential = calculate_potential_force(atoms)
    initial_momentum = set_initial_momentum(atoms,temperature=temperature)
    atoms_list,momentum_list,potential_list,kinetic_list,temperature_list,pressure_list = run_md(atoms,initial_momentum,initial_force)
    
    export_file(atoms_list[::10])
    
    plt.figure()
    plt.plot(np.array(potential_list),label = 'Potential')
    plt.plot(np.array(kinetic_list),label = 'Kinetic')
    plt.plot(np.array(kinetic_list)+np.array(potential_list),label = 'Total')
    plt.legend()
    plt.savefig('./hw3/energy.png')

    momentum_array = np.sum(np.array(momentum_list),axis=1)
    print(momentum_array.shape)
    plt.figure()
    plt.plot(momentum_array[:,0],label='x')
    plt.plot(momentum_array[:,1],label='y')
    plt.plot(momentum_array[:,2],label='z')
    plt.legend()
    plt.savefig('./hw3/momentum.png')


    plt.figure()
    plt.plot(np.array(temperature_list))
    plt.savefig('./hw3/temperature.png')