import os

import numpy as np
import math
import matplotlib.pyplot as plt
import numba as nb
import time
from tqdm import tqdm
from utils import read_txt_file,write_xyz_file,Kb,Epsilon_unit,Argon_mass
t_target = 100
t_steady = t_target *Kb/Epsilon_unit
temperature =  1.00 * t_steady 
# dt = 0.01
steps = 1000000
box_length = 6.8
cutoff = 2.5
box = np.array([box_length,box_length,box_length])
SIGMA = 1       # file have already non-dimensionalize
EPS = 1         # energy constant in LJ
mass = 1
tau = 0.05      # damp coefficient for thermostat



@nb.njit
def calculate_momentum(v,m):
    '''
    Input: Velocity [N,3], Mass [N]
    Return: Momentum [N,3]
    '''
    return np.sum(m * v, axis=0)
@nb.njit
def calculate_msd(disp):
    '''
    Input: Atom displacement [N,3]

    Return: Mean Square Displacement [1]
    '''
    
    return np.mean(np.sum(disp**2,axis=1))
@nb.njit
def calculate_r(r1,r2):
    '''
    Input: Position of two particles [3]
    Return: r_vec [3], r_mag[1]
    '''
    dr = r1-r2
    r_vec = dr-box*np.round((dr)/box)
    r_mag = np.sqrt(np.sum((r_vec)**2))
    return r_vec,r_mag

@nb.njit
def LJ(r_mag):
    r_mag = r_mag/SIGMA
    return 4  * EPS* (pow(1/r_mag, 12) - pow(1/r_mag, 6))

@nb.njit
def dLJ_dr(r_mag):
    r_mag = r_mag/SIGMA
    return -48  * EPS*pow(1/r_mag, 13) + 24 * pow(1/r_mag, 7)

@nb.njit
def LJ_potential(r_mag,cutoff=cutoff):
    if r_mag > cutoff:
        return 0.0
    return LJ(r_mag) - LJ(cutoff) - (r_mag - cutoff) * dLJ_dr(cutoff)

@nb.njit
def lj_force(r_vec,cutoff=cutoff):
    r_mag = np.sqrt(np.sum(r_vec**2))
    if r_mag > cutoff:
        return np.array([0.,0.,0.]),0
    if r_mag <= 1e-8:
        print('Atoms too close, check')
    r_hat = r_vec/r_mag
    force_mag = -dLJ_dr(r_mag) + dLJ_dr(cutoff)
    force = r_hat * force_mag
    return force,force_mag



def apply_PBC(pos, box_length = box_length):
    new_pos = np.mod(pos, box_length)
    return new_pos




def export_file(p, box_length=box_length):
    with open('output.xyz', 'w') as out:
        for time in p:
            out.write(str(atom_count) + '\n')
            out.write(f'Lattice="{box_length} 0.0 0.0 0.0 {box_length} 0.0 0.0 0.0 {box_length} 90.0 90.0 90.0"\n')
            for atom in time:
                atom_string = atom_type
                for dir in range(0, 3):
                    atom_string += ' ' + str(atom[dir])
                out.write(atom_string)
                out.write('\n')

# @nb.njit
def one_step(pos,single_potential,total_pressure):
    idx = np.random.choice(len(pos),1)
    
    new_pos = pos.copy()
    perturb = 0.5 * (np.random.rand(1,3) - 0.5) 
    new_pos[idx] += perturb
    new_pos = apply_PBC(new_pos)
    
    force_array,new_total_potential,new_total_pressure,new_single_potential,new_single_pressure = all_loop(new_pos)
    
    total_potential = np.sum(single_potential)

    deltaE = new_total_potential - total_potential
    if deltaE <= 0.0: 
        return new_pos,new_total_potential,new_total_pressure
    elif np.random.uniform(0.0,1.0) < np.exp(-deltaE / t_steady):
        # print(np.exp(-deltaE / t_steady))
        return new_pos,new_single_potential,new_total_pressure
    else:
        return pos,single_potential,total_pressure

@nb.njit
def all_loop(pos):
    '''
    Input: Atom Position Array [N,3]

    Return: Force vector array [N,3]
            Potential Energy   [1]
            Pressure [1]
    '''
    #TODO Return pair potential to calculate heat capacity
    force_array = np.zeros_like(pos)
    single_potential = np.zeros(len(pos))
    single_pressure = np.zeros(len(pos))
    total_potential = 0.0
    total_pressure = atom_count * t_steady / box_length**3
    for i in range(len(pos)-1):
        for j in range(i+1,len(pos)):
            r_vec,r_mag = calculate_r(pos[i],pos[j])                
            pair_potential = LJ_potential(r_mag)
            total_potential += pair_potential
            single_potential[i] += pair_potential/2
            single_potential[j] += pair_potential/2
            force_ij ,f_mag= lj_force(r_vec,cutoff)
            total_pressure += 1/3/box_length**3 * r_vec @ (force_ij)
            single_pressure[i] += 1/3/box_length**3 * r_vec @ (force_ij)/2
            single_pressure[j] += 1/3/box_length**3 * r_vec @ (force_ij)/2

            force_array[i] += force_ij
            force_array[j] -= force_ij
    return force_array,total_potential,total_pressure,single_potential,single_pressure


def run_mc(pos,initial_pressure,initial_potential,single_potential,single_pressure,steps):
    pos_list = [pos]
    pressure_list = [initial_pressure]
    potential_list = [initial_potential]
    pbar = tqdm(range(steps))
    total_pressure = initial_pressure
    for i in pbar:
        pos,single_potential,total_pressure = one_step(pos,single_potential,total_pressure)
        pos_list.append(pos)
        potential_list.append(np.sum(single_potential))
        pressure_list.append(total_pressure)
        pbar.set_postfix_str(f'Potential : {np.round(np.sum(single_potential),4)}, Pressure: {np.round(total_pressure, 2)}')
    return pos_list,potential_list,pressure_list

def plot_everything(potential_list,pressure_list,path):

    log_path = os.path.join(path,'log.npz')
   
    potential_array = np.array(potential_list)
    
    pressure_array = np.array(pressure_list)
    
    log_data = {
        'potential': potential_array,
        'pressure': pressure_array,
     
    }
    np.savez(log_path,**log_data)
    

    plt.figure()
    plt.plot(potential_array,label = 'Potential')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.legend()
    plt.savefig(os.path.join(path,'energy.png'))


    plt.figure()
    plt.plot(pressure_array)
    plt.xlabel('Time')
    plt.ylabel('Pressure')
    # plt.legend()
    plt.savefig(os.path.join(path,'pressure.png'))
    return
if __name__ == '__main__':  
    # atoms = read_txt_file('./10.txt')
    pos = read_txt_file('./hw6/liquid256.txt')
    path = './hw6'
    os.makedirs(path,exist_ok=True)
    plt.close('all')
    atom_count = len(pos)
    atom_type = 'Z'

    
    # initial_vel = set_initial_vel(pos,temperature=temperature)
    # initial_force,initial_potential,pressure,single_potential = pair_loop(pos)
    # kinetic_energy,temperature = calculate_system_state(pos,initial_vel)
    # pressure += atom_count * temperature / box_length**3

    force_array,initial_potential,initial_pressure,single_potential,single_pressure = all_loop(pos)


    print('============================================================')
    print(f'Simulating a LJ system with {pos.shape[0]} particles')
    print(f'Initial Potential is {initial_potential}')
    print(f'Initial Pressure is {initial_pressure}')
    # print(f'Simulating for {timestep} steps using step size: {dt}, {timestep*dt} unit time')
    print(f'Targetting a dimensionless temperature: {t_steady}')
    # print(f'Initial density (kg/m3): {density}')
    print(f'Using a box with length: {box_length} and cutoff: {cutoff }')
    print(f'All files will be save into folder: {path}')
    print('====================Go !!!===================================')

    
    
    pos_list,potential_list,pressure_list = run_mc(pos,initial_pressure,initial_potential,single_potential,single_pressure,steps=steps)
    
    export_file(p=pos_list[::10])
    
    plot_everything(potential_list,pressure_list,path)
