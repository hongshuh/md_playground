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
temperature = 1.25* t_steady 
dt = 0.01
timestep = 20000
box_length = 6.8
cutoff = 2.5
box = np.array([box_length,box_length,box_length])
SIGMA = 1       # file have already non-dimensionalize
EPS = 1         # energy constant in LJ
mass = 1
@nb.njit
def calculate_momentum(v,m):
    '''
    Input: Velocity [N,3], Mass [N]
    Return: Momentum [N,3]
    '''
    return np.sum(m * v, axis=0)

@nb.njit
def calculate_r(r1,r2):
    '''
    Input: Position of two particles [3]
    Return: r_vec [3], r_mag[1]
    '''
    dr = r1-r2
    r_vec = dr-box*np.round((dr)/box)
    r_mag = np.sqrt(np.sum((dr)**2)) 
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

def calculate_kinetic_energy(vel,mass):
    return np.sum(mass*vel**2 / 2.)

def calculate_temperature(ke, n):
    return 2*ke / (3*(n-1))


def apply_PBC(pos, box_length = box_length):
    pos = np.mod(pos, box_length)
    return pos

def set_initial_vel(atoms,temperature = 100.0):
    half_vel = np.random.normal(0, np.sqrt(1.0*temperature), size=(len(atoms)//2, 3))
    vel = np.concatenate([half_vel,-half_vel],axis=0)
    assert vel.shape == atoms.shape ,'Shape of velocity should equal to position'
    return vel


def calculate_system_state(atoms,vel,force_array):
    kinetic_energy = calculate_kinetic_energy(vel,mass)
    temperature = calculate_temperature(kinetic_energy,n=len(atoms))
    
    return kinetic_energy,temperature



@nb.njit
def pair_loop(atoms,temperature):
    '''
    Input: Atom Position Array [N,3]

    Return: Force vector array [N,3]
            Potential Energy   [1]
            Pressure [1]
    '''
    force_array = np.zeros_like(atoms)
    total_potential = 0.0
    # kinetic_energy,temperature = calculate_system_state(atoms,vel,force_array)
    total_pressure = atom_count * temperature / box_length**3
    for i in range(len(atoms)):
        for j in range(len(atoms)):
            if j > i:
                r_vec,r_mag = calculate_r(atoms[i],atoms[j])
                # r_mag = calculate_dist(atoms[i],atoms[j])
                
                pair_potential = LJ_potential(r_mag)
                total_potential += pair_potential
                force_ij ,f_mag= lj_force(r_vec,cutoff)
                total_pressure += 1/3/box_length**3 * r_vec@(force_ij)
                force_array[i] += force_ij
                force_array[j] -= force_ij
    return force_array,total_potential,total_pressure

def vv_forward(atoms,vel,force_array,dt = dt,cutoff=cutoff,box_length = box_length):
    vel = vel + dt / 2 * force_array/mass
    atoms += dt * vel
    atoms = apply_PBC(atoms)
    kinetic_energy,temperature = calculate_system_state(atoms,vel,force_array)
    force_array,total_potential,pressure = pair_loop(atoms,temperature)
    vel += dt / 2 * force_array/mass
    kinetic_energy,temperature = calculate_system_state(atoms,vel,force_array)

    return atoms,vel,force_array,kinetic_energy,temperature,pressure,total_potential

def run_md(atoms,initial_vel,initial_force_array,initial_potential,pressure):
    vel,force_array = initial_vel,initial_force_array
    vel_list = [initial_vel]
    potential_list=[initial_potential]
    kinetic_energy,temperature = calculate_system_state(atoms,vel,force_array)
    kinetic_list = [kinetic_energy]
    temperature_list = [temperature]
    pressure_list = [pressure]
    atoms_list = [atoms]
    pbar = tqdm(range(timestep))
    for i in pbar:
        atoms,vel,force_array,kinetic_energy,temperature,pressure,total_potential = vv_forward(atoms,vel,force_array)
        vel_list.append(vel)
        potential_list.append(total_potential)
        kinetic_list.append(kinetic_energy)
        temperature_list.append(temperature)
        pressure_list.append(pressure)
        atoms_list.append(atoms)
        pbar.set_postfix_str(({f'Temperature':np.round(temperature,2) 
                               ,'Total E': np.round(total_potential+kinetic_energy,2)
                               ,'Potential':np.round(total_potential,2)
                               ,'Pressure': np.round(pressure,2)}))
    return atoms_list,vel_list,potential_list,kinetic_list,temperature_list,pressure_list



def export_file(p, box_length=box_length):
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


def plot_everything(vel_list,potential_list,kinetic_list,temperature_list,pressure_list,path):

    log_path = os.path.join(path,'log.npz')
    vel_array = np.array(vel_list)
    potential_array = np.array(potential_list)
    kinetic_array = np.array(kinetic_list)
    temperature_array = np.array(temperature_list)
    pressure_array = np.array(pressure_list)
    log_data = {
        'vel':vel_array,
        'potential': potential_array,
        'kinetic' : kinetic_array,
        'temperature': temperature_array,
        'pressure': pressure_array
    }
    np.savez(log_path,**log_data)
    


    plt.figure()
    plt.plot(potential_array,label = 'Potential')
    plt.plot(kinetic_array,label = 'Kinetic')
    plt.plot(kinetic_array+potential_array,label = 'Total')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.legend()
    plt.savefig('./hw3/energy.png')

    momentum_array = np.sum(np.array(vel_list*mass),axis=1)
    print(momentum_array.shape)
    plt.figure()
    plt.plot(momentum_array[:,0],label='x')
    plt.plot(momentum_array[:,1],label='y')
    plt.plot(momentum_array[:,2],label='z')
    plt.xlabel('Time')
    plt.ylabel('Momentum')
    plt.ylim(-1e-10,1e-10)
    plt.legend()
    plt.savefig('./hw3/momentum.png')


    plt.figure()
    plt.plot(temperature_array,label='Simulation temperature')
    plt.plot([0.,len(temperature_array)],[t_steady,t_steady],label = 'Target Steady Temperature')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend()
    plt.savefig('./hw3/temperature.png')


    plt.figure()
    plt.plot(pressure_array)
    plt.xlabel('Time')
    plt.ylabel('Pressure')
    # plt.legend()
    plt.savefig('./hw3/pressure.png')
    return
if __name__ == '__main__':  
    # atoms = read_txt_file('./10.txt')
    atoms = read_txt_file('./hw3/liquid256.txt')
    path = './hw3'
    plt.close('all')
    # r1 = np.array([0.,2.4,0.])
    # r2 = np.array([0.0,0.,0.])
    # r_vec,r_mag = calculate_r(r1,r2)
    atom_count = len(atoms)
    atom_type = 'Z'
    initial_vel = set_initial_vel(atoms,temperature=temperature)
    initial_force,initial_potential,pressure = pair_loop(atoms,temperature)

    print('============================================================')
    print(f'Simulating a LJ system with {atoms.shape[0]} particles')
    print(f'Init momentum (x, y, z): {calculate_momentum(initial_vel,mass)}')
    print(f'Initial Pressure of the system is {pressure}')
    print(f'Initial Potential of the system is {initial_potential}')

    print(f'Simulating for {timestep} steps using step size: {dt}, {timestep*dt} unit time')
    print(f'Targetting a dimensionless temperature: {t_steady}')
    # print(f'Initial density (kg/m3): {density}')
    print(f'Using a box with length: {box_length} and cutoff: {cutoff }')
    print(f'All files will be save into folder: {path}')
    print('====================Go !!!===================================')
    # exit()
    atoms_list,momentum_list,potential_list,kinetic_list,temperature_list,pressure_list = run_md(atoms,initial_vel,initial_force,initial_potential,pressure)
    
    export_file(p=atoms_list[::10])
    
    plot_everything(momentum_list,potential_list,kinetic_list,temperature_list,pressure_list,path)
