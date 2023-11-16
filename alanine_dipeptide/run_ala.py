import os

import mdtraj
import nglview
import numpy as np
import pandas
import torch
from openmm import *
from openmm.app import *
from openmm.unit import *
import seaborn as sns
import numpy as np
import math
import matplotlib.pyplot as plt
import numba as nb
import time
from tqdm import tqdm
from utils import read_txt_file,write_xyz_file,Kb,Epsilon_unit,Argon_mass
t_target = 300
t_steady = t_target *Kb/Epsilon_unit
temperature =  1.00 * t_steady 
dt = 0.0001
timestep = 10000
box_length = 100.0
cutoff = 100.0
box = np.array([box_length,box_length,box_length])
SIGMA = 1       # file have already non-dimensionalize
EPS = 1         # energy constant in LJ
# mass = 1
tau = 0.05      # damp coefficient for thermostat



# @torch.jit.script
def calculate_momentum(m,v):
    '''
    Input: Velocity [N,3], Mass [N]
    Return: Momentum [N,3]
    '''
    return torch.sum(m * v, axis=0)

@torch.jit.script
def calculate_r(r1,r2):
    '''
    Input: Position of two particles [3]
    Return: r_vec [3], r_mag[1]
    '''
    dr = r1-r2
    box = torch.ones(1,3) * 100.0
    r_vec = dr-box*torch.round((dr)/box)
    r_mag = torch.linalg.norm(r_vec)
    return r_vec,r_mag

def calculate_kinetic_energy(vel,mass):
    return torch.sum(mass*torch.sum(vel**2,axis=1) / 2.)

def calculate_temperature(ke, n):
    return 2*ke / (3*(n-1))

@torch.jit.script
def NonBondedPotential(r1,r2,q1,q2,sig):
    return
def apply_PBC(pos, box_length = box_length):
    new_pos = torch.remainder(pos, box_length)
    return new_pos

def set_initial_vel(pos,temperature):
    # half_vel = np.random.normal(0, np.sqrt(1.0*temperature), size=(len(pos)//2, 3))
    # vel = np.concatenate([half_vel,-half_vel],axis=0)
    vel = torch.zeros_like(pos) 
    assert vel.shape == pos.shape ,'Shape of velocity should equal to position'
    return vel

def calculate_torsion_angle(r1,r2,r3,r4):
    a = r2 - r1
    b = r3 - r2
    c = r4 - r3

    u = torch.cross(a,b)
    v = torch.cross(c,b)
    cos_theta = torch.dot(u,v)/(torch.linalg.norm(u)*torch.linalg.norm(v))
    theta = torch.acos(cos_theta)
    return theta

@torch.jit.script
def NonBondedPotential(pos,charge_,sigma_,epsilon_):
    ''' Calculate Coulumb without cutoff and LJ potential
    Args: 
        idx1,idx2 (int): index of two particles 
        pos: Position tensor of all atoms
        charge: The charges tensor 
        sigma: All Sigma parameters
        epsilon: All Epsilon parameters
    
    Return:
        NonBondedPotential 
    '''
    e_nonbonded = torch.zeros(1)
    for idx1 in range(len(pos)-1):
        for idx2 in range(idx1+1,len(pos)):
            r_vec,r_mag = calculate_r(pos[idx1],pos[idx2]) 
            e_coulumb = 1/(4 * torch.pi) * (charge_[idx1] * charge_[idx2])
            # e_coulumb=0.0
            ep = torch.sqrt(epsilon_[idx1] * epsilon_[idx2])
            sig = (sigma_[idx1] + sigma_[idx2]) / 2 
            e_lj = 4 * ep * ((sig/r_mag)**12 - (sig/r_mag)**6 )

            pair_nonbond = e_lj + e_coulumb 
            e_nonbonded += pair_nonbond
    return e_nonbonded

@torch.jit.script
def HarmonicAnglePotential(pos,angle_):
    '''
    
    Args:
        pos: position of all particles 
        angle_ : list of angle parameters of [idx1, idx2, idx3, theta_0, k]

    '''

    e_angle = torch.zeros(1)
    for angle in angle_:
        idx1,idx2,idx3 = angle[0],angle[1],angle[2]
        idx1,idx2,idx3 = idx1.to(torch.int),idx2.to(torch.int),idx3.to(torch.int)

        theta0 = angle[3]
        k = angle[4]
        dr21, r21norm =calculate_r(pos[idx2],pos[idx1])
        dr23, r23norm =calculate_r(pos[idx2],pos[idx3])

        theta = torch.acos((dr21@dr23.T)/(r21norm * r23norm))

        e = 1/2 * k * (theta - theta0) ** 2
        e_angle += e.view(-1)
    return e_angle

@torch.jit.script
def HarmonicBondPotential(pos,bond_):

    e_bond = torch.zeros(1)
    for bond in bond_:
        idx1,idx2 = bond[0],bond[1]
        idx1,idx2 = idx1.to(torch.int),idx2.to(torch.int)
        length = bond[2]
        k = bond[3]
        r_vec,r_mag = calculate_r(pos[idx1],pos[idx2])
        pair_bond = 1/2 * k * (r_mag - length)**2
        e_bond += pair_bond
    return e_bond

@torch.jit.script
def PeriodicTorsionPotential(pos,torsion_):

    e_torsion = torch.zeros(1)
    for torsion in torsion_:
        idx1,idx2,idx3,idx4 = torsion[0],torsion[1],torsion[2],torsion[3]
        idx1,idx2,idx3,idx4 = idx1.to(torch.int),idx2.to(torch.int),idx3.to(torch.int),idx4.to(torch.int)
        n = torsion[4]
        theta0 = torsion[5]
        k = torsion[6]
        theta = calculate_torsion_angle(pos[idx1],pos[idx2],pos[idx3],pos[idx4])
        e = k * (1 + torch.cos(n * theta - theta0))
        e_torsion += e
    return e_torsion

@torch.jit.script
def All_potential(pos,charge_ , sigma_ , epsilon_ ,bond_,angle_,torsion_):
    e_nb = NonBondedPotential(pos,charge_,sigma_,epsilon_)
    e_bond = HarmonicBondPotential(pos,bond_)
    e_angle = HarmonicAnglePotential(pos,angle_)
    e_torsion = PeriodicTorsionPotential(pos,torsion_)

    e_total = e_nb + e_bond + e_angle + e_torsion
    return e_total


def pair_loop(pos,charge_ , sigma_ , epsilon_ ,bond_,angle_,torsion_):
    pos = torch.tensor(pos,requires_grad=True)
    potential = All_potential(pos,charge_ , sigma_ , epsilon_ ,bond_,angle_,torsion_)
    potential.backward()
    # print(pos)
    force = -pos.grad

    pos.grad.data.zero_()
    return force, potential

def vv_forward_thermostat(pos,vel,force,xi,tau=tau,dt = dt):
    vel = vel + dt / 2 * force/mass
    pos = pos + dt * vel
    # pos = apply_PBC(pos)
    force,potential = pair_loop(pos,charge_ , sigma_ , epsilon_ ,bond_,angle_,torsion_)
    e_kinetic = calculate_kinetic_energy(vel,mass)
    temperature = calculate_temperature(e_kinetic,atom_count)


    xi = xi + dt * 1/tau**2 * (temperature/t_steady - 1)
    vel = vel + dt / 2 * force/mass
    vel = vel / (1 + dt / 2 * xi)
    e_kinetic = calculate_kinetic_energy(vel,mass)
    temperature = calculate_temperature(e_kinetic,atom_count)
    
    return pos,vel,force,potential,e_kinetic,temperature,xi


def run_md(pos,initial_vel,initial_force_array,initial_potential,xi = 0):
    vel,force = initial_vel,initial_force_array
    vel_list = [initial_vel.detach().numpy().tolist()]
    potential_list=[initial_potential.detach().numpy().tolist()]
    kinetic_energy = calculate_kinetic_energy(vel,mass)
    temperature = calculate_temperature(kinetic_energy,atom_count)
    kinetic_list = [kinetic_energy]
    temperature_list = [temperature]
    pos_list = [pos]
    pbar = tqdm(range(timestep))
    thermostat = True
    
    for i in pbar:
        
        if thermostat == True:
            pos,vel,force,total_potential,kinetic_energy,temperature,xi\
                = vv_forward_thermostat(pos,vel,force,xi,tau=tau,dt = dt)
            

            # if i*dt > 200.0:
            #     # thermostat = check_eq(kinetic_list,potential_list,temperature_list,thermostat,i)
            #     thermostat = False
        # else:
        #     pos,vel,force_array,kinetic_energy,temperature,pressure,total_potential,disp \
        #         = vv_forward(pos,vel,force_array)
        #     total_disp += disp

        vel_list.append(vel.detach().numpy().tolist())
        potential_list.append(total_potential.detach().numpy().tolist())
        kinetic_list.append(kinetic_energy.detach().numpy().tolist())
        temperature_list.append(temperature.detach().numpy().tolist())
        
        pos_list.append(pos)
        pbar.set_postfix_str({f'Thermostat':thermostat
                               ,'Temp':np.round(temperature.detach().numpy().tolist(),2) 
                               ,'Total E': np.round(total_potential.detach().numpy()+kinetic_energy.detach().numpy(),2)
                               ,'Potential':np.round(total_potential.detach().numpy(),2)})
    return pos_list,vel_list,potential_list,kinetic_list,temperature_list



def export_file(p, box_length=box_length):
    with open('output.xyz', 'w') as out:
        for time in p:
            out.write(str(atom_count) + '\n')
            out.write(f'Lattice="{box_length} 0.0 0.0 0.0 {box_length} 0.0 0.0 0.0 {box_length} 90.0 90.0 90.0" Properties=species:S:1:pos:R:3 Time={dt}\n')
            for i in range(len(time)):
                atom_string = atom_name[i]
                for dir in range(0, 3):
                    atom_string += ' ' + str(time[i][dir])
                out.write(atom_string)
                out.write('\n')
def input_pdb(path):
    pdb = PDBFile(path)
    pos = pdb.getPositions(asNumpy=True).value_in_unit(angstrom)
    atom_name = []
    mass = []
    for atom in pdb.topology.atoms():
        atom_name.append(atom.name)
        mass.append(atom.element.mass.value_in_unit(dalton))
    
    mass_array = np.expand_dims(np.array(mass),axis=1) # reshape mass array into 2d
    pos = torch.tensor(pos,requires_grad=True)
    mass = torch.tensor(mass_array)
    return pdb,pos,atom_name,mass

def input_parameters():
    path = './parameters'
    charge_ = np.load(os.path.join(path,'charges.npy'),allow_pickle=True)
    sigma_ = np.load(os.path.join(path,'sigma.npy'),allow_pickle=True)
    epsilon_ = np.load(os.path.join(path,'epsilon.npy'),allow_pickle=True)
    bond_ = np.load(os.path.join(path,'bonds.npy'),allow_pickle=True)
    angle_ = np.load(os.path.join(path,'angles.npy'),allow_pickle=True)
    torsion_ = np.load(os.path.join(path,'torsions.npy'),allow_pickle=True)


    return torch.tensor(charge_) , torch.tensor(sigma_) , \
        torch.tensor(epsilon_),torch.tensor(bond_),torch.tensor(angle_),torch.tensor(torsion_)
def plot_everything(vel_list,potential_list,kinetic_list,temperature_list,path):

    log_path = os.path.join(path,'log.npz')
    vel_array = np.array(vel_list)
    potential_array = np.array(potential_list).squeeze()
    kinetic_array = np.array(kinetic_list)
    temperature_array = np.array(temperature_list)
    # pressure_array = np.array(pressure_list)
    # msd_array = np.array(msd_list)
    # single_potential_array = np.array(single_potential_list)
    log_data = {
        'vel':vel_array,
        'potential': potential_array,
        'kinetic' : kinetic_array,
        'temperature': temperature_array,
        # 'pressure': pressure_array,
        # 'msd':msd_array,
        # 'pe_single':single_potential_array
    }
    np.savez(log_path,**log_data)
    
    # plt.figure()
    # plt.plot(msd_array)
    # plt.xlabel('Time Step')
    # plt.ylabel('MSD')
    # plt.savefig(os.path.join(path,'msd.png'))

    plt.figure()
    plt.plot(potential_array,label = 'Potential')
    plt.plot(kinetic_array,label = 'Kinetic')
    plt.plot(kinetic_array+potential_array,label = 'Total')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.legend()
    plt.savefig(os.path.join(path,'energy.png'))

    # momentum_array = np.sum(np.array(vel_list*mass),axis=1)
    # print(momentum_array.shape)
    # plt.figure()
    # plt.plot(momentum_array[:,0],label='x')
    # plt.plot(momentum_array[:,1],label='y')
    # plt.plot(momentum_array[:,2],label='z')
    # plt.xlabel('Time')
    # plt.ylabel('Momentum')
    # plt.ylim(-1e-10,1e-10)
    # plt.legend()
    # plt.savefig(os.path.join(path,'momentum.png'))


    plt.figure()
    plt.plot(temperature_array,label='Simulation temperature')
    plt.plot([0.,len(temperature_array)],[t_steady,t_steady],label = 'Target Steady Temperature')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend()
    plt.savefig(os.path.join(path,'temperature.png'))


    # plt.figure()
    # plt.plot(pressure_array)
    # plt.xlabel('Time')
    # plt.ylabel('Pressure')
    # # plt.legend()
    # plt.savefig(os.path.join(path,'pressure.png'))
    return
if __name__ == '__main__':  
    # atoms = read_txt_file('./10.txt')
    pdb,pos,atom_name,mass = input_pdb('alanine-dipeptide.pdb')
    charge_ , sigma_ , epsilon_ ,bond_,angle_,torsion_= input_parameters()
    path = './data'
    os.makedirs(path,exist_ok=True)
    atom_count = len(pos)
    print(pos)

    initial_vel = set_initial_vel(pos,temperature=temperature)
    initial_force,initial_potential = pair_loop(pos,charge_ , sigma_ , epsilon_ ,bond_,angle_,torsion_)
    kinetic_energy = calculate_kinetic_energy(initial_vel,mass)
    temperature = calculate_temperature(kinetic_energy,atom_count)
    


    print('============================================================')
    print(f'Simulating a LJ system with {pos.shape[0]} particles')
    print(f'Init momentum (x, y, z): {calculate_momentum(initial_vel,mass)}')
    # print(f'Initial Pressure of the system is {pressure}')
    print(f'Initial Potential of the system is {initial_potential}')
    print(f'Initial Kinetic Energy is {kinetic_energy}')
    print(f'Simulating for {timestep} steps using step size: {dt}, {timestep*dt} unit time')
    print(f'Targetting a dimensionless temperature: {t_steady}')
    # print(f'Initial density (kg/m3): {density}')
    print(f'Using a box with length: {box_length} and cutoff: {cutoff }')
    print(f'All files will be save into folder: {path}')
    print('====================Go !!!===================================')
    # exit()
    pos_list,vel_list,potential_list,kinetic_list,temperature_list\
        = run_md(pos,initial_vel,initial_force,initial_potential)
    
    # xi_array = np.array(xi_list)
    # plt.figure()
    # plt.plot(xi_array)
    # plt.savefig('test.png')


    export_file(p=pos_list[::10])
    
    plot_everything(vel_list,potential_list,kinetic_list,temperature_list,path)
