from sys import stdout

import matplotlib.pyplot as plt
import mdtraj
import nglview
import numpy as np
import pandas
from openmm import *
from openmm.app import *
from openmm.unit import *
import seaborn as sns


Kb = 1.3806 * pow(10, -23)     # boltmann parameter
Epsilon_unit = 1.67 * pow(10, -21)
Sigma_unit = 3.4 * pow(10, -10)
Dalton = 1.660538921 * pow(10, -27)
Argon_mass = 39.948 * Dalton

def read_txt_file(file_path):
    atoms = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            x, y, z = float(line[0]), float(line[1]), float(line[2])
            atoms.append((x, y, z))
    return np.array(atoms)


def write_xyz_file(atom_positions, filename="atoms.xyz"):
    """
    Write atomic positions to an XYZ file.

    Parameters:
    atom_positions (list): List of atomic positions, each position is a tuple (x, y, z).
    filename (str): Name of the output file.
    """
    num_atoms = len(atom_positions)
    
    with open(filename, 'w') as f:
        f.write(f"{num_atoms}\n")
        f.write("Atoms. Timestep: 0\n")
             
        for pos in atom_positions:
            x, y, z = pos
            f.write(f"A {x} {y} {z}\n")

def plot_ramachandran(traj, file,phi_atoms=None, psi_atoms=None):
    """Generate a basic Ramachandrom plot for a given trajectory.

    Parameters
    ----------
    traj
        An MDTraj trajectory object.
    phi_atoms
        A list of atom names (in order) to identify the phi angle.
        The defaults in MDTraj do not work for termini in CHARMM
        topologies, which can be fixed with this argument.
    psi_atoms
        A list of atom names (in order) to identify the psi angle.
        The defaults in MDTraj do not work for termini in CHARMM
        topologies, which can be fixed with this argument.

    """
    from matplotlib.gridspec import GridSpec
    from scipy.interpolate import interpn

    if phi_atoms is None:
        phis = mdtraj.compute_phi(traj)[1].ravel()
    else:
        phis = mdtraj.compute_dihedrals(
            traj, mdtraj.geometry.dihedral._atom_sequence(traj.topology, phi_atoms)[1]
        )
    if psi_atoms is None:
        psis = mdtraj.compute_psi(traj)[1].ravel()
    else:
        psis = mdtraj.compute_dihedrals(
            traj, mdtraj.geometry.dihedral._atom_sequence(traj.topology, psi_atoms)[1]
        )
    fig = plt.figure()
    
    x = phis * 180 / np.pi
    y = psis * 180 / np.pi
    data, x_e, y_e = np.histogram2d(x, y, bins=200)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
                data,
                np.vstack([x, y]).T,
                method="splinef2d",
                bounds_error=False,
                fill_value=0.0)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    z = np.log(z)
    mp = plt.scatter(x, y, c=z, cmap='Blues', rasterized=True)
    plt.savefig(f'./FES_{file}.png')

# traj1 = mdtraj.load("traj1.dcd", top="alanine-dipeptide.pdb")
# # traj1 = mdtraj.load_xtc('./alanine-dipeptide-0-250ns-nowater.xtc',top='./alanine-dipeptide.pdb')

# print(traj1)
# plot_ramachandran(traj1)