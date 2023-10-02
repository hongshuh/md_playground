import numpy as np
import math



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


if __name__ == '__main__':
    print(100*Kb/Epsilon_unit)