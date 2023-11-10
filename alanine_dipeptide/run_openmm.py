import matplotlib.pyplot as plt
import mdtraj
import nglview
import numpy as np
import pandas
from openmm import *
from openmm.app import *
from openmm.unit import *
from sys import stdout
from utils import plot_ramachandran
def gas():
    pdb = PDBFile("alanine-dipeptide.pdb")
    print(pdb.topology)
    forcefield = ForceField("amber14-all.xml")
    system = forcefield.createSystem(pdb.topology, nonbondedCutoff=3 * nanometer, constraints=HBonds)
    integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 2 * femtoseconds)
    simulation = Simulation(pdb.topology, system, integrator,platform=Platform.getPlatformByName('CUDA'))
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()
    
    # Production run
    simulation.reporters = []
    simulation.reporters.append(DCDReporter("traj1.dcd", 1000))
    simulation.reporters.append(
        StateDataReporter(stdout, 1000, step=True, temperature=True, elapsedTime=True)
    )
    simulation.reporters.append(
        StateDataReporter(
            "scalars1.csv",
            1000,
            time=True,
            potentialEnergy=True,
            totalEnergy=True,
            temperature=True,
        )
    )
    simulation.step(125000000)
    return


def implicit():
    pdb = PDBFile("alanine-dipeptide.pdb")

    forcefield = ForceField("amber99sbnmr.xml", "amber99_obc.xml")
    system = forcefield.createSystem(pdb.topology, nonbondedCutoff=3 * nanometer, constraints=HBonds)
    integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 2 * femtoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()

    simulation.reporters = []
    simulation.reporters.append(DCDReporter("traj_implicit.dcd", 100))
    simulation.reporters.append(
        StateDataReporter(stdout, 1000, step=True, temperature=True, elapsedTime=True)
    )
    simulation.reporters.append(
        StateDataReporter(
            "scalars2.csv",
            100,
            time=True,
            potentialEnergy=True,
            totalEnergy=True,
            temperature=True,
        )
    )
    simulation.step(125000000)
    return


if __name__ == '__main__':
    implicit()
    traj = mdtraj.load("traj_implicit.dcd", top="alanine-dipeptide.pdb")
    print(traj)
    plot_ramachandran(traj,file = 'implicit')