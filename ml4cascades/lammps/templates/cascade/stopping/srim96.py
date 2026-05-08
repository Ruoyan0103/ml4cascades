import numpy as np

MASS_AMU_TO_KG = 1.66053906660e-27
JOULE_TO_EV = 6.242e18

def srim96_stopping_file(stoppingfile: str, mass: float, new_stoppingfile: str):
    data = np.loadtxt(stoppingfile)
    velocity = data[:, 0] 
    stopping_power = data[:, 1]
    kinetic_energy = 0.5 * mass*MASS_AMU_TO_KG * velocity**2 * JOULE_TO_EV
    with open(new_stoppingfile, 'w') as f:
        f.write("# This file is generated from SRIM96 stopping power data\n")
        f.write("# Kinetic energy is calculated from velocity using mass = {} amu\n".format(mass))
        f.write("# eV    eV/Ang     # units metal\n")
        for v, s in zip(kinetic_energy, stopping_power):
            f.write(f"{v}  {s}\n")

srim96_stopping_file('elstop.in.10keV_srim96.Ge', mass=72.56, new_stoppingfile='Ge_Ge_elstop_srim96.txt')