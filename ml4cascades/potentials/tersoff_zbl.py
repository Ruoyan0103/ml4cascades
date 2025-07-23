import os
import numpy as np
from ml4cascades.potentials import IPotential
from ml4cascades.lammps.calcs import CascadeCalculator

module_dir = os.path.dirname(__file__)

class TersoffZBLPotential(IPotential):
    pair_style = 'pair_style        tersoff/zbl'
    pair_coeff_template = 'pair_coeff        * * {} {}'

    def __init__(self):
        self.name = 'TersoffZBL'
        self.ff_settings = None

    def write_param(self, potential_file, element_symbol):
        self.pair_coeff = self.pair_coeff_template.format(potential_file, element_symbol)
        self.ff_settings = [self.pair_style, self.pair_coeff]

if __name__ == "__main__":
    # Initialize potential
    potential_file = os.path.join(module_dir, 'params', 'TersoffZBL', 'Ge.tersoff.zbl')
    element_symbol = 'Ge'
    
    tersoff_zbl = TersoffZBLPotential()
    tersoff_zbl.write_param(potential_file, element_symbol)

    # Simulation parameters
    mass, element, lattice, alat, temperature = 72.64, 'Ge', 'diamond', 5.76, 300

    # Select energies and directions
    energies, num_directions = [400], 500  # <- choose any config
    sizes = []
    radius_fracs = []

    for e in energies:
        if e <= 400:
            sizes.append(int(np.ceil(np.cbrt(e * 20 / 8))))
            radius_fracs.append(0.5)
        elif e <= 2000:
            sizes.append(int(np.ceil(np.cbrt(e * 30 / 8))))
            radius_fracs.append(0.8)
        elif e <= 5000:
            sizes.append(int(np.ceil(np.cbrt(e * 40 / 8))))
            radius_fracs.append(0.8)
        elif e <= 10000:
            sizes.append(int(np.ceil(np.cbrt(e * 60 / 8))))
            radius_fracs.append(0.9)
        elif e <= 50000:
            sizes.append(int(np.ceil(np.cbrt(e * 70 / 8))))
            radius_fracs.append(0.9)
    thicknesses = [alat] * len(energies)

    # Initialize calculator
    cas_calc = CascadeCalculator(
        tersoff_zbl, mass, element, lattice, alat,
        sizes, thicknesses, radius_fracs, temperature,
        energies, num_directions
    )
    
    # Run simulations and optional post-processing
    cas_calc.calculate(relax_flag=False, simulation_flag=False, check_flag=False, postprocess_flag=True)
