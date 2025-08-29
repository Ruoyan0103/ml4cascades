import os
import xml.etree.ElementTree as ET
from ml4cascades.potentials import IPotential
from ml4cascades.lammps.calcs import CascadeCalculator
import numpy as np
module_dir = os.path.dirname(__file__)

class MEAMPotential(IPotential):
    pair_style = 'pair_style        meam'
    pair_coeff = 'pair_coeff        * * {} {} {} {}'


    def __init__(self):
        self.name = 'MEAM'
        self.ff_settings = None


    def write_param(self, library_filename, element_filename, element_symbol):
        self.pair_coeff = self.pair_coeff.format(library_filename, element_symbol,
                                                 element_filename, element_symbol)
        self.ff_settings = [self.pair_style, self.pair_coeff]


if __name__ == "__main__":
    library_file = os.path.join(module_dir, 'params', 'MEAM', 'library.meam')
    element_file = os.path.join(module_dir, 'params', 'MEAM', 'Ge.meam')
    element_symbol = 'Ge'
    meam = MEAMPotential()
    meam.write_param(library_file, element_file, element_symbol)
    
    mass, element, lattice, alat, temperature = 72.64, 'Ge', 'diamond', 5.76, 300    

    # 100, 400, 1000 ok with e*30/8, thickness alat 
    # 2000, 5000, 10e3 ok with e*40/8 tickness alat+3
    # energies, num_directions = [2000, 5000, 10e3, 20e3, 50e3], 59
    # energies, num_directions = [2000, 5000, 10e3], 59
    energies, num_directions = [100], 500
    # energies, num_directions = [1000, 2000], 500
    # energies, num_directions = [5000], 500
    # energies, num_directions = [10e3], 500
    sizes = [] 
    radius_fracs = []
    for e in energies:
        if e < 400:
            sizes.append(int(np.ceil(np.cbrt(e*20/8))))
            radius_fracs.append(0.5)
        elif e >= 400 and e < 2000:
            sizes.append(int(np.ceil(np.cbrt(e*30/8))))
            radius_fracs.append(0.5)
        elif e >= 2000 and e <= 5000:
            sizes.append(int(np.ceil(np.cbrt(e*40/8))))
            radius_fracs.append(0.8)
        elif e > 5000 and e <= 20e3:
            sizes.append(int(np.ceil(np.cbrt(e*65/8))))
            radius_fracs.append(0.95)
    # thicknesses = [alat] * len(energies)
    thicknesses = [alat/2] * len(energies)

    cas_calc = CascadeCalculator(meam, mass, element, lattice, alat, sizes, thicknesses, radius_fracs, temperature,
                                energies, num_directions)
    cas_calc.calculate(relax_flag=False, simulation_flag=False, check_flag=False, postprocess_flag=True)  # check temp, check whether it stops ok -> tgap


    
    
    


        
