import os
import xml.etree.ElementTree as ET
from ml4cascades.potentials import IPotential
from ml4cascades.lammps.calcs import CascadeCalculator
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
    
    mass, element, lattice, alat, temperature = 72.56, 'Ge', 'diamond', 5.76, 300    
    energies, num_directions = [100, 400, 1000, 2000, 5000, 10e3], 59
    sizes = [9, 13, 17, 22, 30, 37]  # 47 - 20e3 63 - 50e3
    pka_ids = [2918, 8674, 17185, 28932, 104738, 191392]

    cas_calc = CascadeCalculator(meam, mass, element, lattice, alat, sizes, temperature,
                                 pka_ids, energies, num_directions)
    cas_calc.calculate()
    
    
    


        