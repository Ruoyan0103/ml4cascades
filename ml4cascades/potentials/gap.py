import os
from ml4cascades.potentials import IPotential
from ml4cascades.utils import BasicCellInfo
from ml4cascades.turbogap import CascadeCalculator


module_dir = os.path.dirname(__file__)

class TGAPotential(IPotential):
    def __init__(self, param_file: str):
        self.name = 'TGAP'
        self.ff_settings = param_file

    def get_pot_files_path(self):
        return os.path.join(module_dir, 'params', 'TGAP')

if __name__ == "__main__":
    gap_file = 'Ge-v10-gap'
    tgap = TGAPotential(gap_file)
    bi = BasicCellInfo(element='Ge', mass=72.64, lattice='diamond', alat=[5.76]*3)
    calc = CascadeCalculator(tgap, bi)
    calc.thermalize_atomic(supercell_size=[2]*3, equ_md_steps=5000, temp=300, taut=100)
