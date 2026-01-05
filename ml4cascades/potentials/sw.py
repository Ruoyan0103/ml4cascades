import os, time
from ml4cascades.potentials import IPotential
from ml4cascades.utils import BasicCellInfo
from ml4cascades.lammps import CascadeCalculator
from ml4cascades.utils import CascadePloter

AMU_TO_KG = 1.66053906660E-27 # Atomic mass unit to kg conversion factor
JOULE_TO_EV = 6.241509074E18  # Joule to eV conversion factor
ANGSTROM_TO_METER = 1E-10     # Angstroms/picosecond to meters/second conversion factor
PS_TO_S = 1E-12               # Picoseconds to seconds conversion factor

module_dir = os.path.dirname(__file__)

class SWPotential(IPotential):
    def __init__(self, pair_style: str, pair_coeff1: str, pair_coeff2: str):
        self.name = 'sw'
        self.ff_settings = ''.join([
             f'pair_style {pair_style}\n'
             f'pair_coeff {pair_coeff1}\n'
             f'pair_coeff {pair_coeff2}\n'
        ])

    def get_pot_files_path(self):
        return os.path.join(module_dir, 'params', 'SW')
    
if __name__ == "__main__":
    pair_style = 'hybrid/overlay table linear 100000000 sw'
    pair_coeff1 = '* * table nlh-SW.table NLH_GE'
    pair_coeff2 = '* * sw Ge_3body.sw Ge'
    sw = SWPotential(pair_style, pair_coeff1, pair_coeff2)
    bi = BasicCellInfo(element=['Ge'], atomic_num=[32], mass=72.64, lattice='diamond', alat=[5.76]*3)
    supercell_size = [16]*3
    calc = CascadeCalculator(sw, bi)

    xhi = bi.alat[0] * supercell_size[0] * 2
    yhi = bi.alat[1] * supercell_size[1] * 2
    zhi = bi.alat[2] * supercell_size[2] * 2
    Ce = 1.29e-4
    kappa_e = 1.29e-1
    input_config = {
        "supercell_size": supercell_size,
        "equ_md_steps": 10000,
        "temp": 300,
        "xlow": 0,
        "xhigh": xhi,
        "ylow": 0,
        "yhigh": yhi,
        "zlow": 0,
        "zhigh": zhi,
        "gsx": int(xhi // 21),
        "gsy": int(yhi // 21),
        "gsz": int(zhi // 21),
        "eph_C_e": Ce,
        "eph_kappa_e": kappa_e,
        "tinfile": 'NULL'
    }
    # calc.thermalize(input_config)

    num_PKA_directions = 2
    radius_frac = 0.8
    PKA_kin_eng = 1000 # in eV
    input_config = {
        "supercell_size": supercell_size,
        "border_thickness": 5.76,
        "cascade_steps": 40000,
        "temp": 300,
        "xlow": 0,
        "xhigh": xhi,
        "ylow": 0,
        "yhigh": yhi,
        "zlow": 0,
        "zhigh": zhi,
        "gsx": int(xhi // 21),
        "gsy": int(yhi // 21),
        "gsz": int(zhi // 21),
        "eph_C_e": Ce,
        "eph_kappa_e": kappa_e,
        "tinfile": 'NULL'
    }
    calc.run_cascade(num_PKA_directions=num_PKA_directions,
                    radius_frac=radius_frac,
                    PKA_kin_eng=PKA_kin_eng,
                    input_config=input_config)
    
    
