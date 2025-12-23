import os, time
from ml4cascades.potentials import IPotential
from ml4cascades.utils import BasicCellInfo
from ml4cascades.turbogap import CascadeCalculator

AMU_TO_KG = 1.66053906660E-27 # Atomic mass unit to kg conversion factor
JOULE_TO_EV = 6.241509074E18  # Joule to eV conversion factor
ANGSTROM_TO_METER = 1E-10     # Angstroms/picosecond to meters/second conversion factor
PS_TO_S = 1E-12               # Picoseconds to seconds conversion factor

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
    bi = BasicCellInfo(element=['Ge'], mass=72.64, lattice='diamond', alat=[5.76]*3)
    supercell_size = [10]*3
    calc = CascadeCalculator(tgap, bi)
    # calc.thermalize_atomic(supercell_size=supercell_size, equ_md_steps=5000, temp=300, taut=100)

    # 300 K Ce, kappa_e
    # from https://github.com/N-Medvedev/XTANT-3_coupling_data/blob/main/K_semiconductors/K_Ge.dat
    Ce = 1.7674049595143981E+02 # J/(m^3*K)
    kappa_e = 4.0526892294595090E-01 # W/(K*m) or J/(K*m*s)
    Ce = Ce * JOULE_TO_EV / ((1/ANGSTROM_TO_METER)**3)  # eV/K/Ang^3
    kappa_e = kappa_e * JOULE_TO_EV / (1/ANGSTROM_TO_METER * (1/PS_TO_S))  # eV/(K*Ang*ps)
    # print('Converted Ce:', Ce)
    # print('Converted kappa_e:', kappa_e)
    xhi = bi.alat[0] * supercell_size[0] * 2
    yhi = bi.alat[1] * supercell_size[1] * 2
    zhi = bi.alat[2] * supercell_size[2] * 2
    calc.thermalize_electronic(equ_md_steps=5000, temp=300,
                               xlow=0, xhigh=xhi, ylow=0, yhigh=yhi, zlow=0, zhigh=zhi, 
                               eph_C_e=Ce, eph_kappa_e=kappa_e, eph_tout_file='eph_tout.dat')

