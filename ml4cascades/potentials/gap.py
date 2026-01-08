import os, time
from ml4cascades.potentials import IPotential
from ml4cascades.utils import BasicCellInfo
from ml4cascades.turbogap import CascadeCalculator, CascadeProcessor, CascadeChecker
from ml4cascades.utils import CascadePloter

AMU_TO_KG = 1.66053906660E-27 # Atomic mass unit to kg conversion factor
JOULE_TO_EV = 6.241509074E18  # Joule to eV conversion factor
ANGSTROM_TO_METER = 1E-10     # Angstrom to meter conversion factor
PS_TO_S = 1E-12               # Picosecond to second conversion factor

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
    bi = BasicCellInfo(element=['Ge'], atomic_num=[32], mass=72.64, lattice='diamond', alat=[5.76]*3)
    supercell_size = [16]*3
    calc = CascadeCalculator(tgap, bi)

    '''
    Parameter testing  
    '''
    # ploter = CascadePloter()
    # ploter.plot_eph_results(datafile=os.path.join(calc.calculation_dir, 'cascade', 'Test-Ce-kappae', '3', 'eph-EnergySharingData.txt'),
    #                         figfile=os.path.join(calc.calculation_dir, 'cascade', 'Test-Ce-kappae', '3', 'eph-EnergySharingData.png'))

    # ploter.plot_thermo_results(datafile=os.path.join(calc.calculation_dir, 'cascade', 'Test-Ce-kappae', '3', 'thermo.log'),
    #                             figfile=os.path.join(calc.calculation_dir, 'cascade', 'Test-Ce-kappae', '3', 'thermolog.png'))
    
    # ploter.plot_mesh_Te(ni=6, nj=6, datafile=os.path.join(calc.calculation_dir, 'cascade', 'Test-Ce-kappae', '3', 'eph-ToutData.txt'), 
    #                     figfile=os.path.join(calc.calculation_dir, 'cascade', 'Test-Ce-kappae', '3', 'eph-TeMeshData.png'))

    '''
    Cascade simulation
    '''
    input_config = {
        "supercell_size": supercell_size,
        "equ_md_steps": 5000,
        "temp": 300,
        "taut": 100
    }
    # calc.thermalize_atomic(input_config)

    # 300 K Ce, kappa_e
    # from https://github.com/N-Medvedev/XTANT-3_coupling_data/blob/main/K_semiconductors/K_Ge.dat
    # Ce = 1.7674049595143981E+02 # J/(m^3*K)
    # Ce = Ce * JOULE_TO_EV / ((1/ANGSTROM_TO_METER)**3)  # eV/K/Ang^3
    # kappa_e = 4.0526892294595090E-01 # W/(K*m) or J/(K*m*s)
    # kappa_e = kappa_e * JOULE_TO_EV / (1/ANGSTROM_TO_METER * (1/PS_TO_S))  # eV/(K*Ang*ps)

    # from Nuclear Instruments and Methods in Physics Research B 485 (2020) 1–9
    rho = 1.0 # electrons/volume unit
    # from PHYSICAL REVIEW B 104, 195203 (2021)
    # option 1: low Ce, low kappa_e
    # Ce = 5e-6
    # kappa_e = 5e-3
    # option 2: high Ce, high kappa_e
    Ce = 1.29e-4
    kappa_e = 1.29e-1
    # option 3: high Ce, low kappa_e
    # Ce = 1.29e-4
    # kappa_e = 5e-3
  
    xhi = bi.alat[0] * supercell_size[0] * 2
    yhi = bi.alat[1] * supercell_size[1] * 2
    zhi = bi.alat[2] * supercell_size[2] * 2
    input_config = {
        "supercell_size": supercell_size,
        "equ_md_steps": 5000,
        "temp": 300,
        "xlow": 0,
        "xhigh": xhi,
        "ylow": 0,
        "yhigh": yhi,
        "zlow": 0,
        "zhigh": zhi,
        "eph_C_e": Ce,
        "eph_kappa_e": kappa_e,
        "eph_tout_file": 'eph-ToutData.txt'
    }
    # calc.thermalize_electronic(input_config)

    num_PKA_directions = 20
    radius_frac = 0.7
    PKA_kin_eng = 1000 # in eV
    input_config = {
        "supercell_size": supercell_size,
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
        "eph_tout_file": 'eph-ToutData.txt'
    }
    # calc.run_cascade(num_PKA_directions=num_PKA_directions,
    #                 radius_frac=radius_frac,
    #                 PKA_kin_eng=PKA_kin_eng,
    #                 input_config=input_config)
    
    '''
    Supercell size checking
    '''
    traj_folder = os.path.join(calc.calculation_dir, 'cascade', 'PKA_1000eV-0.8')
    checker = CascadeChecker(PKA_kin_eng=1000, 
                             supercell_size=supercell_size, 
                             radius_frac=0.8, 
                             traj_folder=traj_folder)
    checker.check_structure(start_traj=1, num_trajs=20, border_thickness=5.76, pot_eng_threshold=-4, kin_eng_threshold=5)
                 
    '''
    Cascade data processing
    '''
    # traj_folder = os.path.join(calc.calculation_dir, 'cascade', 'PKA_1000eV')
    # processor = CascadeProcessor(bi, PKA_kin_eng=1000, traj_folder=traj_folder)
    # processor.cal_ibm(num_trajs=20, n0=1, ed=1)
    # processor.cal_WSDefect(num_trajs=20)
    # processor.cal_cluster(start_traj=3, num_trajs=2, expression='Occupancy!=1')
