import os, time
from ml4cascades.potentials import IPotential
from ml4cascades.utils import BasicCellInfo
from ml4cascades.lammps import CascadeCalculator, CascadeProcessor
from ml4cascades.utils import LammpsCascadePloter, ParameterGetter

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

    '''
    ######################################### Cascade simulation #########################################
    '''
    xhi = bi.alat[0] * supercell_size[0] * 2
    yhi = bi.alat[1] * supercell_size[1] * 2
    zhi = bi.alat[2] * supercell_size[2] * 2
    Ce = 1e-8  # eV/ps/A^3/K
    kappa_e = 1e-5  # eV/ps/A/K

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

    num_PKA_directions = 1
    radius_frac = 0.7
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
        # "tinfile": 'NULL'
    }
    # calc.run_cascade(num_PKA_directions=num_PKA_directions,
    #                 radius_frac=radius_frac,
    #                 PKA_kin_eng=PKA_kin_eng,
    #                 input_config=input_config)

    '''
    ######################################### Get Ce and Ke ########################################## 
    '''
    ParameterGetter(temp=900).get_data1()
    ParameterGetter(temp=900).get_data2()

    '''
    ######################################### Test parameters ######################################### 
    '''
    grid_size = set()
    for space in range(21, 22):
        grid_size.add((int(xhi // space)))
    # -------------------- for timestep -------------------- 
    # Ce_list = [1.1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6]
    # kappa_e_list = [2.52e-4, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    # -------------------- for constant Ce -------------------- 
    Ce_list = [5e-8, 5e-8, 5e-8, 5e-8, 5e-8, 5e-8, 5e-8, 5e-8, 5e-8]
    kappa_e_list = [2.52e-4, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 5e-2]
    Ce_list = [5e-8]
    kappa_e_list = [5e-2]
    # -------------------- for constant ke -------------------- 
    # Ce_list = [1.1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6]
    # kappa_e_list = [5e-3] * 8
    for Ce, kappa_e in zip(Ce_list, kappa_e_list):
        for grid in grid_size:
            running_dir = os.path.join(calc.calculation_dir, 'Test-grid-size', 'Electronic_step_1-constant_Ce', f'Ke_{kappa_e}', f'{grid}-{grid}-{grid}')
            num_PKA_directions = 1
            radius_frac = 0.7
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
                "gsx": grid,
                "gsy": grid,
                "gsz": grid,
                "eph_C_e": Ce,
                "eph_kappa_e": kappa_e,
                # "tinfile": 'NULL'
            }
            # calc.run_cascade(running_dir=running_dir,
            #                 num_PKA_directions=num_PKA_directions,
            #                 radius_frac=radius_frac,
            #                 PKA_kin_eng=PKA_kin_eng,
            #                 input_config=input_config)

    '''
    ######################################### Cascade data processing ##########################################
    '''
    # traj_folder = os.path.join(calc.calculation_dir, 'cascade', 'PKA_1000eV')
    # processor = CascadeProcessor(bi, 1000, traj_folder)
    # # processor.cal_ibm(num_trajs=1, n0=1, ed=1)    # not working for LAMMPS data format
    # processor.cal_WSDefect(start_traj=1, num_trajs=20)
    # processor.cal_cluster(start_traj=1, num_trajs=20, expression='Occupancy!=1')
    # ploter = LammpsCascadePloter()
    # ploter.plot_mesh_Te(ni=8, nj=8, datafile=os.path.join(calc.calculation_dir, 'Test-grid-size', 'Electronic_step_1-constant_Ce', 'Ke_5e-06', '8-8-8', '1', 'T_out_000006'), 
    #                      figfile=os.path.join(calc.calculation_dir, 'Test-grid-size', 'Electronic_step_1-constant_Ce', 'Ke_5e-06', '8-8-8', '1', 'T_out_000006.png'))
    

    
    
