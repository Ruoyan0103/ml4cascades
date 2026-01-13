import os, time, sys, argparse
import numpy as np
from ml4cascades.potentials import IPotential
from ml4cascades.utils import BasicCellInfo
from ml4cascades.turbogap import CascadeCalculator, CascadeProcessor, CascadeChecker
from ml4cascades.utils import TurbogapCascadePlotter, ParameterGetter

module_dir = os.path.dirname(__file__)

class TGAPotential(IPotential):
    def __init__(self, param_file: str):
        self.name = 'TGAP'
        self.ff_settings = param_file

    def get_pot_files_path(self):
        return os.path.join(module_dir, 'params', 'TGAP')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GAP cascade with TurboGAP workflow controller"
    )
    parser.add_argument(
        "--choice",
        type=int,
        help="Select workflow step (1–7)"
    )
    args = parser.parse_args()
    if args.choice is None:
        print("\n################### Enter option: #######################\n")
        print("1: Get Ce and Ke")
        print("2: Test parameters")
        print("3: Thermalization")
        print("4: Cascade simulation")
        print("5: Cascade checker")
        print("6: Cascade data plotting")
        print("7: Cascade output processing")
        print("\n##########################################################\n")
        choice = input("Enter number: ").strip()
    else:
        # print("\n################### Background running example: #######################\n")
        # print("\npython -m ml4cascades.potentials.gap --choice 4 > cascade_choice4.log 2>&1 &\n")
        choice = str(args.choice)
    VALID_CHOICES = {str(i) for i in range(1, 8)}
    if choice not in VALID_CHOICES:
        print(f"Invalid choice '{choice}'. Valid options are 1–7.")
        sys.exit(1)

    gap_file = 'Ge-v10-gap'
    tgap = TGAPotential(gap_file)
    bi = BasicCellInfo(element=['Ge'], atomic_num=[32], mass=72.64, lattice='diamond', alat=[5.76]*3)
    supercell_size = [16]*3
    calc = CascadeCalculator(tgap, bi)

    '''
    ######################################### 1. Get Ce and Ke ########################################## 
    '''
    if choice == '1':
        print("Getting Ce and Ke...")
        Ce, Ke = ParameterGetter(temp=300).get_data1()
        ParameterGetter(temp=900).get_data2()

    '''
    ######################################### 2. Test parameters ######################################### 
    '''
    xhi = bi.alat[0] * supercell_size[0] * 2
    yhi = bi.alat[1] * supercell_size[1] * 2
    zhi = bi.alat[2] * supercell_size[2] * 2
    if choice == '2':
        print("Testing parameters...")
        # -------------------- for grid size -------------------- 
        grid_size = set()
        for space in range(21, 22):
            grid_size.add((int(xhi // space)))
        # -------------------- for timestep -------------------- 
        # Ce_list = [1.1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6]
        # kappa_e_list = [2.52e-4, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
        # -------------------- for constant Ce -------------------- 
        Ce_list = [5e-8] * 9
        kappa_e_list = [2.52e-4, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 5e-2]
        # -------------------- for constant ke -------------------- 
        # Ce_list = [1.1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6]
        # kappa_e_list = [5e-3] * 8
        for Ce, kappa_e in zip(Ce_list, kappa_e_list):
            for grid in grid_size:
                running_dir = os.path.join(calc.calculation_dir, 'Test-CascadeProcess', 'Test-Ke', f'Ke_{kappa_e}', f'{grid}-{grid}-{grid}')
                num_PKA_directions = 1
                radius_frac = 0.7
                PKA_kin_eng = 1000 # in eV
                input_config = {
                    "supercell_size": supercell_size,
                    "border_thickness": 5.76/2,
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
                calc.run_cascade(running_dir=running_dir,
                                num_PKA_directions=num_PKA_directions,
                                radius_frac=radius_frac,
                                PKA_kin_eng=PKA_kin_eng,
                                input_config=input_config)
    
    # Final chosen parameters
    # Ce = 5e-8           # eV/ps/A^3/K, value an order of magnitude higher than that at 300 K
    # kappa_e = 2.529e-4  # eV/ps/A/K, value at 300 K

    '''
    ######################################### 3. Thermalization #########################################
    '''
    rho = 1.0
    Ce = 5e-8           # eV/ps/A^3/K
    kappa_e = 2.529e-4  # eV/ps/A/K
    if choice == '3':
        print("Thermalizing...")
        input_config = {
            "supercell_size": supercell_size,
            "equ_md_steps": 5000,
            "temp": 300,
            "taut": 100
        }
        calc.thermalize_atomic(input_config)

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
        calc.thermalize_electronic(input_config)

    '''
    ######################################### 4. Cascade simulation #########################################
    '''
    border_thickness=5.76/2
    if choice == '4':
        print("Running cascade simulation...")
        print("Use short short time for checking the supercell size...")
        num_PKA_directions = 22
        radius_frac = 0.7
        PKA_kin_eng = 1000 # in eV
        input_config = {
            "supercell_size": supercell_size,
            "border_thickness": border_thickness,
            "cascade_steps": 5000,
            "temp": 300,
            "xlow": 0,
            "xhigh": xhi,
            "ylow": 0,
            "yhigh": yhi,
            "zlow": 0,
            "zhigh": zhi,
            "gsx": int(xhi // 25),
            "gsy": int(yhi // 25),
            "gsz": int(zhi // 25),
            "eph_C_e": Ce,
            "eph_kappa_e": kappa_e,
            "eph_tout_file": 'eph-ToutData.txt'
        }
        # running_case_list = np.arange(1, 23)
        running_case_list = np.arange(13, 23)
        calc.run_cascade(num_PKA_directions=num_PKA_directions,
                        radius_frac=radius_frac,
                        PKA_kin_eng=PKA_kin_eng,
                        running_case_list=running_case_list,
                        input_config=input_config)
    
    '''
    ######################################### 5. Cascade checker #########################################
    '''
    if choice == '5':
        print("Checking cascade results...")
        traj_folder = os.path.join(calc.calculation_dir, 'cascade', 'PKA_1000eV-0.7')
        checker = CascadeChecker(PKA_kin_eng=1000, 
                                supercell_size=supercell_size, 
                                radius_frac=0.7, 
                                traj_folder=traj_folder)
        checker.check_structure(start_traj=1, num_trajs=6, border_thickness=border_thickness, pot_eng_threshold=-4, kin_eng_threshold=5)

    '''
    ######################################### 6. Cascade data plotting ####################################
    '''
    if choice == '6':
        print("Plotting cascade data...")
        ploter = TurbogapCascadePlotter()
        ploter.plot_eph_results(datafile=os.path.join(calc.calculation_dir, 'cascade', 'PKA_1000eV-0.7', '1', 'eph-EnergySharingData.txt'),
                                figfile=os.path.join(calc.calculation_dir, 'cascade', 'PKA_1000eV-0.7', '1', 'eph-EnergySharingData.png'))

    '''
    ######################################### 7. Cascade output processing ####################################
    '''
    if choice == '7':
        print("Processing cascade outputs...")
        traj_folder = os.path.join(calc.calculation_dir, 'cascade', 'PKA_1000eV')
        processor = CascadeProcessor(bi, PKA_kin_eng=1000, traj_folder=traj_folder)
        processor.cal_ibm(num_trajs=20, n0=1, ed=1)
        processor.cal_WSDefect(start_traj=1, num_trajs=3)
        processor.cal_cluster(start_traj=1, num_trajs=3, expression='Occupancy!=1')
