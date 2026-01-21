import os, time, sys, argparse
from ml4cascades.potentials import IPotential
from ml4cascades.utils import BasicCellInfo
from ml4cascades.lammps import CascadeCalculator, CascadeProcessor, CascadeChecker
from ml4cascades.utils import LammpsCascadePlotter, ParameterGetter

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
    parser = argparse.ArgumentParser(
        description="SW cascade workflow controller"
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
        # print("\npython -m ml4cascades.potentials.sw --choice 4 > cascade_choice4.log 2>&1 &\n")
        choice = str(args.choice)
    VALID_CHOICES = {str(i) for i in range(1, 8)}
    if choice not in VALID_CHOICES:
        print(f"Invalid choice '{choice}'. Valid options are 1–7.")
        sys.exit(1)

    pair_style = 'hybrid/overlay table linear 100000000 sw'
    pair_coeff1 = '* * table nlh-SW.table NLH_GE'
    pair_coeff2 = '* * sw Ge_3body.sw Ge'
    sw = SWPotential(pair_style, pair_coeff1, pair_coeff2)
    bi = BasicCellInfo(element=['Ge'], atomic_num=[32], mass=72.64, lattice='diamond', alat=[5.76]*3)
    supercell_size = [16]*3
    calc = CascadeCalculator(sw, bi)

    '''
    ######################################### 1. Get Ce and Ke ########################################## 
    '''
    if choice == '1':
        print("Getting Ce and Ke...")
        Ce, Ke = ParameterGetter().get_data1(temp=300)
        ParameterGetter().get_data2(temp=900)

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
        # for space in range(21, 22):
        #     grid_size.add((int(xhi // space)))
        grid_size.add(8)
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
    
    # Final chosen parameters
    # Ce = 5e-8           # eV/ps/A^3/K, value an order of magnitude higher than that at 300 K
    # kappa_e = 2.529e-4  # eV/ps/A/K, value at 300 K

    '''
    ######################################### 3. Thermalization #########################################
    '''
    Ce = 5e-8           # eV/ps/A^3/K
    kappa_e = 2.529e-4  # eV/ps/A/K
    if choice == '3':
        print("Thermalizing...")
        input_config = {
            "supercell_size": supercell_size,
            "equ_md_steps": 10000,
            "temp": 300,
            "xlow": -xhi/2+bi.alat[0]*supercell_size[0]/2,
            "xhigh": xhi/2+bi.alat[0]*supercell_size[0]/2,
            "ylow": -yhi/2+bi.alat[1]*supercell_size[1]/2,
            "yhigh": yhi/2+bi.alat[1]*supercell_size[1]/2,
            "zlow": -zhi/2+bi.alat[2]*supercell_size[2]/2,
            "zhigh": zhi/2+bi.alat[2]*supercell_size[2]/2,
            "gsx": int(xhi // 21),
            "gsy": int(yhi // 21),
            "gsz": int(zhi // 21),
            "eph_C_e": Ce,
            "eph_kappa_e": kappa_e,
            "tinfile": 'NULL'
        }
        # calc.thermalize(input_config)

    '''
    ######################################### 4. Cascade simulation #########################################
    '''
    num_PKA_directions = 22
    radius_frac = 0.8
    PKA_kin_eng = 1000 # in eV
    if choice == '4':
        print("Cascade simulation...")
        input_config = {
            "supercell_size": supercell_size,
            "border_thickness": 5.76/2,
            "cascade_steps": 40000,
            "temp": 300,
            "xlow": -xhi/2+bi.alat[0]*supercell_size[0]/2,
            "xhigh": xhi/2+bi.alat[0]*supercell_size[0]/2,
            "ylow": -yhi/2+bi.alat[1]*supercell_size[1]/2,
            "yhigh": yhi/2+bi.alat[1]*supercell_size[1]/2,
            "zlow": -zhi/2+bi.alat[2]*supercell_size[2]/2,
            "zhigh": zhi/2+bi.alat[2]*supercell_size[2]/2,
            "gsx": 8,
            "gsy": 8,
            "gsz": 8,
            "eph_C_e": Ce,
            "eph_kappa_e": kappa_e,
            # "tinfile": 'NULL'
            "temperature_dependent": True
        }
        PKA_kin_eng_dir = calc.run_cascade(num_PKA_directions=num_PKA_directions,
                                            radius_frac=radius_frac,
                                            PKA_kin_eng=PKA_kin_eng,
                                            input_config=input_config,
                                            running_dir='/scratch/phys/t30429_nume-dft-ml/04-Ruoyan/02-Paper2/02-cascade/ml4cascades/ml4cascades/lammps/results/cascade/EPH/cascade/test')
    
    '''
    ######################################### 5. Cascade checker #########################################
    '''
    if choice == '5':
        print("Cascade checking...")
        PKA_kin_eng_dir = os.path.join(calc.calculation_dir, 'cascade', f'PKA_1000eV-{radius_frac}')
        checker = CascadeChecker(PKA_kin_eng=PKA_kin_eng, 
                                 supercell_size=supercell_size, 
                                 radius_frac=radius_frac,
                                 traj_folder=PKA_kin_eng_dir,
                                 successful_folder=os.path.join(calc.calculation_dir, 'cascade', 'PKA_1000eV-suc'))
        checker.check_output(start_output=1, num_outputs=22, running_time=35) # in ps 

    '''
    ######################################### 6. Cascade data plotting ####################################
    '''
    if choice == '6':
        print(os.path.join(calc.calculation_dir, 'cascade', 'test', '1', 'eng.out'))
        print("Cascade data plotting...")
        plotter = LammpsCascadePlotter()
        fig_file1 = os.path.join(calc.calculation_dir, 'Test-CascadeProcess', 'Test-thermostat', 'berendsen', 'Test-Ce/5e-6/nothermostat', 'eph_results.png')
        plotter.plot_eph_results(datafile1=os.path.join(calc.calculation_dir, 'Test-CascadeProcess', 'Test-thermostat', 'berendsen', 'Test-Ce/5e-6/nothermostat', 'eng.out'), 
                                 datafile2=os.path.join(calc.calculation_dir, 'Test-CascadeProcess', 'Test-thermostat', 'berendsen', 'Test-Ce/5e-6/nothermostat', 'thermo.out'),
                                 figfile=fig_file1)
        
        dump_file = os.path.join(calc.calculation_dir, 'Test-CascadeProcess', 'Test-thermostat', 'berendsen', 'Test-Ce/5e-6/berendsen', 'data.output')
        T_out_folder = os.path.join(calc.calculation_dir, 'Test-CascadeProcess', 'Test-thermostat', 'berendsen', 'Test-Ce/5e-6/berendsen', 'T_out')
        new_dump_file = os.path.join(calc.calculation_dir, 'Test-CascadeProcess', 'Test-thermostat', 'berendsen', 'Test-Ce/5e-6/berendsen', 'data.output.modified') 
        new_tout_file = os.path.join(calc.calculation_dir, 'Test-CascadeProcess', 'Test-thermostat', 'berendsen', 'Test-Ce/5e-6/berendsen', 'T_out.modified') 
        avg_Ta_file = os.path.join(calc.calculation_dir, 'Test-CascadeProcess', 'Test-thermostat', 'berendsen', 'Test-Ce/5e-6/berendsen', 'avg_Ta.out')
        avg_Te_file = os.path.join(calc.calculation_dir, 'Test-CascadeProcess', 'Test-thermostat', 'berendsen', 'Test-Ce/5e-6/berendsen', 'avg_Te.out')  

        flag = 5 # 1: new_dump_file, 2: new_tout_file, 3: avg_Ta_file, 4: avg_Te_file, 5: all
        plotter.get_grid_Ta_Te(dump_file=dump_file, 
                               T_out_folder=T_out_folder, 
                               new_dump_file=new_dump_file, 
                               new_tout_file=new_tout_file, 
                               avg_Ta_file=avg_Ta_file, 
                               avg_Te_file=avg_Te_file,
                               flag=flag)
        
        frame_id_list = [3, 17, 25, 35, 45, 65, 105, 174]
        time_list = [0.01, 0.1, 0.2, 1, 2, 4, 8, 15]  # in ps
        fig_file2 = os.path.join(calc.calculation_dir, 'Test-CascadeProcess', 'Test-thermostat', 'berendsen', 'Test-Ce/5e-6/berendsen', 'extreme_Te.png')
        plotter.get_extreme_Ta_Te(new_tout_file, frame_id_list, time_list, fig_file2) 
        
        grid_list = [282, 283, 284, 285]
        electron_grid_list = [280, 281, 282, 283, 284, 285, 286, 287]
        # grid_list = [283, 284]
        # electron_grid_list = [280, 281, 282, 283, 284, 285, 286, 287]
        # grid_list = [2166, 2167, 2168, 2169]
        # electron_grid_list = [2163, 2164, 2165, 2166, 2167, 2168, 2169, 2170, 2171, 2172]
        fig_file3 = os.path.join(calc.calculation_dir, 'Test-CascadeProcess', 'Test-thermostat', 'berendsen', 'Test-Ce/5e-6/berendsen', 'Te_along_x.png')
        plotter.plot_te_ta_along_x(new_tout_file, new_dump_file, 
                                   [26, 66, 106], [0.2, 4, 8], 
                                   grid_list, electron_grid_list, 
                                   fig_file3)
    '''
    ######################################### 7. Cascade output processing ####################################
    '''
    if choice == '7':
        print("Cascade output processing...")
        PKA_kin_eng_dir = os.path.join(calc.calculation_dir, 'cascade', 'PKA_1000eV-suc')
        processor = CascadeProcessor(bi, PKA_kin_eng=1000, traj_folder=PKA_kin_eng_dir)
        # # processor.cal_ibm(num_trajs=1, n0=1, ed=1)    # not working for LAMMPS data format
        # processor.cal_WSDefect(start_traj=1, num_trajs=20)
        processor.cal_cluster(start_traj=1, num_trajs=20, expression='Occupancy!=1')



    
    
