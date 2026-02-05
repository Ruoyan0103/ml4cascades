import os, time, sys, argparse

from matplotlib.pylab import f
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
    # supercell_size = [16]*3
    supercell_size = [18]*3
    model_name  = 'STOPPING'
    calc = CascadeCalculator(sw, bi, model_name=model_name)

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
        for grid in [1, 2, 4, 8, 16, 32, 64]:
            grid_size.add(grid)
        Ce_list = [5e-6]
        kappa_e_list = [2.52e-4]
        # -------------------- for timestep -------------------- 
        # Ce_list = [1.1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6]
        # kappa_e_list = [2.52e-4, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
        # -------------------- for constant Ce -------------------- 
        # Ce_list = [5e-6] * 9
        # kappa_e_list = [2.52e-4, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 5e-2]
        # -------------------- for constant ke -------------------- 
        # Ce_list = [1.1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6]
        # kappa_e_list = [5e-3] * 8
        for Ce, kappa_e in zip(Ce_list, kappa_e_list):
            for grid in grid_size:
                running_dir = os.path.join(calc.calculation_dir, 'Test-CascadeProcess', 'Test-num_grid_points', f'{grid}-{grid}-{grid}')
                num_PKA_directions = 1
                radius_frac = 0.7
                PKA_kin_eng = 1000 # in eV
                input_config = {
                    "supercell_size": supercell_size,
                    "border_thickness": 5.76,
                    "cascade_steps": 40000,
                    "temp": 300,
                    "xlow": -xhi/2+bi.alat[0]*supercell_size[0]/2,
                    "xhigh": xhi/2+bi.alat[0]*supercell_size[0]/2,
                    "ylow": -yhi/2+bi.alat[1]*supercell_size[1]/2,
                    "yhigh": yhi/2+bi.alat[1]*supercell_size[1]/2,
                    "zlow": -zhi/2+bi.alat[2]*supercell_size[2]/2,
                    "zhigh": zhi/2+bi.alat[2]*supercell_size[2]/2,
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
    Ce = 5e-6          # eV/ps/A^3/K
    kappa_e = 2.52e-4  # eV/ps/A/K
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
            "gsx": 8,
            "gsy": 8,
            "gsz": 8,
            "eph_C_e": Ce,
            "eph_kappa_e": kappa_e,
            "tinfile": 'NULL'
        }
        calc.thermalize(input_config)

    '''
    ######################################### 4. Cascade simulation #########################################
    '''
    num_PKA_directions = 30
    radius_frac = 0.7
    PKA_kin_eng = 1000 # in eV
    grid_value = 1
    if choice == '4':
        print("Cascade simulation...")
        input_config = {
            "supercell_size": supercell_size,
            "border_thickness": 5.76/2,
            "cascade_steps": 80000, 
            "temp": 300,
            "xlow": -xhi/2+bi.alat[0]*supercell_size[0]/2,
            "xhigh": xhi/2+bi.alat[0]*supercell_size[0]/2,
            "ylow": -yhi/2+bi.alat[1]*supercell_size[1]/2,
            "yhigh": yhi/2+bi.alat[1]*supercell_size[1]/2,
            "zlow": -zhi/2+bi.alat[2]*supercell_size[2]/2,
            "zhigh": zhi/2+bi.alat[2]*supercell_size[2]/2,
            "gsx": grid_value,
            "gsy": grid_value,
            "gsz": grid_value,
            "eph_C_e": Ce,
            "eph_kappa_e": kappa_e,
            "cutoff_eng": 10,
            # "tinfile": 'NULL',
            # "temperature_dependent": False
        }
        PKA_kin_eng_dir = calc.run_cascade(num_PKA_directions=num_PKA_directions,
                                           radius_frac=radius_frac,
                                           PKA_kin_eng=PKA_kin_eng,
                                           input_config=input_config)
    
    '''
    ######################################### 5. Cascade checker #########################################
    '''
    if choice == '5':
        print("Cascade checking...")
        grid_value = 16
        PKA_kin_eng_dir = os.path.join(calc.calculation_dir, 'cascade', f'PKA_1000eV-{radius_frac}-10')
        checker = CascadeChecker(PKA_kin_eng=PKA_kin_eng, 
                                 supercell_size=supercell_size, 
                                 radius_frac=radius_frac,
                                 traj_folder=PKA_kin_eng_dir,
                                 grid=grid_value)
        checker.check_output(start_output=1, num_outputs=30, running_time=65) # in ps 

    '''
    ######################################### 6. Cascade data plotting ####################################
    '''
    if choice == '6':
        if model_name == 'STOPPING':
            print("Cascade data plotting...")
            plotter = LammpsCascadePlotter()
            print("Plotting stopping results...")
            folder = '/scratch/phys/t30429_nume-dft-ml/04-Ruoyan/02-Paper2/02-cascade/ml4cascades/ml4cascades/lammps/results/cascade/STOPPING/Test-CascadeProcess/Test-border-10/berendsen'
            fig_file = f'{folder}/stopping_results.png'
            plotter.plot_stopping_results(datafile=f'{folder}/thermo.out', figfile=fig_file, ecut=10)
    
        elif model_name == 'EPH':
            print("Cascade data plotting...")
            plotter = LammpsCascadePlotter()
            print("Plotting eph results...")
            folder = '/scratch/phys/t30429_nume-dft-ml/04-Ruoyan/02-Paper2/02-cascade/ml4cascades/ml4cascades/lammps/results/cascade/EPH/Test-CascadeProcess/Test-num_grid_points/32-32-32/1'
            fig_file1 = f'{folder}/eph_results.png'
            plotter.plot_eph_results(datafile1=f'{folder}/eng.out', 
                                    datafile2=f'{folder}/thermo.out', figfile=fig_file1)

            dump_file = f'{folder}/data.output'
            T_out_folder = f'{folder}/T_out'
            new_dump_file = f'{folder}/data.output.modified'
            new_tout_file = f'{folder}/T_out.modified'
            avg_Ta_file = f'{folder}/avg_Ta.out'
            avg_Te_file = f'{folder}/avg_Te.out'
            flag = 5 # 1: new_dump_file, 2: new_tout
            # plotter.get_grid_Ta_Te(dump_file=dump_file, 
            #                        T_out_folder=T_out_folder, 
            #                        new_dump_file=new_dump_file, 
            #                        new_tout_file=new_tout_file, 
            #                        avg_Ta_file=avg_Ta_file, 
            #                        avg_Te_file=avg_Te_file,
            #                        flag=flag)
            
            print("Plotting extreme Ta and Te...")
            frame_id_list = [9, 25, 35, 45, 65, 105, 174]
            time_list = [0.04, 0.2, 1, 2, 4, 8, 15]  # in ps
            fig_file2 = f'{folder}/extreme_Te_Ta.png'
            # plotter.get_extreme_Te_Ta(new_tout_file, new_dump_file, frame_id_list, time_list, fig_file2) 
            
            #---------------------- for grid size 8, electronic system twice size of atomic system ----------------------#
            # grid_list = [282, 283, 284, 285]
            # electron_grid_list = [280, 281, 282, 283, 284, 285, 286, 287]
            #---------------------- for grid size 4, electronic system twice size of atomic system ----------------------#
            # grid_list = [21, 22]
            # electron_grid_list = [20, 21, 22, 23]
            #---------------------- for grid size 8, electronic system forth size of atomic system ----------------------#
            # grid_list = [283, 284]
            # electron_grid_list = [278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289]
            #---------------------- for grid size 16, electronic system forth size as atomic system ----------------------#
            # grid_list = [2166, 2167, 2168, 2169]
            # electron_grid_list = [2163, 2164, 2165, 2166, 2167, 2168, 2169, 2170, 2171, 2172]
            #---------------------- for grid size 16, electronic system double size as atomic system ----------------------#
            # grid_list = [1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915]
            # electron_grid_list = [1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919]
            #---------------------- for grid size 16, electronic system double size as atomic system ----------------------#
            # grid_list = [1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915]
            # electron_grid_list = [1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919]
            # ---------------------- for grid size 32, electronic system double size as atomic system ----------------------#
            grid_list = range(16872, 16888)
            electron_grid_list = range(16864, 16896)
            #---------------------- for grid size 64, electronic system double size as atomic system ----------------------#
            # grid_list = range(128976, 129008)
            # electron_grid_list = range(124864, 124928)
            # for grid size 8, electronic system same size as atomic system
            # grid_list = [280, 281, 282, 283, 284, 285, 286, 287]
            # electron_grid_list = [280, 281, 282, 283, 284, 285, 286, 287]
            print("Plotting Te and Ta along x...")
            fig_file3 = f'{folder}/Te_Ta_along_x.png'
            plotter.plot_te_ta_along_x(new_tout_file, new_dump_file, 
                                    [9, 26, 66, 106], [0.04, 0.2, 4, 8], 
                                    grid_list, electron_grid_list, 
                                    fig_file3)
     
    '''
    ######################################### 7. Cascade output processing ####################################
    '''
    if choice == '7':
        print("Cascade output processing...")
        PKA_kin_eng_dir = os.path.join(calc.calculation_dir, 'cascade', 'PKA_1000eV-0.7-10')
        processor = CascadeProcessor(bi, PKA_kin_eng=1000, traj_folder=PKA_kin_eng_dir)
        # # processor.cal_ibm(num_trajs=1, n0=1, ed=1)    # not working for LAMMPS data format
        processor.cal_WSDefect(start_traj=1, num_trajs=30, exclude_list=[])  
        # processor.cal_cluster(start_traj=1, num_trajs=30, expression='Occupancy!=1')
   
    
  

