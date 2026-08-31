import os, sys, argparse
from pathlib import Path
from matplotlib.pylab import f
from ml4cascades.potentials import IPotential
from ml4cascades.utils import BasicCellInfo
from ml4cascades.lammps import CascadeCalculator, CascadeChecker, CascadeProcessor, EPHprocessor
from ml4cascades.utils import LammpsCascadePlotter, ParameterGetter, LammpsCascadeVisualizer

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
        help="Select workflow step (1–8)"
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
        print("8: EPH processing")
        print("\n##########################################################\n")
        choice = input("Enter number: ").strip()
    else:
        # print("\n################### Background running example: #######################\n")
        # print("\npython -m ml4cascades.potentials.sw --choice 4 > cascade_choice4.log 2>&1 &\n")
        choice = str(args.choice)
    VALID_CHOICES = {str(i) for i in range(1, 9)}
    if choice not in VALID_CHOICES:
        print(f"Invalid choice '{choice}'. Valid options are 1–8.")
        sys.exit(1)

    pair_style = 'hybrid/overlay table linear 100000000 sw'
    pair_coeff1 = '* * table nlh-SW.table NLH_GE'
    pair_coeff2 = '* * sw Ge_3body.sw Ge'
    sw = SWPotential(pair_style, pair_coeff1, pair_coeff2)
    bi = BasicCellInfo(element=['Ge'], atomic_num=[32], mass=72.64, lattice='diamond', alat=[5.76]*3)
    supercell_size = [100]*3
    model_name  = 'EPH' # 'EPH' or 'STOPPING' or 'STOPPING-0K' or 'STOPPING-100K'
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
    if choice == '4':
        atomsfile = os.path.join(calc.calculation_dir, 'thermalize', f'{supercell_size[0]}-{supercell_size[1]}-{supercell_size[2]}', 'data.output')
        with open(atomsfile, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'xlo xhi' in line:
                    xlo = float(line.split()[0])
                    xhi = float(line.split()[1])
                elif 'ylo yhi' in line:
                    ylo = float(line.split()[0])
                    yhi = float(line.split()[1])
                elif 'zlo zhi' in line:
                    zlo = float(line.split()[0])
                    zhi = float(line.split()[1])
        num_PKA_directions = 50
        # running_directions = [x for x in range(1, num_PKA_directions+1) if x not in [16, 22, 25, 32, 34]] # for PKA_kin_eng=5000 eV
        running_directions = range(26, 31) 
        Ce = 5e-7
        radius_frac = 0.7
        PKA_kin_eng = 50000 # in eV
        grid_value = 44
        print("Cascade simulation...")
        input_config = {
            "supercell_size": supercell_size,
            "border_thickness": 5.65,
            "cascade_steps": 80000, 
            "temp": 300,
            "xlow": xlo-(xhi-xlo)/2,
            "xhigh": xhi+(xhi-xlo)/2,
            "ylow": ylo-(yhi-ylo)/2,
            "yhigh": yhi+(yhi-ylo)/2,
            "zlow": zlo-(zhi-zlo)/2,
            "zhigh": zhi+(zhi-zlo)/2,
            "gsx": grid_value,
            "gsy": grid_value,
            "gsz": grid_value, 
            "eph_C_e": Ce,
            "eph_kappa_e": kappa_e,
            "cutoff_eng": 10, 
            # "tinfile": 'NULL',
            "temperature_dependent": True
        }
        PKA_kin_eng_dir = calc.run_cascade(num_PKA_directions=num_PKA_directions,
                                           running_directions=running_directions,
                                           radius_frac=radius_frac,
                                           PKA_kin_eng=PKA_kin_eng,
                                           input_config=input_config)
    
    '''
    ######################################### 5. Cascade checker #########################################
    '''
    if choice == '5':
        print("Cascade checking...")
        PKA_kin_eng = 20000
        grid_value = 16
        cutoff = 10
        radius_frac = 0.7
        if model_name == 'EPH':
            PKA_kin_eng_dir = os.path.join(calc.calculation_dir, 'cascade', f'PKA_{PKA_kin_eng}eV', f'{radius_frac}-{grid_value}')
            checker = CascadeChecker(PKA_kin_eng=PKA_kin_eng, 
                                    supercell_size=supercell_size, 
                                    radius_frac=radius_frac,
                                    traj_folder=PKA_kin_eng_dir,
                                    grid=grid_value,
                                    model_name=model_name)
            print(f"Checking cascade outputs in {PKA_kin_eng_dir}...")
        elif model_name == 'STOPPING':
            PKA_kin_eng_dir = os.path.join(calc.calculation_dir, 'cascade', f'PKA_{PKA_kin_eng}eV', f'{radius_frac}-{cutoff}')
            checker = CascadeChecker(PKA_kin_eng=PKA_kin_eng, 
                                    supercell_size=supercell_size, 
                                    radius_frac=radius_frac,
                                    traj_folder=PKA_kin_eng_dir,
                                    grid=cutoff,
                                    model_name=model_name)
        checker.check_output(start_output=1, num_outputs=30, running_time=40) # in ps  


    '''
    ######################################### 6. Cascade data plotting ####################################
    '''
    if choice == '6':
        if model_name == 'STOPPING':
            print("Cascade data plotting...")
            plotter = LammpsCascadePlotter()
            print("Plotting stopping results...")
            folder = '/scratch/phys/t30429_nume-dft-ml/04-Ruoyan/02-Paper2/02-cascade/ml4cascades/ml4cascades/lammps/results/cascade/STOPPING/cascade/PKA_20000eV/0.7-1/5'
            fig_file = f'{folder}/stopping_results.png'
            #plotter.plot_stopping_results(datafile=f'{folder}/thermo.out', figfile=fig_file, ecut=10)
            # visualizer = LammpsCascadeVisualizer(task_name='Visualizing', model_name=model_name)
            # visualizer.exportDampfile(traj_folder=folder, export_step=10)
    
        elif model_name == 'EPH':
            plotter = LammpsCascadePlotter() 
            #print("Plotting eph results...")
            folder = '/scratch/phys/t30429_nume-dft-ml/04-Ruoyan/02-Paper2/02-cascade/ml4cascades/ml4cascades/lammps/results/cascade/EPH/cascade/PKA_20000eV-Clo/0.7-32/14'
            #folder = '/scratch/phys/t30429_nume-dft-ml/04-Ruoyan/02-Paper2/02-cascade/ml4cascades/ml4cascades/lammps/results/cascade/STOPPING/cascade/PKA_20000eV/0.7-1/5'
            fig_file1 = f'{folder}/eph_results.png'
            plotter.plot_eph_results(datafile=f'{folder}/thermo.out', figfile=fig_file1)

            # print('Coupling results...')
            parent_folder = Path('/scratch/phys/t30429_nume-dft-ml/04-Ruoyan/02-Paper2/02-cascade/ml4cascades/ml4cascades/lammps/results/cascade/EPH/cascade')
            folder_list = [parent_folder/'PKA_20000eV-Ce_5e-6/0.7-16/19', parent_folder/'PKA_20000eV-Ce_5e-7/0.7-16/19']
            label_list = ['High Ce', 'Low Ce']
            fig_file2 = parent_folder/'PKA_20000eV-Ce_5e-6/0.7-16/19/coupling.png'
            # plotter.plot_eph_coupling(folder_list, label_list, fig_file2)

            dump_file = f'{folder}/data.output'
            T_out_folder = f'{folder}/T_out'
            new_dump_file = f'{folder}/data.output.modified'
            new_tout_file = f'{folder}/T_out.modified'
            avg_Ta_file = f'{folder}/avg_Ta.out'
            avg_Te_file = f'{folder}/avg_Te.out'
            yes_dump = False
            yes_tout = False
            plotter.get_grid_Ta_Te(dump_file=dump_file, 
                                   T_out_folder=T_out_folder, 
                                   new_dump_file=new_dump_file, 
                                   new_tout_file=new_tout_file, 
                                   yes_dump=yes_dump,
                                   yes_tout=yes_tout,
                                   te_time_file=f'{folder}/record_time.out',
                                   block_id_list=[6, 10, 17],
                                   tout_step_id_list=[25, 33, 37])
            # plotter.get_Ta_time(dump_file=dump_file, out_ta_time_file=f'{folder}/Ta_time.out')
            # current, peak = tracemalloc.get_traced_memory()
            # print(f"Peak memory: {peak / 1e9:.2f} GB")
            # tracemalloc.stop()

            print("Plotting extreme Te...")
            num_of_frames = 4
            fig_file2 = f'{folder}/extreme_Te.png'
            txt_file2 = f'{folder}/extreme_Te.txt'
            #plotter.get_extreme_Te(new_tout_file, num_of_frames, fig_file2, txt_file2) 
            num_of_frames = 30
            fig_file2 = f'{folder}/extreme_Ta.png'
            txt_file2 = f'{folder}/extreme_Ta.txt'
            #plotter.get_extreme_Ta(new_dump_file, num_of_frames, fig_file2, txt_file2) 

            #print("Plotting Te and Ta along x...")
            grid_value = 32 
            height = 17
            width = 14
            grid_start = grid_value*grid_value*height + grid_value*width
            grid_end = grid_start + grid_value 
            grid_list = range(int(grid_start+grid_value/4), int(grid_end-grid_value/4))
            electron_grid_list = range(int(grid_start), int(grid_end))
            fig_file3 = f'{folder}/Te_Ta_along_x.png'
            plotter.plot_te_ta_along_x(new_tout_file, new_dump_file, 
                                       grid_list, electron_grid_list, fig_file3)

            # print("Plotting Te and Ta xy heatmaps...")
            fig_file4 = f'{folder}/Te_heatmap.png'
            fig_file5 = f'{folder}/Ta_heatmap.png'
            fig_file6 = f'{folder}/Tdiff_heatmap.png'
            plotter.plot_xy_heatmap(new_tout_file, new_dump_file,
                                    fig_file4, fig_file5, fig_file6,
                                    z=17, gridx=32, gridy=32, border=8)

            # visualizer = LammpsCascadeVisualizer(task_name='Visualizing', model_name=model_name)
            # visualizer.exportDampfile(traj_folder=folder, export_step=10)
    '''
    ######################################### 7. Cascade output processing ####################################
    '''
    if choice == '7':
        print("Cascade output processing...")
        PKA_kin_dirs = [
            #os.path.join(calc.calculation_dir, 'cascade', 'PKA_20000eV', '0.7-1'),
            #os.path.join(calc.calculation_dir, 'cascade', 'PKA_20000eV', '0.7-10'),
            #os.path.join(calc.calculation_dir, 'cascade', 'PKA_20000eV-Clo', '0.7-1'),
            #os.path.join(calc.calculation_dir, 'cascade', 'PKA_20000eV-Clo', '0.7-32'),
            os.path.join(calc.calculation_dir, 'cascade', 'PKA_50000eV-Td', '0.7-44'),
        ]
        for PKA_kin_eng_dir in PKA_kin_dirs:
            processor = CascadeProcessor(bi, PKA_kin_eng=50000, traj_folder=PKA_kin_eng_dir, model_name=model_name)
            reference_traj_file = '/scratch/phys/t30429_nume-dft-ml/04-Ruoyan/02-Paper2/02-cascade/ml4cascades/ml4cascades/lammps/results/cascade/STOPPING/minimize/100-100-100/data.minimize'
            #reference_traj_file = None
            processor.cal_WSDefect(start_traj=30, num_trajs=1, final=True, reference_traj_file=reference_traj_file)

            # processor.cal_liquid_atoms(start_traj=10, num_trajs=1)
            # processor.cal_local_T(start_traj=9, num_trajs=1, export_step=5)
            # parent_folder = os.path.join(calc.calculation_dir, 'cascade', 'PKA_10000eV-Ce_5e-7', '0.8-16')

            parent_folder = None

            # processor.cal_R2(start_traj=8, num_trajs=1)
            #processor.cal_R2_ovito(start_traj=1, num_trajs=35, reference_traj_file=reference_traj_file)
            # other_traj_folder = os.path.join(calc.calculation_dir, 'cascade', 'PKA_10000eV-Ce_5e-7', '0.8-16')
            other_traj_folder = None
            # processor.cal_avg_R2(other_traj_folder=other_traj_folder)

            
            processor.cal_defect_cluster_final(start_traj=30, num_trajs=1, final=True, reference_traj_file=reference_traj_file)
            # other_traj_folder = os.path.join(calc.calculation_dir, 'cascade', 'PKA_30000eV', '0.8-1')
            other_traj_folder = None
            # processor.cal_defect_cluster_final_avg(outlier_traj=outlier_traj, 
            #                                        other_traj_folder=other_traj_folder, 
            #                                        other_outlier_traj=other_outlier_traj)

    '''
    ######################################### 8. EPH processing ####################################
    '''
    if choice == '8':
        print("EPH processing...")
        parent_folder = Path('/scratch/phys/t30429_nume-dft-ml/04-Ruoyan/02-Paper2/02-cascade/ml4cascades/ml4cascades/lammps/results/cascade/EPH/cascade')
        folder_list = [parent_folder/'PKA_20000eV-Ce_5e-6/0.7-16/19', parent_folder/'PKA_20000eV-Ce_5e-7/0.7-16/19']
        label_list = ['5e-6', '5e-7']
        frame_id_list = [[69, 45], [37, 34], [23, 22]]
        time_list = [342, 89, 30]  # in fs       
        processor = EPHprocessor(task_name='EPHprocessing')
        processor.analyze_coupling(folder_list=folder_list,
                                   label_list=label_list,
                                   expression='c_ek>0.5',
                                   frame_idx_list=frame_id_list, 
                                   time_list=time_list)


    
  

