'''
TGAP and GAP are using the same potential but in different format 
and simulate with different engine (LAMMPS and TurboGAP)
'''

import os, re, time, sys, argparse
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
        return os.path.join(module_dir, 'params', 'TGAP', 'gap_files')

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

    gap_file = 'Ge.gap'
    tgap = TGAPotential(gap_file)
    bi = BasicCellInfo(element=['Ge'], atomic_num=[32], mass=72.64, lattice='diamond', alat=[5.76]*3)
    supercell_size = [30]*3
    model_name = 'EPH'  # 'EPH' or 'STOPPING' or 'STOPPING-0K'
    calc = CascadeCalculator(tgap, bi, model_name=model_name)

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
                                running_directions=range(1, num_PKA_directions+1),
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
    Ce = 5e-7           # eV/ps/A^3/K
    kappa_e = 2.529e-4  # eV/ps/A/K
    if choice == '3':
        print("Thermalizing...")
        atomic_config = {
            "supercell_size": supercell_size,
            "atomic_equ_md_steps": 10000,
            "temp": 300,
            "taut": 100,
            "press": 0,
            "taup": 50,
            "gammap": 0.001
        }

        electronic_config = {
            "supercell_size": supercell_size,
            "elec_equ_md_steps": 3000,
            "temp": 300,
            "xlow": -xhi/2+bi.alat[0]*supercell_size[0]/2,
            "xhigh": xhi/2+bi.alat[0]*supercell_size[0]/2,
            "ylow": -yhi/2+bi.alat[1]*supercell_size[1]/2,
            "yhigh": yhi/2+bi.alat[1]*supercell_size[1]/2,
            "zlow": -zhi/2+bi.alat[2]*supercell_size[2]/2,
            "zhigh": zhi/2+bi.alat[2]*supercell_size[2]/2,
            "eph_C_e": Ce,
            "eph_kappa_e": kappa_e,
            "eph_tout_file": 'eph-ToutData.txt',
            "gx": 8,
            "gy": 8,
            "gz": 8
        }
        calc.thermalize(atomic_config, electronic_config)

    '''
    ######################################### 4. Cascade simulation #########################################
    '''
    if choice == '4':
        print("Running cascade simulation...")
        lammps_thermalize_file = '/scratch/phys/t30429_nume-dft-ml/04-Ruoyan/03-Paper3/01-GAP/ml4cascades/ml4cascades/lammps/results/cascade/STOPPING/thermalize/30-30-30/dump_atoms/atoms.11000.dump'
        if lammps_thermalize_file is not None:
            with open(lammps_thermalize_file, 'r') as f:
                box_lines = f.readlines()[5:8]
            xlo, xhi = (float(v) for v in box_lines[0].split())
            ylo, yhi = (float(v) for v in box_lines[1].split())
            zlo, zhi = (float(v) for v in box_lines[2].split())
        else:
            atomsfile = os.path.join(calc.calculation_dir, 'thermalize_atomic', f'{supercell_size[0]}-{supercell_size[1]}-{supercell_size[2]}', 'thermalize', 'thermalized.data')
            with open(atomsfile, 'r') as f:
                lattice_line = f.readlines()[1]
            lattice_vals = [float(v) for v in re.search(r'Lattice="([^"]+)"', lattice_line).group(1).split()]
            xlo, ylo, zlo = 0.0, 0.0, 0.0
            xhi, yhi, zhi = lattice_vals[0], lattice_vals[4], lattice_vals[8]

        num_PKA_directions = 50
        radius_frac = 0.7
        PKA_kin_eng = 2000 # in eV
        grid_value = 16
        input_config = {
            "supercell_size": supercell_size,
            "border_thickness": 5.76,
            "cascade_steps": 5000,
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
            "eph_tout_file": 'eph-ToutData.txt'
        }
        running_directions = np.arange(6, 7)
        PKA_kin_eng_dir = calc.run_cascade(num_PKA_directions=num_PKA_directions,
                                           running_directions=running_directions,
                                           radius_frac=radius_frac,
                                           PKA_kin_eng=PKA_kin_eng,
                                           input_config=input_config,
                                           lammps_thermalize=lammps_thermalize_file)

    '''
    ######################################### 5. Cascade checker #########################################
    '''
    if choice == '5':
        print("Checking cascade results...")
        traj_folder = os.path.join(calc.calculation_dir, 'cascade', 'PKA_1000eV-0.7')
        checker = CascadeChecker(PKA_kin_eng=1000, 
                                supercell_size=supercell_size, 
                                radius_frac=0.7, 
                                traj_folder=traj_folder,
                                successful_folder=os.path.join(calc.calculation_dir, 'cascade', 'PKA_1000eV-suc'))
        checker.check_structure(start_traj=1, num_trajs=3, border_thickness=border_thickness, pot_eng_threshold=-4, kin_eng_threshold=5)

    '''
    ######################################### 6. Cascade data plotting ####################################
    '''
    if choice == '6':
        print("Plotting cascade data...")
        ploter = TurbogapCascadePlotter()
        ploter.plot_eph_results(datafile=os.path.join(calc.calculation_dir, 'cascade', 'PKA_2000eV-0.7', '4', 'eph-EnergySharingData.txt'),
                                figfile=os.path.join(calc.calculation_dir, 'cascade', 'PKA_2000eV-0.7', '4', 'eph-EnergySharingData.png'))

        # ploter.plot_thermo_results(thermofile='/scratch/phys/t30429_nume-dft-ml/04-Ruoyan/03-Paper3/01-GAP/ml4cascades/ml4cascades/turbogap/results/cascade/EPH/thermalize_atomic/30-30-30/thermalize/thermo.log',
        #                             figfile='/scratch/phys/t30429_nume-dft-ml/04-Ruoyan/03-Paper3/01-GAP/ml4cascades/ml4cascades/turbogap/results/cascade/EPH/thermalize_atomic/30-30-30/thermalize/thermo.png')

        # ploter.plot_stopping_results(thermofile=os.path.join(calc.calculation_dir, 'cascade', 'PKA_2000eV-0.7', '2', 'thermo.log'),
        #                            elossfile=os.path.join(calc.calculation_dir, 'cascade', 'PKA_2000eV-0.7', '2', 'ElectronicEnergyLoss.txt'),
        #                            figfile=os.path.join(calc.calculation_dir, 'cascade', 'PKA_2000eV-0.7', '2', 'thermo.png'))

        folder = '/scratch/phys/t30429_nume-dft-ml/04-Ruoyan/03-Paper3/01-GAP/ml4cascades/ml4cascades/turbogap/results/cascade/EPH/cascade/PKA_2000eV-0.7/5/'
        tout_file = os.path.join(folder, 'eph-ToutData.txt')
        new_tout_file = os.path.join(folder, 'eph-ToutData.modified')
        traj_file = os.path.join(folder, 'trajectory_out.xyz')
        new_dump_file = os.path.join(folder, 'trajectory_out.modified')
        mass_map = {bi.element[0]: bi.mass}
        with open(os.path.join(folder, 'input')) as f:
            for line in f:
                key = line.strip().split('=')[0].strip()
                if key == 'eph_box_limits':
                    box_limits = tuple(float(v) for v in line.split('=')[1].split('!')[0].split())
                elif key == 'eph_gsx':
                    gsx = int(line.split('=')[1].split('!')[0])
                elif key == 'eph_gsy':
                    gsy = int(line.split('=')[1].split('!')[0])
                elif key == 'eph_gsz':
                    gsz = int(line.split('=')[1].split('!')[0])
        grid_dims = (gsx, gsy, gsz)

        # eph-ToutData.txt has one block per eph_freq_mesh_Tout steps, starting
        # at step eph_freq_mesh_Tout (no step-0 block); trajectory_out.xyz has
        # one frame per same-size step interval but *does* include step 0. So
        # trajectory frame index (tout block index + 1) is the one that lines
        # up in step/time with tout block index -- read both frame counts and
        # times directly from the files instead of hardcoding them.
        with open(tout_file, 'r') as f:
            n_tout_blocks = sum(1 for line in f if line.strip().lstrip('-').isdigit() and len(line.split()) == 1)

        traj_steps = []
        with open(traj_file, 'r') as f:
            while True:
                n_line = f.readline()
                if not n_line:
                    break
                n_atoms = int(n_line.strip())
                comment = f.readline()
                traj_steps.append(int(re.search(r'\bstep=(\S+)', comment).group(1)))
                for _ in range(n_atoms):
                    f.readline()

        n_frames = min(n_tout_blocks, len(traj_steps) - 1)

        if n_frames > 0:
            tout_step_id_list = list(range(n_frames))
            frame_id_list = list(range(1, n_frames + 1))

            print("Converting eph-ToutData.txt to unified Tout grid format...")
            ploter.get_grid_Te(tout_file=tout_file,
                                new_tout_file=new_tout_file,
                                box_limits=box_limits,
                                tout_step_id_list=tout_step_id_list)
            elec_tout_file = new_tout_file
        else:
            # no Tout block has a matching trajectory frame yet (run still in
            # progress) -- fall back to just the first trajectory frame
            # (step 0) and skip the electronic side (Te/Tdiff) entirely
            print("No matched Te/Ta frame yet; plotting only the first trajectory frame (step 0).")
            frame_id_list = [0]
            elec_tout_file = None

        print("Converting trajectory_out.xyz to unified Ta grid format...")
        ploter.get_grid_Ta(traj_file=traj_file,
                            new_dump_file=new_dump_file,
                            box_limits=box_limits,
                            grid_dims=grid_dims,
                            mass_map=mass_map,
                            frame_id_list=frame_id_list)

        # print("Plotting extreme Te/Ta...")
        n_grid = gsx * gsy * gsz
        ploter.get_extreme_Te_Ta(tout_file=new_tout_file,
                                 dump_file=new_dump_file,
                                 frame_idx_list=list(range(n_frames)),
                                 grid_list=range(n_grid),
                                 electron_grid_list=range(n_grid),
                                 figfile=os.path.join(folder, 'extreme_Te_Ta.png'))
        ploter.plot_xy_heatmap(tout_file=elec_tout_file,
                        dump_file=new_dump_file,
                        figfile1=os.path.join(folder, 'Te_heatmap.png'),
                        figfile2=os.path.join(folder, 'Ta_heatmap.png'),
                        figfile3=os.path.join(folder, 'Tdiff_heatmap.png'),
                        z=4, gridx=gsx, gridy=gsy, border=2)

        # iy0, iz0 = 4, 5  # y=4, z=5, sweeping x
        # grid_start = iz0 * (gsx * gsy) + iy0 * gsx
        # atomic_grid_list = range(grid_start, grid_start + gsx)
        # electron_grid_list = atomic_grid_list  # or a wider/narrower slice as needed
        # ploter.plot_te_ta_along_x(tout_file=new_tout_file,
        #                         dump_file=new_dump_file,
        #                         atomic_grid_list=atomic_grid_list,
        #                         electron_grid_list=electron_grid_list,
        #                         figfile=os.path.join(folder, 'Te_Ta_along_x.png'),
        #                         axis=0,
        #                         frame_idx_list=[0, 10, 20, 30, 40])

    '''
    ######################################### 7. Cascade output processing ####################################
    '''
    if choice == '7':
        print("Processing cascade outputs...")
        traj_folder = os.path.join(calc.calculation_dir, 'cascade', 'PKA_2000eV-0.7')
        processor = CascadeProcessor(bi, PKA_kin_eng=2000, traj_folder=traj_folder)
        #processor.cal_ibm(num_trajs=20, n0=1, ed=1)
        processor.cal_WSDefect(start_traj=4, num_trajs=1, final=True)
        #processor.cal_cluster(start_traj=1, num_trajs=3, expression='Occupancy!=1')
