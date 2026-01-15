import numpy as np
import os
from ase.io import read 
from ml4cascades.loggers.logger import AppLogger

module_dir = os.path.dirname(__file__)

AMU_TO_KG = 1.66053906660E-27 # Atomic mass unit to kg conversion factor
JOULE_TO_EV = 6.241509074E18  # Joule to eV conversion factor
ANGSTROM_TO_METER = 1E-10     # Angstrom to meter conversion factor
FS_TO_S = 1E-15               # Femtosecond to second conversion factor

class CascadeChecker:
    def __init__(self,
                 PKA_kin_eng: float,
                 supercell_size: list[int],
                 radius_frac: float,
                 traj_folder: str,
                 successful_folder: str,
                 task_name='checking',
                 model_name='EPH'):
        self.PKA_kin_eng = PKA_kin_eng
        self.supercell_size = supercell_size
        self.radius_frac = radius_frac
        self.traj_folder = traj_folder
        self.successful_folder = successful_folder
        os.makedirs(self.successful_folder, exist_ok=True)
        self.log_dir = os.path.join(module_dir, 'logs', task_name)
        self.log_file = os.path.join(self.log_dir, f'{model_name}.log')
        self.logger = AppLogger(__name__, self.log_file, overwrite=True).get_logger()
        os.makedirs(self.log_dir, exist_ok=True)

    def check_structure(self, 
                        start_traj: int,
                        num_trajs: int, 
                        border_thickness: float,
                        pot_eng_threshold: float,
                        kin_eng_threshold: float):
        # pot_eng_threshold in eV, initial lattice potential energies from -4.5 to -4.3 eV
        # after cascade, atoms in the border region should not have to much higher potential energy than initial
        # kin_eng_threshold in eV
        self.logger.info(f'#---------Checking, PKA energy: {self.PKA_kin_eng}, supercellsize: {self.supercell_size[0]}-{self.supercell_size[1]}-{self.supercell_size[2]}, radius_frac: {self.radius_frac}---------#')
        failed_case = 0
        for num_traj in range(num_trajs):
            traj_file = os.path.join(self.traj_folder, f'{start_traj+num_traj}', 'trajectory_out.xyz')
            structs = read(traj_file, format='extxyz', index=':60')
            num_structs = len(structs)
            self.logger.info(f'Checking trajectory {start_traj+num_traj}, total frames: {num_structs}.')
            failed_flag = False       
            for i in range(20, min(num_structs, 60)):    # checking the ballistic phase, num_structs usually 50 corresponding to 2 ps
                struct = structs[i]
                positions = struct.get_positions()
                velocities = struct.get_array('velocities')
                local_energies = struct.get_array('local_energy')
                for idx, p in enumerate(positions):
                    if (p[0] < border_thickness or p[0] > struct.cell[0,0] - border_thickness or \
                        p[1] < border_thickness or p[1] > struct.cell[1,1] - border_thickness or \
                        p[2] < border_thickness or p[2] > struct.cell[2,2] - border_thickness):
                        pot_eng = local_energies[idx]                                          # in eV
                        mass = struct.get_masses()[idx]*AMU_TO_KG                              # in kg
                        vel = np.array([v*ANGSTROM_TO_METER/FS_TO_S for v in velocities[idx]]) # in m/s
                        kin_eng = 0.5 * mass * (vel @ vel) * JOULE_TO_EV                       # in eV
                        if pot_eng > pot_eng_threshold or kin_eng > kin_eng_threshold:
                            self.logger.info(f'Trajectory {start_traj+num_traj} failed on the {i} frame, pot_eng: {pot_eng} eV, kin_eng: {kin_eng} eV.')
                            failed_flag = True
                            break
                if failed_flag:
                    break
            if failed_flag: 
                failed_case += 1
        self.logger.info(f'#---------{failed_case}/{num_trajs} cases failed.---------#')

            

                







