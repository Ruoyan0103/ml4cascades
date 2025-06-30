from ml4cascades.turbogap import TurboGAPCalculator
import os, subprocess, math, shutil, time
from ase.build import bulk 
from ase.io import read, write
import numpy as np


AMU_TO_KG = 1.66053906660E-27 # Atomic mass unit to kg conversion factor
JOULE_TO_EV = 6.241509074E18  # Joule to eV conversion factor
ANGSTROM_TO_METER = 1E-10     # Angstroms/picosecond to meters/second conversion factor
FS_TO_S = 1E-15               # Picoseconds to seconds conversion factor
module_dir = os.path.dirname(__file__) 


class CascadeCalculator(TurboGAPCalculator):
    """ 
    Threshold displacement energy calculator.
    """          
    def __init__(self, potential, num_species, mass, element, lattice, alat, sizes, temperature, 
                 energies, num_sampling_points, equilibration_steps, cascade_steps, gap_file_folder, task_name='pka'):
        """
        Initialize the CascadeCalculator.
        Args:
            potential (Potential): The potential object containing force field settings.
            num_species (int): Number of species in the system.
            mass (float): Mass of the atoms.
            element (str): Element symbol.
            lattice (str): Lattice type.
            alat (float): Lattice constant.
            sizes (list): List of supercell sizes for different PKA energies.
            temperature (float): Temperature for the simulation in Kelvin.
            pka_ids (list): List of primary knock-on atom IDs in different supercells.
            energies (list): List of energies for the PKA in eV.
            num_sampling_points (int): Number of random directions to sample.
            equilibration_steps (int): Number of equilibration steps.
            cascade_steps (int): Number of steps in the cascade simulation.
            gap_file_folder (str): Path to the folder containing GAP files.
            task_name (str, optional): Name of the task. Defaults to 'pka'.
        """
        super().__init__(task_name, potential, num_species, mass, element, lattice, alat)
        self.sizes = sizes
        self.temp = temperature
        self.energies = energies
        self.num_sampling_points = num_sampling_points
        self.equilibration_steps = equilibration_steps
        self.cascade_steps = cascade_steps
        self.gap_file_folder = gap_file_folder
        self.angle_set = set()
        self.hkl_list = []
        self.min_phi = 0
        self.max_phi = 54.7
        self.min_theta = 0
        self.max_theta = 45


    def _get_random_angles(self):
        """
        Generate random angles in spherical coordinates.
        """
        np.random.seed(42)  
        phi = np.random.uniform(self.min_phi, self.max_phi, self.num_points)           # azimuthal angle (φ)
        costheta = np.random.uniform(np.cos(self.min_theta), np.cos(self.max_theta), self.num_points) 
        theta = np.arccos(costheta)                                                    # polar angle (θ)
        self.angle_set = set(zip(phi, theta))                                          # Store unique angles

    
    def _set_hkl_from_angles(self, threshold_deg=15, num_directions=1):
        """
        Convert spherical angles to normalized Miller indices (hkl).
        """
        cos_thresh = np.cos(np.radians(threshold_deg))
        channeling_vectors = [
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]
        channeling_dirs = [v / np.linalg.norm(np.array(v)) for v in channeling_vectors]
        for angle in self.angle_set:
            if len(self.hkl_list) >= num_directions:
                break
            phi, theta = angle
            h = np.sin(theta) * np.cos(phi)
            k = np.sin(theta) * np.sin(phi)
            l = np.cos(theta)
            hkl = np.array((h, k, l))
            add_hkl = True
            for d in channeling_dirs:
                if abs(np.dot(hkl, d)) > cos_thresh:
                       add_hkl = False
            if add_hkl:
                self.hkl_list.append(hkl)
        hkl_file = os.path.join(self.calculation_dir, 'hkl_list.dat')
        with open(hkl_file, 'w') as f:  
            np.savetxt(f, np.array(self.hkl_list), 
                       fmt='%.8f',      
                       delimiter=' ',   
                       header='h     k     l') 


    def _relax(self, relax_dir, size): 
        """
        Set up the relaxation simulation for a given supercell size.
        Args:
            relax_dir (str): Directory for the relaxation simulation.
            size (int): Size of the supercell.
        """
        with open(os.path.join(self.template_dir, 'submit-triton.sh'), 'r') as f:
            submit_template = f.read()
        submit_file = os.path.join(relax_dir, 'submit.sh')
        with open(submit_file, 'w') as f:
            f.write(submit_template.format(job_name=f'relax_{size}{size}{size}'))
        with open(os.path.join(self.template_dir, 'input-relax'), 'r') as f:
            input_template = f.read()
        input_file = os.path.join(relax_dir, 'input')
        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings=self.ff_settings, num_species=self.num_species,
                                          element=self.element, mass=self.mass, equilibration_steps=self.equilibration_steps,
                                          Temp=self.temp))
        unit_cell = bulk(self.element, self.lattice, a=self.alat, cubic=True)
        super_cell = unit_cell * [size, size, size]
        write(os.path.join(relax_dir, 'data.input'), super_cell, format='extxyz')
        # subprocess.run('sbatch submit.sh', shell=True, check=True, cwd=relax_dir)


    def _get_pka_id(self, trajectory_file):
        relaxed_struct = read(trajectory_file, format='extxyz', index=-1)
        center = relaxed_struct.get_center_of_mass()
        positions_list = relaxed_struct.get_positions()
        dist = 1000
        closest_idx = None
        for idx, positions in enumerate(positions_list):
            temp_dist = np.linalg.norm(positions - center)
            if temp_dist < dist:
                dist = temp_dist
                closest_idx = idx
        pka_id = closest_idx
        self.logger.info(f'PKA ID: {pka_id} with positions: {positions_list[pka_id]}')
        return relaxed_struct, pka_id


    def _setup_helper(self, velocity, relaxed_struct, pka_id, hkl, eng_hkl_dir):
        """
        Helper function to set up the input file for the LAMMPS simulation.
        Args:
            velocity (float): Velocity of the PKA in m/s.
            relaxed_struct (ase.Atoms): Relaxed structure with velocities set.
            pka_id (int): ID of the primary knock-on atom.
            hkl (np.array): Miller indices for the direction of the PKA.
            eng_hkl_dir (str): Directory for the specific energy and hkl combination.
        """
        with open(os.path.join(self.template_dir, 'input-pka'), 'r') as f:
            input_template = f.read()
        ff_settings = self.ff_settings
        input_file = os.path.join(eng_hkl_dir, 'input')
        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings=ff_settings, num_species=self.num_species,
                                          element=self.element, mass=self.mass, Temp=self.temp, cascade_steps=self.cascade_steps,
                                          stopping_file=os.path.join(self.template_dir, 'Ge_Ge_elstop.txt')))
        velocities = relaxed_struct.get_array('velocities')
        self.logger.info(f'PKA ID {pka_id} with old velocities {velocities[pka_id]} ang/fs')
        Vx = velocity * hkl[0]
        Vy = velocity * hkl[1]
        Vz = velocity * hkl[2]
        velocities[pka_id]  = [Vx, Vy, Vz]  
        relaxed_struct.set_array('velocities', velocities)
        self.logger.info(f'PKA ID {pka_id} with new velocities {velocities[pka_id]} ang/fs')
        new_trajectory_file = os.path.join(eng_hkl_dir, 'thermalized.xyz')
        write(new_trajectory_file, relaxed_struct, format='extxyz')
        
                    
    def _setup(self):
        """
        Set up the directories and input files for the LAMMPS simulation.
        """
        self._get_random_angles()
        self._set_hkl_from_angles()
        for energy, size in zip(self.energies, self.sizes):
            eng_dir = os.path.join(self.calculation_dir, str(int(energy)))
            os.makedirs(eng_dir, exist_ok=True)
            shutil.copytree(self.gap_file_folder, os.path.join(eng_dir, 'gap_files'), dirs_exist_ok=True)
            self._relax(eng_dir, size)
        
    
    def calculate(self, runcascade=False):
        """
        Run the cascade calculations.
        """
        self._setup()
        # time.sleep(120)
        if runcascade:
            for energy in self.energies:
                self.logger.info(f'---------------------------------------------------')
                self.logger.info(f'Running cascade calculations for energy: {energy} eV')
                eng_dir = os.path.join(self.calculation_dir, str(int(energy)))
                trajectory_file = os.path.join(eng_dir, 'trajectory_out.xyz')
                relaxed_struct, pka_id = self._get_pka_id(trajectory_file)
                velocity = np.sqrt(2 * energy  / (self.mass * AMU_TO_KG * JOULE_TO_EV)) / (ANGSTROM_TO_METER/FS_TO_S) 
                for idx, hkl in enumerate(self.hkl_list):
                    eng_hkl_dir = os.path.join(eng_dir, str(idx))
                    os.makedirs(eng_hkl_dir, exist_ok=True)
                    shutil.copytree(self.gap_file_folder, os.path.join(eng_hkl_dir, 'gap_files'), dirs_exist_ok=True)
                    self._setup_helper(velocity, relaxed_struct, pka_id, hkl, eng_hkl_dir)
                    shutil.copy(os.path.join(self.template_dir, 'submit-triton.sh'), os.path.join(eng_hkl_dir, 'submit.sh')) eng_hkl_dir = os.path.join(eng_dir, str(idx))
                    subprocess.run('sbatch submit.sh', shell=True, check=True, cwd=eng_hkl_dir)


    def check_border(self):
        """
        Check no atoms are outside the simulation box.
        """
        pass 
                






   
    



