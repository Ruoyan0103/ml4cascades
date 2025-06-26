from ml4cascades.turbogap import TurboGAPCalculator
import os, subprocess, math, shutil, time
from ase.build import bulk 
from ase.io import read, write
import numpy as np


AMU_TO_KG = 1.66053906660E-27 # Atomic mass unit to kg conversion factor
JOULE_TO_EV = 6.241509074E18  # Joule to eV conversion factor
ANGSTROM_TO_METER = 1E-10     # Angstroms/picosecond to meters/second conversion factor
PS_TO_S = 1E-12               # Picoseconds to seconds conversion factor
module_dir = os.path.dirname(__file__) 


class CascadeCalculator(TurboGAPCalculator):
    """ 
    Threshold displacement energy calculator.
    """          
    def __init__(self, potential, num_species, mass, element, lattice, alat, sizes, temperature, 
                 pka_ids, energies, num_sampling_points, simulation_steps, task_name='pka'):
        """
        Initialize the CascadeCalculator.
        Args:
            potential (Potential): The potential object containing force field settings.
            mass (float): Mass of the atoms.
            element (str): Element symbol.
            lattice (str): Lattice type.
            alat (float): Lattice constant.
            sizes (list): List of supercell sizes for different PKA energies.
            temperature (float): Temperature for the simulation in Kelvin.
            pka_ids (list): List of primary knock-on atom IDs in different supercells.
            energies (list): List of energies for the PKA in eV.
            num_sampling_points (int): Number of random directions to sample.
            task_name (str, optional): Name of the task. Defaults to 'pka'.
        """
        super().__init__(task_name, num_species, potential, mass, element, lattice, alat)
        self.sizes = sizes
        self.temp = temperature
        self.pka_ids = pka_ids
        self.energies = energies
        self.num_sampling_points = num_sampling_points 
        self.simulation_steps = simulation_steps
        self.angle_set = set()
        self.hkl_list = []
        self.min_phi = 0
        self.max_phi = 54.7
        self.min_theta = 0
        self.max_theta = 45


    def _get_random_angles(self, min_phi, max_phi, min_theta, max_theta, num_points):
        """
        Generate random angles in spherical coordinates.
        Args:
            min_phi (float): Minimum azimuthal angle in degrees.
            max_phi (float): Maximum azimuthal angle in degrees.
            min_theta (float): Minimum polar angle in degrees.
            max_theta (float): Maximum polar angle in degrees.
            num_points (int): Number of random points to generate.
        """
        _min_phi = np.radians(min_phi)
        _max_phi = np.radians(max_phi)
        _min_theta = np.radians(min_theta)
        _max_theta = np.radians(max_theta)
        np.random.seed(42)  
        phi = np.random.uniform(_min_phi, _max_phi, num_points)                        # azimuthal angle (φ)
        costheta = np.random.uniform(np.cos(_min_theta), np.cos(_max_theta), num_points) 
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
        shutil.copy(os.path.join(self.template_dir, 'submit-mahti.sh'), os.path.join(relax_dir, 'submit.sh'))
        with open(os.path.join(self.template_dir, 'input-relax'), 'r') as f:
            input_template = f.read()
        input_file = os.path.join(relax_dir, 'input')
        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings='\n'.join(self.ff_settings), num_species=self.num_species,
                                          element=self.element, mass=self.mass, Temp=self.temp))
        unit_cell = bulk(self.element, self.lattice, a=self.alat, cubic=True)
        super_cell = unit_cell * [size, size, size]
        write(os.path.join(relax_dir, 'data.input'), super_cell, format='extxyz')
        subprocess.run('sbatch submit.sh', shell=True, check=True, cwd=relax_dir)


    def _get_pka_id(self, trajectory_file):
        relaxed_struct = read(trajectory_file, format='extxyz', index=-1)
        center = traj.get_center_of_mass()
        positions_list = relaxed_struct.get_positions()
        dist = 1000
        closest_idx = None
        for idx, positions in enumerate(positions_list):
            temp_dist = np.linalg.norm(positions - center)
            if temp_dist < dist:
                dist = temp_dist
                closest_idx = idx
        pka_id = relaxed_struct.get_atomic_numbers()[closest_idx]
        return pka_id



    def _setup_helper(self, velocity, pka_id, hkl, eng_hkl_dir, trajectory_file):
        """
        Helper function to set up the input file for the LAMMPS simulation.
        Args:
            velocity (float): Velocity of the PKA in m/s.
            pka_id (int): ID of the primary knock-on atom.
            hkl (np.array): Miller indices for the direction of the PKA.
            eng_hkl_dir (str): Directory for the specific energy and hkl combination.
            trajectory_file (str): Path to the trajectory file.
        """
        with open(os.path.join(self.template_dir, 'in.pka'), 'r') as f:
            input_template = f.read()
        ff_settings = self.ff_settings
        input_file = os.path.join(eng_hkl_dir, 'input')
        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings='\n'.join(ff_settings), num_species=self.num_species,
                                          element=self.element, mass=self.mass, Temp=self.temp, simulation_steps=self.simulation_steps,
                                          stopping_file=os.path.join(template_dir, 'Ge_Ge_elstop.txt')))
        relaxed_struct = read(trajectory_file, format='extxyz', index=-1)
        velocities = relaxed_struct.get_velocities()
        with open(new_trajectory_file, 'w') as f:
            Vx = velocity * hkl[0]
            Vy = velocity * hkl[1]
            Vz = velocity * hkl[2]
            velocities[pka_id]  = [Vx, Vy, Vz]  
            relaxed_struct.set_velocities(velocities)
            new_trajectory_file = os.path.join(eng_hkl_dir, 'thermalized.xyz')
            write(new_trajectory_file, relaxed_struct, format='extxyz')
        
                    
    def _setup(self):
        """
        Set up the directories and input files for the LAMMPS simulation.
        """
        self._get_random_angles(self.min_phi, self.max_phi, self.min_theta, self.max_theta, self.num_sampling_points)
        self._set_hkl_from_angles()
        for energy, size in zip(self.energies, self.sizes):
            eng_dir = os.path.join(self.calculation_dir, str(int(energy)))
            self._relax(eng_dir, size)
            trajectory_file = os.path.join(eng_dir, 'trajectory_out.xyz')
            pka_id = self._get_pka_id(trajectory_file)  
            # energy = 0.5 * self.mass * AMU_TO_KG * np.sum(hkl**2) * (velocity*ANGSTROM_TO_METER/PS_TO_S)**2 * JOULE_TO_EV
            # np.sum(hkl**2) approximate to 1
            velocity = np.sqrt(2 * energy  / (self.mass * AMU_TO_KG * JOULE_TO_EV)) / (ANGSTROM_TO_METER/PS_TO_S) 
            for idx, hkl in enumerate(self.hkl_list):
                eng_hkl_dir = os.path.join(eng_dir, str(idx))
                os.makedirs(eng_hkl_dir, exist_ok=True)
                self._setup_helper(velocity, pka_id, hkl, eng_hkl_dir, trajectory_file)
                shutil.copy(os.path.join(self.template_dir, 'submit-mahti.sh'), os.path.join(eng_hkl_dir, 'submit.sh')) 


    def calculate(self):
        """
        Run the cascade calculations.
        """
        self._setup()
        for energy in self.energies:
            eng_dir = os.path.join(self.calculation_dir, str(int(energy)))
            for idx, _ in enumerate(self.hkl_list):
                eng_hkl_dir = os.path.join(eng_dir, str(idx))
                subprocess.run('sbatch submit.sh', shell=True, check=True, cwd=eng_hkl_dir)






   
    



