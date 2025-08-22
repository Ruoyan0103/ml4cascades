from ml4cascades.lammps import LMPStaticCalculator
from ml4cascades.lammps.post_process import process_hkl, postProcess, process_hkl, _timeDependentAnalysis, _countVacAndInter, _clustersAndDXA
import os, subprocess, math, shutil, time
import numpy as np
from ase.build import bulk
from ase.io import read, write
from ovito.io import import_file
from ovito.modifiers import WignerSeitzAnalysisModifier, ClusterAnalysisModifier, ExpressionSelectionModifier
from ovito.modifiers import CalculateDisplacementsModifier
from ovito.pipeline import StaticSource, Pipeline
from collections import Counter
import logging
import multiprocessing as mp
import glob

AMU_TO_KG = 1.66053906660E-27  # Atomic mass unit to kg conversion factor
JOULE_TO_EV = 6.241509074E18   # Joule to eV conversion factor
ANGSTROM_TO_METER = 1E-10      # Angstroms/picosecond to meters/second conversion factor
PS_TO_S = 1E-12                # Picoseconds to seconds conversion factor
module_dir = os.path.dirname(__file__)

# Constant atomic number density for germanium
n0 = 0.0442  # atoms/Å³

# Lookup table for ionization and damage energies
energy_lookup = {
    100: (14.2, 85.8),    # (ionization, damage)
    400: (61.8, 338.2),
    1000: (168.2, 831.8),
    2000: (361.4, 1638.6),
    5000: (1016, 3984),
    10000: (2420, 7580),
    20000: (6330, 13670),
    50000: (22300, 27700)
}

class CascadeCalculator(LMPStaticCalculator):

    """
    Threshold displacement energy calculator.
    """

    postProcess = postProcess 
    process_hkl = process_hkl
    _timeDependentAnalysis = _timeDependentAnalysis
    _countVacAndInter = _countVacAndInter
    _clustersAndDXA = _clustersAndDXA
    
    def __init__(self, potential, mass, element, lattice, alat, sizes, thicknesses, radius_fracs, temperature,
                 energies, num_sampling_points, task_name='pka'):
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
            energies (list): List of energies for the PKA in eV.
            num_sampling_points (int): Number of random directions to sample.
            task_name (str, optional): Name of the task. Defaults to 'pka'.
        """
        super().__init__(task_name, potential, mass, element, lattice, alat)
        self.sizes = sizes
        self.border_thicknesses = thicknesses
        self.radius_fracs = radius_fracs
        self.temp = temperature
        self.energies = energies
        self.num_sampling_points = num_sampling_points
        self.angle_set = set()
        self.hkl_list = []
        self.min_phi = 0
        self.max_phi = 54.7
        self.min_theta = 0
        self.max_theta = 45
        self.num_directions = 30

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
        np.random.seed(96)
        phi = np.random.uniform(_min_phi, _max_phi, num_points)
        costheta = np.random.uniform(np.cos(_min_theta), np.cos(_max_theta), num_points)
        theta = np.arccos(costheta)
        self.angle_set = set(zip(phi, theta))

    def _set_hkl_from_angles(self, threshold_deg=15, eng_dir=None):
        """
        Convert spherical angles to normalized Miller indices (hkl) and save to hkl_list.dat in eng_dir.
        Args:
            threshold_deg (float): Threshold angle for avoiding channeling directions.
            eng_dir (str): Directory to save hkl_list.dat (e.g., calculations/energy).
        """
        cos_thresh = np.cos(np.radians(threshold_deg))
        channeling_vectors = [
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]
        channeling_dirs = [v / np.linalg.norm(np.array(v)) for v in channeling_vectors]
        self.hkl_list = []
        for angle in self.angle_set:
            if len(self.hkl_list) >= self.num_directions:
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

        if eng_dir:
            hkl_file = os.path.join(eng_dir, 'hkl_list.dat')
            os.makedirs(eng_dir, exist_ok=True)
            with open(hkl_file, 'w') as f:
                np.savetxt(f, np.array(self.hkl_list),
                           fmt='%.8f',
                           delimiter=' ',
                           header='h     k     l')
                self.logger.info(f"Saved hkl_list.dat to {hkl_file}")

    def _get_pka_id_center(self, trajectory_file):
        relaxed_struct = read(trajectory_file, format='lammps-data', index=-1)
        center = relaxed_struct.get_center_of_mass()
        pka_id = self._get_pka_id(relaxed_struct, center)
        return relaxed_struct, pka_id

    def _get_pka_id_sphere(self, trajectory_file, hkl, radius_frac):
        relaxed_struct = read(trajectory_file, format='lammps-data', index=-1)
        a, b, c, _, _, _ = relaxed_struct.get_cell_lengths_and_angles()
        radius = 0.5 * min(a, b, c) * radius_frac
        position = hkl * radius + relaxed_struct.get_center_of_mass()
        pka_id = self._get_pka_id(relaxed_struct, position)
        return relaxed_struct, pka_id

    def _get_pka_id(self, relaxed_struct, target_position):
        positions_list = relaxed_struct.get_positions()
        dist = 1000
        closest_idx = None
        for idx, positions in enumerate(positions_list):
            temp_dist = np.linalg.norm(positions - target_position)
            if temp_dist < dist:
                dist = temp_dist
                closest_idx = idx
        pka_id = closest_idx + 1
        return pka_id

    def _setup_helper(self, velocity, pka_id, heatsink_thickness, hkl, eng_hkl_dir):
        """
        Helper function to set up the input file for the LAMMPS simulation.
        """
        with open(os.path.join(self.template_dir, 'in.pka'), 'r') as f:
            input_template = f.read()
        ff_settings = self.ff_settings
        input_file = os.path.join(eng_hkl_dir, 'in.pka')
        Vx = velocity * -hkl[0]
        Vy = velocity * -hkl[1]
        Vz = velocity * -hkl[2]
        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings='\n'.join(ff_settings), mass=self.mass,
                                          pka_id=pka_id, Temp=self.temp,
                                          V_x=Vx, V_y=Vy, V_z=Vz,
                                          heatsink_thickness=heatsink_thickness,
                                          border_thickness=heatsink_thickness))
        self.logger.info(f'PKA ID: {pka_id} with Velocities: {Vx:.2f}, {Vy:.2f}, {Vz:.2f} ang/ps')
        shutil.copy(os.path.join(self.template_dir, 'Ge_Ge_elstop.txt'),
                    os.path.join(eng_hkl_dir, 'Ge_Ge_elstop.txt'))

    def _setup(self, velocity, pka_id, border_thickness, hkl, eng_hkl_dir):
        """
        Set up the directories and input files for the LAMMPS simulation.
        """
        super()._setup(eng_hkl_dir)
        self._setup_helper(velocity, pka_id, border_thickness, hkl, eng_hkl_dir)

    def _relax(self, relax_dir, size):
        """
        Set up the relaxation simulation for a given supercell size.
        """
        with open(os.path.join(self.template_dir, 'submit-triton.sh'), 'r') as f:
            submit_template = f.read()
        submit_file = os.path.join(relax_dir, 'submit-relax.sh')
        with open(submit_file, 'w') as f:
            f.write(submit_template.format(file='in.relax'))
        with open(os.path.join(self.template_dir, 'in.relax'), 'r') as f:
            input_template = f.read()
        input_file = os.path.join(relax_dir, 'in.relax')
        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings='\n'.join(self.ff_settings), mass=self.mass, Temp=self.temp))
        unit_cell = bulk(self.element, self.lattice, a=self.alat, cubic=True)
        super_cell = unit_cell * [size, size, size]
        write(os.path.join(relax_dir, 'data.input'), super_cell, format='lammps-data')
        subprocess.run('sbatch submit-relax.sh', shell=True, check=True, cwd=relax_dir)

    def calculate(self, relax_flag=False, simulation_flag=False, check_flag=False, postprocess_flag=False):
        """
        Run the cascade calculations and optional post-processing.
        """
        self._get_random_angles(self.min_phi, self.max_phi, self.min_theta, self.max_theta, self.num_sampling_points)
        for energy, size in zip(self.energies, self.sizes):
            eng_dir = os.path.join(self.calculation_dir, str(int(energy)))
            self._set_hkl_from_angles(eng_dir=eng_dir)  # Save hkl_list.dat in energy folder
            for idx, _ in enumerate(self.hkl_list):
                eng_hkl_dir = os.path.join(eng_dir, str(idx))
                os.makedirs(eng_hkl_dir, exist_ok=True)
        if relax_flag:
            for energy, size in zip(self.energies, self.sizes):
                self.logger.info(f'------------------Relaxation for energy: {energy} eV, supercell size: {size} --------------------')
                eng_dir = os.path.join(self.calculation_dir, str(int(energy)))
                self._relax(eng_dir, size)
        if simulation_flag:
            for energy, border_thickness, size, radius_frac in zip(self.energies, self.border_thicknesses, self.sizes, self.radius_fracs):
                self.logger.info(f'------------------Cascade simulation for energy: {energy} eV, supercell size: {size} --------------------')
                eng_dir = os.path.join(self.calculation_dir, str(int(energy)))
                velocity = np.sqrt(2 * energy / (self.mass * AMU_TO_KG * JOULE_TO_EV)) / (ANGSTROM_TO_METER/PS_TO_S)
                for idx, hkl in enumerate(self.hkl_list):
                    _, pka_id = self._get_pka_id_sphere(os.path.join(eng_dir, 'relax.out'), hkl, radius_frac)
                    eng_hkl_dir = os.path.join(eng_dir, str(idx))
                    self._setup(velocity, pka_id, border_thickness, hkl, eng_hkl_dir)
                    subprocess.run('sbatch submit.sh', shell=True, check=True, cwd=eng_hkl_dir)
        if check_flag:
            self.logger.info(f'--------------------------------------Check supercell size ----------------------------------------')
            self._check_cell_size()
        if postprocess_flag:
            self.logger.info(f'--------------------------------------Post-processing simulations ----------------------------------------')
            self.postProcess()

    def _check_cell_size(self):
        for energy, size in zip(self.energies, self.sizes):
            failed_cnt = 0
            eng_dir = os.path.join(self.calculation_dir, str(int(energy)))
            for idx, hkl in enumerate(self.hkl_list):
                eng_hkl_dir = os.path.join(eng_dir, str(idx))
                lammps_output = os.path.join(eng_hkl_dir, 'data.txt')
                if not os.path.exists(lammps_output):
                    self.logger.info(f"LAMMPS output file not found for energy {energy} eV, hkl {hkl}.")
                    continue
                data = np.unique(np.loadtxt(lammps_output, skiprows=1), axis=0)
                Step, Time, Epot, Ekin, Etot, Temp, max_ek, max_ek_border = data.T
                if Time[-1] < 30:
                    self.logger.warning(f"Simulation run less than 30 ps for energy {energy} eV, hkl: {idx}. Final time: {Time[-1]} ps.")
                    failed_cnt += 1
            self.logger.info(f"Energy: {energy} eV, Failed cases: {failed_cnt} for energy {energy} eV with size {size}.")
