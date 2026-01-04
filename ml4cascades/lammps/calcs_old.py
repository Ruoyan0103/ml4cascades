from ml4cascades.lammps import LMPStaticCalculator
import os, subprocess, math, shutil, time
import numpy as np
from ase.build import bulk
from ase.io import read, write
from ovito.io import import_file
from ovito.modifiers import WignerSeitzAnalysisModifier, ClusterAnalysisModifier, ExpressionSelectionModifier, DislocationAnalysisModifier
from ovito.pipeline import StaticSource, Pipeline
from ovito.data import DislocationNetwork
from collections import Counter
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

AMU_TO_KG = 1.66053906660E-27  # Atomic mass unit to kg conversion factor
JOULE_TO_EV = 6.241509074E18   # Joule to eV conversion factor
ANGSTROM_TO_METER = 1E-10      # Angstroms/picosecond to meters/second conversion factor
PS_TO_S = 1E-12                # Picoseconds to seconds conversion factor
module_dir = os.path.dirname(__file__)

class CascadeCalculator(LMPStaticCalculator):
    """ 
    Threshold displacement energy calculator.
    """          
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

    def _timeDependentAnalysis(self, pipeline, reference_pipeline, eng_hkl_dir):
        """
        Analyze defects and temperature as a function of time for a given trajectory.
        Args:
            pipeline: OVITO pipeline for the trajectory file.
            reference_pipeline: OVITO pipeline for the reference structure.
            eng_hkl_dir: Directory for the energy and hkl combination.
        Returns:
            times: List of simulation times from data.txt.
            vacancies: List of vacancy counts per frame.
            interstitials: List of interstitial counts per frame.
            temperatures: List of temperatures per frame.
        """
        times = []
        vacancies = []
        interstitials = []
        temperatures = []
        data_file = os.path.join(eng_hkl_dir, 'data.txt')
        
        # Extract time and temperature from data.txt
        if os.path.exists(data_file):
            try:
                # Remove duplicates from data.txt
                data = np.unique(np.loadtxt(data_file, skiprows=1), axis=0)
                Step, Time, Epot, Ekin, Etot, Temp, max_ek, max_ek_border = data.T
                if Time[-1] < 30:
                    self.logger.warning(f"Simulation in {eng_hkl_dir} has final time {Time[-1]} ps < 30 ps. Skipping.")
                    return [], [], [], []
                times = list(Time)
                temperatures = list(Temp)
            except Exception as e:
                self.logger.warning(f"Failed to read data.txt from {eng_hkl_dir}: {e}. Skipping.")
                return [], [], [], []
        else:
            self.logger.warning(f"data.txt not found in {eng_hkl_dir}. Skipping.")
            return [], [], [], []
        
        # Defect analysis for each frame
        wsam = WignerSeitzAnalysisModifier(per_type_occupancies=True, output_displaced=False)
        wsam.reference = reference_pipeline.source
        pipeline.modifiers.append(wsam)
        
        for frame in range(pipeline.source.num_frames):
            data = pipeline.compute(frame)
            cnt_vacancies = 0
            cnt_interstitials = 0
            for occupancy in data.particles['Occupancy']:
                if occupancy == 0:
                    cnt_vacancies += 1
                if occupancy > 1:
                    cnt_interstitials += 1
            vacancies.append(cnt_vacancies)
            interstitials.append(cnt_interstitials)
        
        pipeline.modifiers.remove(wsam)
        
        # Align temperatures with defect frames
        if len(temperatures) != len(vacancies):
            try:
                frame_times = np.linspace(0, times[-1], len(vacancies))  # Interpolate to match frame count
                interp_temps = np.interp(frame_times, times, temperatures)
                temperatures = list(interp_temps)
                times = list(frame_times)
            except Exception as e:
                self.logger.warning(f"Failed to interpolate temperatures in {eng_hkl_dir}: {e}. Using initial temperature {self.temp} K.")
                temperatures = [self.temp] * len(vacancies)
                times = list(np.linspace(0, times[-1], len(vacancies)))
        
        return times, vacancies, interstitials, temperatures
    
    def postProcess(self):
        vac_data = {}      # dict, {energy: list of number of vacancies}
        inter_data = {}    # dict, {energy: list of number of interstitials}
        defect_data = {}   # dict, {energy: list of number of defects}
        vac_values = {}    # dict, {energy: mean, std of number of vacancies}
        inter_values = {}  # dict, {energy: mean, std of number of interstitials}
        defect_values = {} # dict, {energy: mean, std of number of defects}

        cluster_sizes_lists = [] 
        cluster_bins = [(1, 2), (3, 4), (5, 6)]
        for energy in self.energies:
            eng_dir = os.path.join(self.calculation_dir, str(int(energy)))
            hkl_file = os.path.join(self.calculation_dir, 'hkl_list.dat')
            hkl_list = np.loadtxt(hkl_file, skiprows=1)
            if hkl_list.ndim == 1:
                hkl_list = hkl_list.reshape(1, -1)
            relaxed_file = os.path.join(eng_dir, 'relax.out')
            reference_pipeline = import_file(relaxed_file)

            cnt_vacancies_list = []
            cnt_interstitials_list = []
            cnt_defects_list = []
            cluster_sizes_list = []
            for idx, _ in enumerate(hkl_list):
                eng_hkl_dir = os.path.join(eng_dir, str(idx))
                trajectory_file = os.path.join(eng_hkl_dir, 'dump.PKA')
                all_pipeline = import_file(trajectory_file)
                last_frame = all_pipeline.compute(all_pipeline.source.num_frames-1)
                pipeline = Pipeline(source=StaticSource(data=last_frame))
                cnt_vacancies, cnt_interstitials = self._countVacAndInter(pipeline, reference_pipeline)
                cnt_defects = cnt_vacancies + cnt_interstitials
                cnt_vacancies_list.append(cnt_vacancies)
                cnt_interstitials_list.append(cnt_interstitials)
                cnt_defects_list.append(cnt_defects)
                defects_clusters = self._clusters(pipeline, reference_pipeline, cutoff=8.4, expression='Occupancy != 1')
                cluster_sizes_list.append(defects_clusters)
            cluster_sizes_lists.append(cluster_sizes_list)
            
            vac_data[energy] = cnt_vacancies_list
            inter_data[energy] = cnt_interstitials_list
            defect_data[energy] = cnt_defects_list

        clusterhist = ClusterHistogram(cluster_sizes_lists, labels=self.energies, bins=cluster_bins)
        fig_path = os.path.join(self.calculation_dir, 'cluster_histogram.png')
        clusterhist.plot(fig_path, palette='pastel')
        
        vac_values = {energy: (np.mean(vac_data[energy]), np.std(vac_data[energy])) for energy in vac_data}
        inter_values = {energy: (np.mean(inter_data[energy]), np.std(inter_data[energy])) for energy in inter_data}
        defect_values = {energy: (np.mean(defect_data[energy]), np.std(defect_data[energy])) for energy in defect_data}

        self.logger.info(f'{energy}-Postprocessing results: ')
        self.logger.info(f'Vacancies (mean, std): {vac_values}')
        self.logger.info(f'Interstitials (mean, std): {inter_values}')
        self.logger.info(f'Defects (mean, std): {defect_values}')

        return vac_values, inter_values, defect_values

    def _countVacAndInter(self, pipeline, reference_pipeline):
        wsam = WignerSeitzAnalysisModifier(per_type_occupancies=True, output_displaced=False)
        wsam.reference = reference_pipeline.source
        pipeline.modifiers.append(wsam)
        data = pipeline.compute(0)
        cnt_vacancies = 0
        cnt_interstitials = 0
        for occupancy in data.particles['Occupancy']:
            if occupancy == 0:
                cnt_vacancies += 1
            elif occupancy > 1:
                cnt_interstitials += 1
        pipeline.modifiers.remove(wsam)
        return cnt_vacancies, cnt_interstitials

    def _clusters(self, pipeline, reference_pipeline, cutoff=8.4, expression='Occupancy == 0'):
        wsam = WignerSeitzAnalysisModifier(per_type_occupancies=True, output_displaced=False)
        wsam.reference = reference_pipeline.source
        pipeline.modifiers.append(wsam)
        sel = ExpressionSelectionModifier(expression=expression)
        pipeline.modifiers.append(sel)
        cls = ClusterAnalysisModifier(cutoff=cutoff, sort_by_size=True, only_selected=True)
        pipeline.modifiers.append(cls)
        data = pipeline.compute(0)
        cluster_table = data.tables['clusters']
        cluster_sizes = cluster_table['Cluster Size']
        pipeline.modifiers.remove(wsam)
        pipeline.modifiers.remove(sel)
        pipeline.modifiers.remove(cls)
        count_dict = Counter(cluster_sizes)
        return count_dict


    # def postProcess(self):
    #     """
    #     Post-process LAMMPS simulation outputs to count vacancies, interstitials, their clusters,
    #     dislocations, and time-dependent defects and temperature.
    #     Excludes simulations with final time < 30 ps and reads hkl_list.dat from energy folder.
    #     Aligns time-dependent data to a common 0-30 ps time axis.
    #     """
    #     eng_vac = {}             # {energy: list of number of vacancies}
    #     eng_inter = {}           # {energy: list of number of interstitials}
    #     eng_vac_cluster = {}     # {energy: list of number of vacancy clusters}
    #     eng_inter_cluster = {}   # {energy: list of number of interstitial clusters}
    #     eng_vac_time = {}        # {energy: list of (time, mean_vac, std_vac)}
    #     eng_inter_time = {}      # {energy: list of (time, mean_inter, std_inter)}
    #     eng_temp_time = {}       # {energy: list of (time, mean_temp, std_temp)}
        
    #     # bin clusters
    #     def bin_clusters(cluster_dict, value_list):
    #         for size, cnt in cluster_dict.items():
    #             if 1 <= size < 3:
    #                 value_list[0] += cnt
    #             elif 4 <= size < 6:
    #                 value_list[1] += cnt
    #             elif 7 <= size < 9:
    #                 value_list[2] += cnt
    #             else:
    #                 value_list[3] += cnt

    #     for energy in self.energies:
    #         num_correct_cluster = 0
    #         eng_dir = os.path.join(self.calculation_dir, str(int(energy)))
    #         sum_vac_cluster_sizes = [0] * 4
    #         sum_inter_cluster_sizes = [0] * 4
    #         self.logger.info(f'---------------------------------------------------')
    #         self.logger.info(f'Post-processing for energy: {energy} eV')
            
    #         # Load hkl_list from eng_dir/hkl_list.dat
    #         hkl_file = os.path.join(eng_dir, 'hkl_list.dat')
    #         if not os.path.exists(hkl_file):
    #             self.logger.warning(f"hkl_list.dat not found in {eng_dir}. Skipping energy {energy} eV.")
    #             continue
    #         try:
    #             hkl_list = np.loadtxt(hkl_file, skiprows=1)
    #             if hkl_list.ndim == 1:  # Ensure 2D array even for single row
    #                 hkl_list = hkl_list.reshape(1, -1)
    #         except Exception as e:
    #             self.logger.warning(f"Failed to read hkl_list.dat from {eng_dir}: {e}. Skipping energy {energy} eV.")
    #             continue
                
    #         number_of_vacancies = []
    #         number_of_interstitials = []
    #         number_of_vacancy_clusters = []
    #         number_of_interstitial_clusters = []
    #         time_data = []  # List of (times, vacancies, interstitials, temperatures) per hkl
    #         relaxed_file = os.path.join(eng_dir, 'relax.out')
    #         if not os.path.exists(relaxed_file):
    #             self.logger.info(f"Reference file {relaxed_file} not found for energy {energy} eV. Skipping.")
    #             continue
    #         reference_pipeline = import_file(relaxed_file)
            
    #         for idx, hkl in enumerate(hkl_list):
    #             eng_hkl_dir = os.path.join(eng_dir, str(idx))
    #             trajectory_file = os.path.join(eng_hkl_dir, 'dump.PKA')
    #             data_file = os.path.join(eng_hkl_dir, 'data.txt')
                
    #             # Check if simulation reached 30 ps
    #             if os.path.exists(data_file):
    #                 try:
    #                     data = np.unique(np.loadtxt(data_file, skiprows=1), axis=0)
    #                     Time = data[:, 1]  # Second column is Time
    #                     if Time[-1] < 30:
    #                         self.logger.info(f"Skipping hkl {hkl} (index {idx}) for energy {energy} eV: final time {Time[-1]} ps < 30 ps.")
    #                         continue
    #                 except Exception as e:
    #                     self.logger.warning(f"Failed to read data.txt from {eng_hkl_dir}: {e}. Skipping.")
    #                     continue
    #             else:
    #                 self.logger.info(f"data.txt not found for energy {energy} eV, hkl {hkl}. Skipping.")
    #                 continue
                
    #             if not os.path.exists(trajectory_file):
    #                 self.logger.info(f"Trajectory file {trajectory_file} not found for energy {energy} eV, hkl {hkl}. Skipping.")
    #                 continue
    #             all_pipeline = import_file(trajectory_file)
                
    #             # Time-dependent analysis
    #             times, vacancies, interstitials, temperatures = self._timeDependentAnalysis(all_pipeline, reference_pipeline, eng_hkl_dir)
    #             if not times:  # Skip if time-dependent analysis failed (e.g., time < 30 ps)
    #                 continue
    #             time_data.append((times, vacancies, interstitials, temperatures))
                
    #             # Analysis for last frame
    #             last_frame = all_pipeline.compute(all_pipeline.source.num_frames-1)
    #             pipeline = Pipeline(source=StaticSource(data=last_frame))
    #             cnt_vacancies, cnt_interstitials = self._countVacAndInter(pipeline, reference_pipeline)
    #             self.logger.info(f'index {idx}: hkl {hkl} ')
    #             self.logger.info(f'Vacancies: {cnt_vacancies}, Interstitials: {cnt_interstitials}')
    #             if cnt_vacancies != cnt_interstitials:
    #                 self.logger.warning(f"!!!Mismatch in vacancies and interstitials, ignore.")
    #                 # continue
    #             num_correct_cluster += 1
    #             number_of_vacancies.append(cnt_vacancies)
    #             number_of_interstitials.append(cnt_interstitials)
    #             expression = 'Occupancy == 0'
    #             vac_clusters, total_line_length, cell_volume, dislocation_lines = self._clustersAndDXA(pipeline, reference_pipeline, expression)
    #             self.logger.info(f'Vacancy Clusters: {vac_clusters}')
    #             # sum_vac_clusters = sum(vac_clusters.keys())
    #             bin_clusters(vac_clusters, sum_vac_cluster_sizes)
    #             # number_of_vacancy_clusters.append(sum_vac_clusters)
    #             self.logger.info(f'Total dislocation line length: {total_line_length}, Dislocation density: {total_line_length/cell_volume}')
    #             self.logger.info(f'Number of dislocation lines: {len(dislocation_lines)}')
    #             for line in dislocation_lines:
    #                 self.logger.info(f'Dislocation line {line.id}: Length = {line.length}, Burgers vector = {line.true_burgers_vector}')
    #             expression = 'Occupancy > 1'
    #             inter_clusters, total_line_length, cell_volume, dislocation_lines = self._clustersAndDXA(pipeline, reference_pipeline, expression)
    #             self.logger.info(f'Interstitial Clusters: {inter_clusters}')
    #             # sum_inter_clusters = sum(inter_clusters.keys())
    #             bin_clusters(inter_clusters, sum_inter_cluster_sizes)
    #             # number_of_interstitial_clusters.append(sum_inter_clusters)
    #             self.logger.info(f'Total dislocation line length: {total_line_length}, Dislocation density: {total_line_length/cell_volume}')
    #             self.logger.info(f'Number of dislocation lines: {len(dislocation_lines)}')
    #             for line in dislocation_lines:
    #                 self.logger.info(f'Dislocation line {line.id}: Length = {line.length}, Burgers vector = {line.true_burgers_vector}')
            
    #         eng_vac[energy] = number_of_vacancies
    #         eng_inter[energy] = number_of_interstitials
    #         eng_vac_cluster[energy] = sum_vac_cluster_sizes
    #         eng_inter_cluster[energy] = sum_inter_cluster_sizes
    #         self.logger.info(f'correct cluster count: {num_correct_cluster} for energy {energy} eV')
    #         self.logger.info(f'Energy: {energy} eV, number of vac clusters by size: {sum_vac_cluster_sizes}')
    #         self.logger.info(f'Energy: {energy} eV, number of inter clusters by size: {sum_inter_cluster_sizes}')
            
    #         # Aggregate time-dependent data
    #         if time_data:
    #             # Use the maximum number of frames from valid simulations
    #             max_frames = max(len(t) for t, _, _, _ in time_data)
    #             # Create a common time axis from 0 to 30 ps
    #             common_times = np.linspace(0, 30, max_frames)
    #             vac_time = []
    #             inter_time = []
    #             temp_time = []
                
    #             for times, vacancies, interstitials, temperatures in time_data:
    #                 # Interpolate to common time axis
    #                 try:
    #                     interp_vac = np.interp(common_times, times, vacancies, left=vacancies[0], right=vacancies[-1])
    #                     interp_inter = np.interp(common_times, times, interstitials, left=interstitials[0], right=interstitials[-1])
    #                     interp_temp = np.interp(common_times, times, temperatures, left=temperatures[0], right=temperatures[-1])
    #                 except Exception as e:
    #                     self.logger.warning(f"Failed to interpolate data for {eng_hkl_dir}: {e}. Skipping.")
    #                     continue
    #                 vac_time.append(interp_vac)
    #                 inter_time.append(interp_inter)
    #                 temp_time.append(interp_temp)
                
    #             # Convert to numpy arrays for mean and std
    #             vac_time = np.array(vac_time)
    #             inter_time = np.array(inter_time)
    #             temp_time = np.array(temp_time)
                
    #             # Compute mean and std
    #             eng_vac_time[energy] = [(t, np.mean(v), np.std(v)) for t, v in zip(common_times, vac_time.T)]
    #             eng_inter_time[energy] = [(t, np.mean(i), np.std(i)) for t, i in zip(common_times, inter_time.T)]
    #             eng_temp_time[energy] = [(t, np.mean(temp), np.std(temp)) for t, temp in zip(common_times, temp_time.T)]

    #     # Write existing outputs
    #     def write_txt(vacancy_dict, interstitial_dict, file_name1, file_name2):
    #         for energy in vacancy_dict:
    #             output_dir = os.path.join(self.calculation_dir, f"{int(energy)}_postprocessing")
    #             os.makedirs(output_dir, exist_ok=True)
    #             num_vac = vacancy_dict.get(energy, [])
    #             num_int = interstitial_dict.get(energy, [])
    #             if num_vac:
    #                 mean_vac, std_vac = np.mean(num_vac), np.std(num_vac)
    #                 with open(os.path.join(output_dir, file_name1), 'w') as f:
    #                     f.write("# Energy (eV)    Mean value    Std Dev value\n")
    #                     f.write("# ---------------------------------------------\n")
    #                     f.write(f"{energy:>10.1f} {mean_vac:>15} {std_vac:>15}\n")
    #             if num_int:
    #                 mean_int, std_int = np.mean(num_int), np.std(num_int)
    #                 with open(os.path.join(output_dir, file_name2), 'w') as f:
    #                     f.write("# Energy (eV)    Mean value    Std Dev value\n")
    #                     f.write("# ----------------------------------------------------\n")
    #                     f.write(f"{energy:>10.1f} {mean_int:>15} {std_int:>15}\n")

    #     def write_txt_cluster(vacancy_dict, interstitial_dict, file_name1, file_name2, num_correct_cluster):
    #         ranges = ['1 - 3', '4 - 6', '7 - 9', '10+']
    #         for energy in vacancy_dict:
    #             output_dir = os.path.join(self.calculation_dir, f"{int(energy)}_postprocessing")
    #             os.makedirs(output_dir, exist_ok=True)
    #             vac_cluster_num_by_size = vacancy_dict.get(energy)
    #             inter_cluster_num_by_size = interstitial_dict.get(energy)
    #             if vac_cluster_num_by_size:
    #                 for i in range(len(vac_cluster_num_by_size)):
    #                     mean_vac_cluster_num_by_size = vac_cluster_num_by_size[i] / len(self.hkl_list) 
    #                     with open(os.path.join(output_dir, file_name1), 'a') as f:
    #                         if i == 0:
    #                             f.write("# Energy (eV)    Mean value\n")
    #                             f.write("# ---------------------------------------------\n")
    #                             f.write(f"{energy:>10.1f}\n")
    #                         f.write(f"{ranges[i]} {mean_vac_cluster_num_by_size:>15}\n")
    #             if inter_cluster_num_by_size:
    #                 for i in range(len(inter_cluster_num_by_size)):
    #                     mean_inter_cluster_num_by_size = inter_cluster_num_by_size[i] / len(self.hkl_list) 
    #                     with open(os.path.join(output_dir, file_name2), 'a') as f:
    #                         if i == 0:
    #                             f.write("# Energy (eV)    Mean value\n")
    #                             f.write("# ----------------------------------------------------\n")
    #                             f.write(f"{energy:>10.1f}\n")
    #                         f.write(f"{ranges[i]} {mean_inter_cluster_num_by_size:>15}\n")
        
    #     write_txt(eng_vac, eng_inter, 'eng_vac.txt', 'eng_inter.txt')
    #     write_txt_cluster(eng_vac_cluster, eng_inter_cluster, 'eng_vac_cluster.txt', 'eng_inter_cluster.txt', num_correct_cluster)
        
    #     # Write time-dependent outputs
    #     for energy in eng_vac_time:
    #         output_dir = os.path.join(self.calculation_dir, f"{int(energy)}_postprocessing")
    #         os.makedirs(output_dir, exist_ok=True)
    #         with open(os.path.join(output_dir, 'eng_vac_time.txt'), 'w') as f:
    #             f.write("# Time (ps)    Mean Vacancies    Std Dev Vacancies\n")
    #             f.write("# ---------------------------------------------\n")
    #             for t, mean_v, std_v in eng_vac_time[energy]:
    #                 f.write(f"{t:>10.3f} {mean_v:>15} {std_v:>15}\n")
    #         with open(os.path.join(output_dir, 'eng_inter_time.txt'), 'w') as f:
    #             f.write("# Time (ps)    Mean Interstitials    Std Dev Interstitials\n")
    #             f.write("# ---------------------------------------------\n")
    #             for t, mean_i, std_i in eng_inter_time[energy]:
    #                 f.write(f"{t:>10.3f} {mean_i:>15} {std_i:>15}\n")
    #         with open(os.path.join(output_dir, 'eng_temp_time.txt'), 'w') as f:
    #             f.write("# Time (ps)    Mean Temperature    Std Dev Temperature\n")
    #             f.write("# ---------------------------------------------\n")
    #             for t, mean_t, std_t in eng_temp_time[energy]:
    #                 f.write(f"{t:>10.3f} {mean_t:>15} {std_t:>15}\n")

    # def _countVacAndInter(self, pipeline, reference_pipeline):
    #     """
    #     Count vacancies and interstitials using Wigner-Seitz analysis.
    #     """
    #     wsam = WignerSeitzAnalysisModifier(per_type_occupancies=True, output_displaced=False)
    #     wsam.reference = reference_pipeline.source
    #     pipeline.modifiers.append(wsam)
    #     data = pipeline.compute(0)
    #     cnt_vacancies = 0
    #     cnt_interstitials = 0
    #     for occupancy in data.particles['Occupancy']:
    #         if occupancy == 0:
    #             cnt_vacancies += 1
    #         if occupancy > 1:
    #             cnt_interstitials += 1
    #     pipeline.modifiers.remove(wsam)
    #     return cnt_vacancies, cnt_interstitials

    # def _clustersAndDXA(self, pipeline, reference_pipeline, expression):
    #     """
    #     Perform cluster and dislocation analysis for vacancies or interstitials.
    #     """
    #     wsam = WignerSeitzAnalysisModifier(per_type_occupancies=True, output_displaced=False)
    #     wsam.reference = reference_pipeline.source
    #     pipeline.modifiers.append(wsam)
    #     sel = ExpressionSelectionModifier(expression=expression)
    #     pipeline.modifiers.append(sel)
    #     cls = ClusterAnalysisModifier(cutoff=8.4, sort_by_size=True, only_selected=True)
    #     pipeline.modifiers.append(cls)
    #     dxa = DislocationAnalysisModifier(only_selected=True)
    #     dxa.input_crystal_structure = DislocationAnalysisModifier.Lattice.CubicDiamond
    #     pipeline.modifiers.append(dxa)
    #     data = pipeline.compute(0)
    #     cluster_table = data.tables['clusters']
    #     cluster_sizes = cluster_table['Cluster Size']
    #     total_line_length = data.attributes['DislocationAnalysis.total_line_length']
    #     cell_volume = data.attributes['DislocationAnalysis.cell_volume']
    #     pipeline.modifiers.remove(wsam)
    #     pipeline.modifiers.remove(sel)
    #     pipeline.modifiers.remove(cls)
    #     pipeline.modifiers.remove(dxa)
    #     count_dict = Counter(cluster_sizes)
    #     return count_dict, total_line_length, cell_volume, data.dislocations.lines
    
class ClusterHistogram:
    def __init__(self, cluster_sizes_lists, labels, bins):
        self.cluster_sizes_lists = cluster_sizes_lists
        self.labels = [f'{l} eV' for l in labels]
        self.bins = bins
        self.df = self._prepare_combined_data()
    
    def _assign_bin(self, size):
        for b in self.bins:
            if b[0] <= size <= b[1]:
                return f'{b[0]}-{b[1]}'
        return f'>{self.bins[-1][1]}'
    
    def _prepare_data(self, cluster_sizes_list, label):
        all_sizes = set().union(*[d.keys() for d in cluster_sizes_list])
        avg_counts = {c: np.mean([d.get(c,0) for d in cluster_sizes_list]) for c in all_sizes}

        data_for_plot = []
        for size, avg in avg_counts.items():
            bin_label = self._assign_bin(size)
            data_for_plot.append({
                'cluster_bin': bin_label,
                'avg_count': avg,
                'energy': label
            })
        return pd.DataFrame(data_for_plot)
    
    def _prepare_combined_data(self):
        dfs = []
        for cls_list, label in zip(self.cluster_sizes_lists, self.labels):
            dfs.append(self._prepare_data(cls_list, label))
        return pd.concat(dfs)
    
    def plot(self, fig_path, palette='pastel'):
        # assign a color to each energy
        unique_energies = self.df['energy'].unique()
        colors = sns.color_palette(palette, n_colors=len(unique_energies))
        color_dict = dict(zip(unique_energies, colors))
        
        # plot each energy separately
        plt.figure(figsize=(8,6))
        for energy in unique_energies:
            subset = self.df[self.df['energy'] == energy]
            sns.barplot(
                x='cluster_bin',
                y='avg_count',
                data=subset,
                color=color_dict[energy],
                errorbar=None
            )
        
        # create a manual legend
        handles = [plt.Rectangle((0,0),1,1, color=color_dict[e]) for e in unique_energies]
        plt.legend(handles, unique_energies, title=None, fontsize=16)
        
        plt.xlabel('Cluster size', fontsize=16)
        plt.ylabel('Average count', fontsize=16)
        plt.tight_layout()
        plt.savefig(fig_path)
