from typing import Tuple
import os, subprocess, math, shutil, time
from ase import Atoms
from ase.build import bulk 
from ase.io import read, write
import numpy as np
from ovito.io import import_file
from ovito.modifiers import WignerSeitzAnalysisModifier, ClusterAnalysisModifier, ExpressionSelectionModifier, DislocationAnalysisModifier
from ovito.pipeline import StaticSource, Pipeline
from collections import Counter
from ovito.data import DislocationNetwork
from .calcs_base import TurboGAPCalculator
from .utils import TCeKappa
from ml4cascades.utils import BasicInput
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

AMU_TO_KG = 1.66053906660E-27 # Atomic mass unit to kg conversion factor
JOULE_TO_EV = 6.241509074E18  # Joule to eV conversion factor
ANGSTROM_TO_METER = 1E-10     # Angstroms/picosecond to meters/second conversion factor
FS_TO_S = 1E-15               # Picoseconds to seconds conversion factor

module_dir = os.path.dirname(__file__)
result_dir = os.path.join(module_dir, 'results')
log_dir = os.path.join(module_dir, 'logs')

class CascadeCalculatorEPH(TurboGAPCalculator): 
    def __init__(
        self, 
        basicinput: BasicInput, 
        sizes: list[int], 
        radius_fracs: list[float], 
        temperature: float, 
        energies: list[float], 
        num_sampling_direcs: int,
        equ_md_steps: int, 
        cascade_md_steps: int, 
        gap_file_folder: str,
        eph_parameter_from_file: int=0, # 1: read from parameters file, 0: do not read
        task_name='pka'
    ):
        super().__init__(task_name, basicinput.potential)
        self.bi = basicinput
        self.sizes = sizes
        self.radius_fracs = radius_fracs
        self.temp = temperature
        self.energies = energies
        self.num_sampling_direcs = num_sampling_direcs
        self.equ_md_steps = equ_md_steps
        self.cascade_md_steps = cascade_md_steps
        self.gap_file_folder = gap_file_folder
        self.eph_parameter_from_file = eph_parameter_from_file
        self.task_name = task_name
        self.angle_set = set()
        self.hkl_list = []

        self.min_phi = 0
        self.max_phi = 2*np.pi
        self.min_theta = 0
        self.max_theta = np.pi

    def _get_random_angles(self):
        np.random.seed(42)  
        phi = np.random.uniform(self.min_phi, self.max_phi, int(self.num_sampling_direcs*1.1))           
        costheta = np.random.uniform(np.cos(self.min_theta), np.cos(self.max_theta), int(self.num_sampling_direcs*1.1)) 
        theta = np.arccos(costheta)                                                    
        self.angle_set = set(zip(phi, theta))                                    

    def _set_hkl_from_angles(self):
        '''
        self.num_sampling_direcs * 1.1
        if <= 10 % cases failed, then supercell size and sampling directions are both satisfied
        otherwise, the supercell size needed to be increased
        '''
        for angle in self.angle_set:
            if len(self.hkl_list) >= self.num_sampling_direcs * 1.1: 
                break
            phi, theta = angle
            h = np.sin(theta) * np.cos(phi)
            k = np.sin(theta) * np.sin(phi)
            l = np.cos(theta)
            hkl = np.array((h, k, l))
            self.hkl_list.append(hkl)
        hkl_file = os.path.join(self.calculation_dir, 'hkl_list.dat')
        with open(hkl_file, 'w') as f:  
            np.savetxt(f, np.array(self.hkl_list), 
                       fmt='%.8f',      
                       delimiter=' ',   
                       header='h     k     l') 

    def _relax(
        self, 
        relax_dir: str, 
        size: int
    ): 
        with open(os.path.join(self.template_dir, 'submit-triton.sh'), 'r') as f:
            submit_template = f.read()
        submit_file = os.path.join(relax_dir, 'submit.sh')
        with open(submit_file, 'w') as f:
            f.write(submit_template.format(job_name=f'r_{size}{size}{size}'))
        with open(os.path.join(self.template_dir, 'input-relax'), 'r') as f:
            input_template = f.read()
        input_file = os.path.join(relax_dir, 'input')
        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings=self.bi.potential.ff_settings, num_species=len(self.bi.element),
                                          element=self.bi.element, mass=self.bi.mass, equilibration_steps=self.equ_md_steps,
                                          Temp=self.temp))
        unit_cell = bulk(self.bi.element, self.bi.lattice, a=self.bi.alat[0], b=self.bi.alat[1], c=self.bi.alat[2], cubic=True)
        super_cell = unit_cell * [size, size, size]
        write(os.path.join(relax_dir, 'data.input'), super_cell, format='extxyz')
        subprocess.run('sbatch submit.sh', shell=True, check=True, cwd=relax_dir)

    def _get_pka_id_center(
        self, 
        trajectory_file: str
    ) -> Tuple[Atoms, int]:  
        relaxed_struct = read(trajectory_file, format='extxyz', index=-1)
        center = relaxed_struct.get_center_of_mass()
        pka_id = self._get_pka_id(relaxed_struct, center)
        return relaxed_struct, pka_id

    def _get_pka_id_sphere(
        self, 
        trajectory_file: str, 
        hkl: np.ndarray, 
        radius_frac: float
    ) -> Tuple[Atoms, int]:
        relaxed_struct = read(trajectory_file, format='extxyz', index=-1)         
        a, b, c, _, _, _ = relaxed_struct.get_cell_lengths_and_angles()
        radius = 0.5 * min(a, b, c) * radius_frac
        position = hkl * radius + relaxed_struct.get_center_of_mass()
        pka_id = self._get_pka_id(relaxed_struct, position)
        return relaxed_struct, pka_id
    
    def _get_pka_id(
        self, 
        relaxed_struct: Atoms, 
        target_position: np.ndarray
    ) -> int:
        positions_list = relaxed_struct.get_positions()
        dist = 1000
        closest_idx = None
        for idx, positions in enumerate(positions_list):
            temp_dist = np.linalg.norm(positions - target_position)
            if temp_dist < dist:
                dist = temp_dist
                closest_idx = idx
        pka_id = closest_idx + 1      # LAMMPS IDs start from 1
        return pka_id

    def _setup_helper(
        self, 
        velocity: float, 
        relaxed_struct: Atoms, 
        pka_id: int, 
        hkl: np.ndarray, 
        eng_hkl_dir: str,
        eng_dir: str
    ):
        with open(os.path.join(self.template_dir, 'input-pka-eph'), 'r') as f:
            input_template = f.read()
        a, b, c, _, _, _ = relaxed_struct.cell.cellpar()
        input_file = os.path.join(eng_hkl_dir, 'input')

        # ------------------------ eph setting start ------------------------
        beta_file = os.path.join(self.template_dir, 'beta.dat')
        voxel_size = 25 # Å
        scaling_factor = 30
        thickness_x, thickness_y, thickness_z = (a*scaling_factor-a)/2, (b*scaling_factor-b)/2, (c*scaling_factor-c)/2
        xlow = 0-thickness_x
        xhigh = a+thickness_x
        ylow = 0-thickness_y
        yhigh = b+thickness_y
        zlow = 0-thickness_z
        zhigh = c+thickness_z
        gsx, gsy, gsz = a*scaling_factor/voxel_size, b*scaling_factor/voxel_size, c*scaling_factor/voxel_size
        parameters_in_file = os.path.join(self.template_dir, 'K_Ge.dat')
        parameters_out_file = os.path.join(eng_hkl_dir, 'Te-dependent_e-parameters.txt')
        tin_file = os.path.join(eng_dir, 'tin.dat')
        tout_file = 'tout.dat'
        # ---------- from tin file (and 'Te-dependent_e-parameters.txt') ------------
        if self.eph_parameter_from_file == 1:
            tcekappa = TCeKappa(parameters_in_file=parameters_in_file,
                    parameters_out_file=parameters_out_file,
                    tin_file=tin_file,
                    grids=[int(gsx), int(gsy), int(gsz)],
                    boxsize=[xlow, xhigh, ylow, yhigh, zlow, zhigh],
                    T_e=self.temp,
                    C_e=1, # will read from file
                    K_e=1,
                    read_from_param_file=1)
            tcekappa.write_tinfile_turbogap()
            C_e, K_e = 1, 1 # just for placeholder
            # tcekappa.write_tinfile_lammps()
        # ---------------------------from command line ---------------------------------
        elif self.eph_parameter_from_file == 0:
            tcekappa = TCeKappa(parameters_in_file=parameters_in_file,
                                parameters_out_file=parameters_out_file,
                                tin_file=tin_file,
                                grids=[int(gsx), int(gsy), int(gsz)],
                                boxsize=[xlow, xhigh, ylow, yhigh, zlow, zhigh],
                                T_e=self.temp,
                                C_e=1, # will be reset
                                K_e=1,
                                read_from_param_file=0)
            C_e, K_e = tcekappa._get_Ce_Ke_for_T()
            tcekappa = TCeKappa(parameters_in_file=parameters_in_file,
                                parameters_out_file=parameters_out_file,
                                tin_file=tin_file,
                                grids=[int(gsx), int(gsy), int(gsz)],
                                boxsize=[xlow, xhigh, ylow, yhigh, zlow, zhigh],
                                T_e=self.temp,
                                C_e=C_e,
                                K_e=K_e,
                                read_from_param_file=0)
        # ------------------------ eph setting end ------------------------

        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings=self.bi.potential.ff_settings, 
                                          element=self.bi.element, 
                                          mass=self.bi.mass, 
                                          Temp=self.temp, 
                                          cascade_steps=self.cascade_md_steps, 
                                          beta_file=os.path.join(self.template_dir, 'betafile/beta.dat'),
                                          xlow=xlow, xhigh=xhigh,
                                          ylow=ylow, yhigh=yhigh,
                                          zlow=zlow, zhigh=zhigh,
                                          eph_tin_file=tin_file, 
                                          eph_tout_file=tout_file,
                                          eph_C_e=C_e, eph_kappa_e=K_e))

        velocities = relaxed_struct.get_array('velocities')
        self.logger.info(f'PKA ID {pka_id} with old velocities {velocities[pka_id]} ang/fs')
        Vx = velocity * -hkl[0]
        Vy = velocity * -hkl[1]
        Vz = velocity * -hkl[2]
        velocities[pka_id]  = [Vx, Vy, Vz]  
        relaxed_struct.set_array('velocities', velocities)
        self.logger.info(f'PKA ID: {pka_id} with new Velocities: {Vx:.2f}, {Vy:.2f}, {Vz:.2f} ang/fs')
        new_trajectory_file = os.path.join(eng_hkl_dir, 'thermalized.xyz')
        write(new_trajectory_file, relaxed_struct, format='extxyz')
        
    def _setup(
        self, 
        relax_flag: bool=False
    ):
        self._get_random_angles()
        self._set_hkl_from_angles()
        for energy, size in zip(self.energies, self.sizes):
            eng_dir = os.path.join(self.calculation_dir, str(int(energy)))
            os.makedirs(eng_dir, exist_ok=True)
            shutil.copytree(self.gap_file_folder, os.path.join(eng_dir, 'gap_files'), dirs_exist_ok=True)
            if relax_flag:
                self.logger.info(f'------------------RELAXATION for energy: {energy} eV, supercell size: {size} --------------------')
                self._relax(eng_dir, size)

    def calculate(
        self, 
        relax_flag: bool=False, 
        simulation_flag: bool=False,
        postprocess_flag: bool=False
    ):
        if relax_flag and simulation_flag and postprocess_flag:
            raise ValueError("Only one of relax_flag, simulation_flag or postprocess_flag can be True at a time.")
        self._setup(relax_flag)
        if simulation_flag:
            for energy, size, radius_frac, in zip(self.energies, self.sizes, self.radius_fracs):
                self.logger.info(f'------------------SIMULATION for energy: {energy} eV, supercell size: {size} --------------------')
                eng_dir = os.path.join(self.calculation_dir, str(int(energy)))
                trajectory_file = os.path.join(eng_dir, 'trajectory_out.xyz')
                velocity = np.sqrt(2 * energy  / (self.bi.mass * AMU_TO_KG * JOULE_TO_EV)) / (ANGSTROM_TO_METER/FS_TO_S) 
                for idx, hkl in enumerate(self.hkl_list):
                    eng_hkl_dir = os.path.join(eng_dir, str(idx))
                    relaxed_struct, pka_id = self._get_pka_id_sphere(trajectory_file, hkl, radius_frac)
                    os.makedirs(eng_hkl_dir, exist_ok=True)
                    shutil.copytree(self.gap_file_folder, os.path.join(eng_hkl_dir, 'gap_files'), dirs_exist_ok=True)
                    self._setup_helper(velocity, relaxed_struct, pka_id, hkl, eng_hkl_dir, eng_dir)
                    with open(os.path.join(self.template_dir, 'submit-triton.sh'), 'r') as f:
                        submit_template = f.read()
                    submit_file = os.path.join(eng_hkl_dir, 'submit.sh')
                    with open(submit_file, 'w') as f:
                        f.write(submit_template.format(job_name=f'c_{energy}_{idx}'))
                    subprocess.run('sbatch submit.sh', shell=True, check=True, cwd=eng_hkl_dir)
        if postprocess_flag:
            self.postProcess()

    def postProcess(self):
        self.logger.info(f'------------------Postprocessing --------------------')
        vac_data = {}      # dict, {energy: list of number of vacancies}
        inter_data = {}    # dict, {energy: list of number of interstitials}
        defect_data = {}   # dict, {energy: list of number of defects}
        vac_values = {}    # dict, {energy: mean, std of number of vacancies}
        inter_values = {}  # dict, {energy: mean, std of number of interstitials}
        defect_values = {} # dict, {energy: mean, std of number of defects}

        cluster_sizes_lists = [] 
        bins = [(1,2), (3,4), (5,6)]
        for energy in self.energies:
            eng_dir = os.path.join(self.calculation_dir, str(int(energy)))
            hkl_file = os.path.join(eng_dir, 'hkl_list.dat')
            hkl_list = np.loadtxt(hkl_file, skiprows=1)
            if hkl_list.ndim == 1:
                hkl_list = hkl_list.reshape(1, -1)
            relaxed_file = os.path.join(eng_dir, 'trajectory_out.xyz')
            all_pipeline = import_file(relaxed_file)
            last_frame = all_pipeline.compute(all_pipeline.source.num_frames-1)
            reference_pipeline = Pipeline(source=StaticSource(data=last_frame))

            cnt_vacancies_list = []
            cnt_interstitials_list = []
            cnt_defects_list = []
            cluster_sizes_list = []
            for idx, _ in enumerate(hkl_list):
                eng_hkl_dir = os.path.join(eng_dir, str(idx))
                trajectory_file = os.path.join(eng_hkl_dir, 'trajectory_out.xyz')
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

        clusterhist = ClusterHistogram(cluster_sizes_lists, labels=self.energies, bins=bins)
        fig_path = os.path.join(self.calculation_dir, 'cluster_histogram.png')
        clusterhist.plot(fig_path, error_bars=True, palette='pastel')
        
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
    

class ClusterHistogram:
    def __init__(
            self, 
            cluster_sizes_lists: list[list[dict]], 
            labels: list[str], 
            bins: list[tuple[int, int]]
        ):
        self.cluster_sizes_lists = cluster_sizes_lists
        self.labels = labels
        self.bins = bins
        self.df = self._prepare_combined_data()
    
    def _assign_bin(self, size: int) -> str:
        for b in self.bins:
            if b[0] <= size <= b[1]:
                return f'{b[0]}-{b[1]}'
        return f'>{self.bins[-1][1]}'
    
    def _prepare_data(
            self, 
            cluster_sizes_list: list[dict], 
            label: str
        ) -> pd.DataFrame:
        all_sizes = set().union(*[d.keys() for d in cluster_sizes_list])
        avg_counts = {c: np.mean([d.get(c,0) for d in cluster_sizes_list]) for c in all_sizes}
        std_counts = {c: np.std([d.get(c,0) for d in cluster_sizes_list]) for c in all_sizes}
        
        data_for_plot = []
        for size, avg in avg_counts.items():
            bin_label = self._assign_bin(size)
            data_for_plot.append({
                'cluster_bin': bin_label,
                'avg_count': avg,
                'std_count': std_counts[size],
                'label': label
            })
        return pd.DataFrame(data_for_plot)
    
    def _prepare_combined_data(self):
        dfs = []
        for cls_list, label in zip(self.cluster_sizes_lists, self.labels):
            dfs.append(self._prepare_data(cls_list, label))
        return pd.concat(dfs)
    
    def plot(
            self, 
            fig_path: str, 
            error_bars=True, 
            palette='pastel'
        ):
        if error_bars:
            sns.barplot(
                x='cluster_bin',
                y='avg_count',
                hue='label',
                data=self.df,
                palette=palette,
                yerr=self.df['std_count']
            )
        else:
            sns.barplot(
                x='cluster_bin',
                y='avg_count',
                hue='label',
                data=self.df,
                palette=palette
            )
        plt.xlabel('Cluster size bin')
        plt.ylabel('Average count')
        plt.savefig(fig_path)
        

    



   
    



