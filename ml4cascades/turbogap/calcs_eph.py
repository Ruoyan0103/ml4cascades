from ml4cascades.turbogap import TurboGAPCalculator, TCeKappa
from ml4cascades.utils import BasicInput
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
        eph_parameter_from_file: int, # 1: read from parameters file, 0: do not read
        task_name='pka'
    ):
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
        phi = np.random.uniform(self.min_phi, self.max_phi, self.num_sampling_direcs*1.1)           
        costheta = np.random.uniform(np.cos(self.min_theta), np.cos(self.max_theta), self.num_sampling_direcs*1.1) 
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
            f.write(submit_template.format(job_name=f'rlx_{size}{size}{size}'))
        with open(os.path.join(self.template_dir, 'input-relax'), 'r') as f:
            input_template = f.read()
        input_file = os.path.join(relax_dir, 'input')
        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings=self.bi.ff_settings, num_species=len(self.bi.element),
                                          element=self.bi.element, mass=self.bi.mass, equilibration_steps=self.equ_md_steps,
                                          Temp=self.temp))
        unit_cell = bulk(self.element, self.lattice, a=self.alat, cubic=True)
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
        scaling_factor = 3
        thickness_x, thickness_y, thickness_z = (a*scaling_factor-a)/2, (b*scaling_factor-b)/2, (c*scaling_factor-c)/2
        gsx, gsy, gsz = a*scaling_factor/voxel_size, b*scaling_factor/voxel_size, c*scaling_factor/voxel_size
        parameters_in_file = os.path.join(self.template_dir, 'K_Ge.dat')
        parameters_out_file = os.path.join(self.eng_hkl_dir, 'Te-dependent_e-parameters.txt')
        tin_file = os.path.join(self.eng_dir, 'tin.dat')
        tout_file = os.path.join(self.eng_hkl_dir, 'tout.dat')
        if self.eph_parameter_from_file == 1:
            tcekappa = TCeKappa(parameters_in_file=parameters_in_file,
                    parameters_out_file=parameters_out_file,
                    tin_file=tin_file,
                    grids=[int(gsx), int(gsy), int(gsz)],
                    boxsize=[a*scaling_factor, b*scaling_factor, c*scaling_factor],
                    T_e=self.temp,
                    C_e=1, # will read from file
                    K_e=1,
                    read_from_param_file=1)
        elif self.eph_parameter_from_file == 0:
            tcekappa = TCeKappa(parameters_in_file=parameters_in_file,
                                parameters_out_file=parameters_out_file,
                                tin_file=tin_file,
                                grids=[int(gsx), int(gsy), int(gsz)],
                                boxsize=[a*scaling_factor, b*scaling_factor, c*scaling_factor],
                                T_e=self.temp,
                                C_e=1, # will be reset
                                K_e=1,
                                read_from_param_file=0)
            C_e, K_e = tcekappa._get_Ce_Ke_for_T()
            tcekappa = TCeKappa(parameters_in_file=parameters_in_file,
                                parameters_out_file=parameters_out_file,
                                tin_file=tin_file,
                                grids=[int(gsx), int(gsy), int(gsz)],
                                boxsize=[a*scaling_factor, b*scaling_factor, c*scaling_factor],
                                T_e=self.temp,
                                C_e=C_e,
                                K_e=K_e,
                                read_from_param_file=0)
        # ------------------------ eph setting end ------------------------

        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings=self.ff_settings, 
                                          num_species=self.bi.num_species,
                                          element=self.bi.element, 
                                          mass=self.bi.mass, 
                                          Temp=self.temp, 
                                          cascade_steps=self.cascade_md_steps, 
                                          beta_file=os.path.join(self.template_dir, 'beta.dat'),
                                          xlow=0-thickness_x, xhigh=a+thickness_x,
                                          ylow=0-thickness_y, yhigh=b+thickness_y,
                                          zlow=0-thickness_z, zhigh=c+thickness_z,
                                          eph_tin_file=tin_file, 
                                          eph_tout_file=tout_file))

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
        
    def _setup(self, relaxflag):
        """
        Set up the directories and input files for the LAMMPS simulation.
        """
        self._get_random_angles()
        self._set_hkl_from_angles()
        for energy, size in zip(self.energies, self.sizes):
            eng_dir = os.path.join(self.calculation_dir, str(int(energy)))
            os.makedirs(eng_dir, exist_ok=True)
            shutil.copytree(self.gap_file_folder, os.path.join(eng_dir, 'gap_files'), dirs_exist_ok=True)
            if relaxflag:
                self.logger.info(f'------------------Relaxation for energy: {energy} eV, supercell size: {size} --------------------')
                self._relax(eng_dir, size)

    def calculate(self, relaxflag=False, cascadeflag=False):
        """
        Run the cascade calculations.
        """
        self._setup(relaxflag)
        # each time set one flag True
        if cascadeflag:
            for energy, size, radius_frac, in zip(self.energies, self.sizes, self.radius_fracs):
                self.logger.info(f'------------------Cascade simulation for energy: {energy} eV, supercell size: {size} --------------------')
                eng_dir = os.path.join(self.calculation_dir, str(int(energy)))
                trajectory_file = os.path.join(eng_dir, 'trajectory_out.xyz')
                velocity = np.sqrt(2 * energy  / (self.mass * AMU_TO_KG * JOULE_TO_EV)) / (ANGSTROM_TO_METER/FS_TO_S) 
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
                        f.write(submit_template.format(job_name=f'cas_{energy}_{idx}'))
                    subprocess.run('sbatch submit.sh', shell=True, check=True, cwd=eng_hkl_dir)


    def postProcess(self):
        eng_vac = {}             # {energy: list of number of vacancies}
        eng_inter = {}           # {energy: list of number of intersttials}
        eng_vac_cluster = {}     # {energy: list of number of vacancy clusters}
        eng_inter_cluster = {}   # {energy: list of number of interstitial clusters}
        for energy in self.energies:
            eng_dir = os.path.join(self.calculation_dir, str(int(energy)))
            
            self.logger.info(f'---------------------------------------------------')
            self.logger.info(f'Post-processing for energy: {energy} eV')
            number_of_vacancies = []
            number_of_interstitials = []
            number_of_vacancy_clusters = []
            number_of_interstitial_clusters = []
            for idx, hkl in enumerate(self.hkl_list):
                self.logger.info(f'Processing hkl: {hkl}')
                eng_hkl_dir = os.path.join(eng_dir, str(idx))
                relaxed_file = os.path.join(eng_hkl_dir, 'thermalized.xyz')
                reference_pipeline = import_file(relaxed_file) 
                
                all_pipeline = import_file(os.path.join(eng_hkl_dir, 'trajectory_out.xyz'))
                last_frame = all_pipeline.compute(all_pipeline.source.num_frames-1)
                pipeline = Pipeline(source=StaticSource(data=last_frame))

                # Count vacancies and interstitials
                cnt_vacancies, cnt_inteerstitials = self._countVacAndInter(pipeline, reference_pipeline)
                self.logger.info(f'Vacancies: {cnt_vacancies}, Interstitials: {cnt_inteerstitials}')
                number_of_vacancies.append(cnt_vacancies)
                number_of_interstitials.append(cnt_inteerstitials)
                # Count vacancy clusters
                expression = 'Occupancy == 0'
                vac_clusters, total_line_length, cell_volume, dislocation_lines = self._clustersAndDXA(pipeline, reference_pipeline, expression)
                self.logger.info(f'Vacancy Clusters: {vac_clusters}')
                sum_vac_clusters = sum(vac_clusters.values())
                number_of_vacancy_clusters.append(sum_vac_clusters)
                self.logger.info(f'Total dislocation line length: {total_line_length}, Cell volume: {cell_volume}, Dislocation density : {total_line_length/cell_volume}')
                self.logger.info(f'Number of dislocation lines: {len(dislocation_lines)}')
                for line in dislocation_lines:
                    self.logger.info(f'Dislocation line {line.id}: Length = {line.length}, Burgers vector = {line.true_burgers_vector}')
                # Count interstitial clusters
                expression = 'Occupancy > 1'
                inter_clusters, total_line_length, cell_volume, dislocation_lines = self._clustersAndDXA(pipeline, reference_pipeline, expression)
                self.logger.info(f'Interstitial Clusters: {inter_clusters}')
                sum_inter_clusters = sum(inter_clusters.values())
                number_of_interstitial_clusters.append(sum_inter_clusters)
                self.logger.info(f'Total dislocation line length: {total_line_length}, Cell volume: {cell_volume}, Dislocation density : {total_line_length/cell_volume}')
                self.logger.info(f'Number of dislocation lines: {len(dislocation_lines)}')
                for line in dislocation_lines:
                    self.logger.info(f'Dislocation line {line.id}: Length = {line.length}, Burgers vector = {line.true_burgers_vector}')

            eng_vac[energy] = number_of_vacancies
            eng_inter[energy] = number_of_interstitials
            eng_vac_cluster[energy] = number_of_vacancy_clusters
            eng_inter_cluster[energy] = number_of_interstitial_clusters

        def write_txt(vacancy_dict, interstitial_dict, file_name1, file_name2):
            eng_meanVac_stdVac = {
                energy: (np.mean(num_vac), np.std(num_vac))
                for energy, num_vac in vacancy_dict.items()
            }
            eng_meanInt_stdInt = {
                energy: (np.mean(num_int), np.std(num_int))
                for energy, num_int in interstitial_dict.items()
            }
            with open(os.path.join(self.calculation_dir, file_name1), 'w') as f:
                # Write header (title)
                f.write("# Energy (eV)    Mean value    Std Dev value\n")
                f.write("# ---------------------------------------------\n")
                
                for energy, (mean_vac, std_vac) in eng_meanVac_stdVac.items():
                    # Align values with fixed-width formatting
                    f.write(f"{energy:>10.1f} {mean_vac:>15} {std_vac:>15}\n")
            with open(os.path.join(self.calculation_dir, file_name2), 'w') as f:
                # Write header (title)
                f.write("# Energy (eV)    Mean value    Std Dev value\n")
                f.write("# ----------------------------------------------------\n")
                
                for energy, (mean_int, std_int) in eng_meanInt_stdInt.items():
                    # Align values with fixed-width formatting
                    f.write(f"{energy:>10.1f} {mean_int:>15} {std_int:>15}\n")
            
        write_txt(eng_vac, eng_inter, 'eng_vac.txt', 'eng_inter.txt')
        write_txt(eng_vac_cluster, eng_inter_cluster, 'eng_vac_cluster.txt', 'eng_inter_cluster.txt')


    def _countVacAndInter(self, pipeline, reference_pipeline):
        # wigner_seitz analysis
        wsam = WignerSeitzAnalysisModifier(per_type_occupancies=True, output_displaced=False)
        wsam.reference = reference_pipeline.source
        pipeline.modifiers.append(wsam)
        data = pipeline.compute(0)
        cnt_vacancies = 0
        cnt_interstitials = 0
        for occupancy, position in zip(data.particles['Occupancy'], data.particles['Position']):
            if occupancy == 0:
                cnt_vacancies += 1
            if occupancy > 1:
                cnt_interstitials += 1
        pipeline.modifiers.remove(wsam)        # otherwise it influence consecutive analysis
        return cnt_vacancies, cnt_interstitials

                
    def _clustersAndDXA(self, pipeline, reference_pipeline, expression):
        # wigner_seitz analysis
        wsam = WignerSeitzAnalysisModifier(per_type_occupancies=True, output_displaced=False)
        wsam.reference = reference_pipeline.source
        pipeline.modifiers.append(wsam)
        # selection modifier to select vacancies
        sel = ExpressionSelectionModifier(expression=expression)
        pipeline.modifiers.append(sel)
        # cluster analysis
        cls = ClusterAnalysisModifier(cutoff=1, sort_by_size=True, only_selected=True)
        pipeline.modifiers.append(cls)
        # dislocation analysis
        dxa = DislocationAnalysisModifier(only_selected=True)
        dxa.input_crystal_structure = DislocationAnalysisModifier.Lattice.CubicDiamond
        pipeline.modifiers.append(dxa)

        data = pipeline.compute(0)
        cluster_table = data.tables['clusters']
        cluster_sizes = cluster_table['Cluster Size']
        total_line_length = data.attributes['DislocationAnalysis.total_line_length']
        cell_volume = data.attributes['DislocationAnalysis.cell_volume']
        
        pipeline.modifiers.remove(wsam)
        pipeline.modifiers.remove(sel)
        pipeline.modifiers.remove(cls)
        pipeline.modifiers.remove(dxa)
        count_dict = Counter(cluster_sizes)  # {cluster size: count}
        return count_dict, total_line_length, cell_volume, data.dislocations.lines       
    

        

        
                



    
                






   
    



