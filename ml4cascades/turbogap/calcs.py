from ml4cascades.turbogap import TurboGAPCalculator
import os, subprocess, math, shutil, time
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


class CascadeCalculator(TurboGAPCalculator):
    """ 
    Threshold displacement energy calculator.
    """          
    def __init__(self, potential, num_species, mass, element, lattice, alat, sizes, thicknesses, radius_fracs, temperature, 
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
        self.num_points = num_sampling_points
        self.equilibration_steps = equilibration_steps
        self.cascade_steps = cascade_steps
        self.gap_file_folder = gap_file_folder
        self.thicknesses = thicknesses 
        self.radius_fracs = radius_fracs 
        self.angle_set = set()
        self.hkl_list = []
        self.min_phi = 0
        self.max_phi = 54.7
        self.min_theta = 0
        self.max_theta = 45
        self.num_directions = 30


    def _get_random_angles(self):
        """
        Generate random angles in spherical coordinates.
        """
        np.random.seed(42)  
        phi = np.random.uniform(self.min_phi, self.max_phi, self.num_points)           # azimuthal angle (φ)
        costheta = np.random.uniform(np.cos(self.min_theta), np.cos(self.max_theta), self.num_points) 
        theta = np.arccos(costheta)                                                    # polar angle (θ)
        self.angle_set = set(zip(phi, theta))                                          # Store unique angles

    
    def _set_hkl_from_angles(self, threshold_deg=15):
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
            f.write(submit_template.format(job_name=f'rlx_{size}{size}{size}'))
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
        subprocess.run('sbatch submit.sh', shell=True, check=True, cwd=relax_dir)


    def _get_pka_id_center(self, trajectory_file):
        relaxed_struct = read(trajectory_file, format='extxyz', index=-1)
        center = relaxed_struct.get_center_of_mass()
        pka_id = self._get_pka_id(relaxed_struct, center)
        return relaxed_struct, pka_id
    

    def _get_pka_id_sphere(self, trajectory_file, hkl, radius_frac):
        relaxed_struct = read(trajectory_file, format='extxyz', index=-1)         
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
        pka_id = closest_idx + 1      # LAMMPS IDs start from 1
        return pka_id


    def _setup_helper(self, velocity, relaxed_struct, thickness, pka_id, hkl, eng_hkl_dir):
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
        a, b, c, _, _, _ = relaxed_struct.cell.cellpar()
        with open(input_file, 'w') as f:
            f.write(input_template.format(ff_settings=ff_settings, num_species=self.num_species,
                                          element=self.element, mass=self.mass, Temp=self.temp, cascade_steps=self.cascade_steps,
                                          stopping_file=os.path.join(self.template_dir, 'Ge_Ge_elstop.txt'), xlow=thickness, xhigh=a-thickness,
                                          ylow=thickness, yhigh=a-thickness, zlow=thickness, zhigh=a-thickness))
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
        # time.sleep(120)
        if cascadeflag:
            for energy, thickness, size, radius_frac, in zip(self.energies, self.thicknesses, self.sizes, self.radius_fracs):
                self.logger.info(f'------------------Cascade simulation for energy: {energy} eV, supercell size: {size} --------------------')
                eng_dir = os.path.join(self.calculation_dir, str(int(energy)))
                trajectory_file = os.path.join(eng_dir, 'trajectory_out.xyz')
                velocity = np.sqrt(2 * energy  / (self.mass * AMU_TO_KG * JOULE_TO_EV)) / (ANGSTROM_TO_METER/FS_TO_S) 
                for idx, hkl in enumerate(self.hkl_list):
                    eng_hkl_dir = os.path.join(eng_dir, str(idx))
                    relaxed_struct, pka_id = self._get_pka_id_sphere(trajectory_file, hkl, radius_frac)
                    os.makedirs(eng_hkl_dir, exist_ok=True)
                    shutil.copytree(self.gap_file_folder, os.path.join(eng_hkl_dir, 'gap_files'), dirs_exist_ok=True)
                    self._setup_helper(velocity, relaxed_struct, thickness, pka_id, hkl, eng_hkl_dir)
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
    

        

        
                



    
                






   
    



