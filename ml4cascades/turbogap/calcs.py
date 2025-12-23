import os, shutil, subprocess, copy
import numpy as np  
from .calcs_base import TurboGAPCalculator
from ml4cascades.utils import BasicCellInfo
from ml4cascades.potentials import IPotential
from ase.build import bulk
from ase.io import write, read

AMU_TO_KG = 1.66053906660E-27 # Atomic mass unit to kg conversion factor
JOULE_TO_EV = 6.241509074E18  # Joule to eV conversion factor
ANGSTROM_TO_METER = 1E-10     # Angstroms/picosecond to meters/second conversion factor
FS_TO_S = 1E-15               # Picoseconds to seconds conversion factor
module_dir = os.path.dirname(__file__)

class CascadeCalculator(TurboGAPCalculator): 
    def __init__(
        self,
        potential: IPotential,
        basicCellInfo: BasicCellInfo,
        task_name='cascade',
        model_name='EPH'
    ): 
        super().__init__(task_name, model_name)
        self.potential = potential  
        self.tgap_files = potential.get_pot_files_path()
        self.bi = basicCellInfo

    # -------------------------------------- Thermalize Steps --------------------------------------#
    def thermalize_atomic(self, supercell_size: list[int], equ_md_steps: int, temp: float, taut: float) -> str:
        thermalize_dir = os.path.join(self.calculation_dir, 'thermalize_atomic')
        os.makedirs(thermalize_dir, exist_ok=True)
        shutil.copy(os.path.join(self.template_dir, 'submit-thermo.sh'), thermalize_dir)
        shutil.copytree(self.tgap_files, os.path.join(thermalize_dir, 'gap_files'), dirs_exist_ok=True)
        with open(os.path.join(self.template_dir, 'input-thermalize-atomic'), 'r') as f:
            input_template = f.read()
        input_file = os.path.join(thermalize_dir, 'input')
        unit_cell = bulk(''.join(self.bi.element), self.bi.lattice, 
                         a=self.bi.alat[0], b=self.bi.alat[1], c=self.bi.alat[2], cubic=True)
        supercell = unit_cell * supercell_size
        atomsfile = os.path.join(thermalize_dir, 'data.input')
        write(atomsfile, supercell, format='extxyz')
        with open(input_file, 'w') as f:
            f.write(input_template.format(atomsfile=atomsfile, ff_settings=self.potential.ff_settings, 
                                          num_species=len(self.bi.element), element=' '.join(self.bi.element), 
                                          mass=self.bi.mass, equ_md_steps=equ_md_steps, temp=temp, taut=taut)) 
        subprocess.run('sbatch submit-thermo.sh', shell=True, check=True, cwd=thermalize_dir)

    def thermalize_electronic(self, equ_md_steps: int, temp: float, 
                              xlow: float, xhigh: float, ylow: float, yhigh: float, zlow: float, zhigh: float,
                              eph_C_e: float, eph_kappa_e: float, eph_tout_file: str, atomsfile: str=None):
        thermalize_dir = os.path.join(self.calculation_dir, 'thermalize_electronic')
        os.makedirs(thermalize_dir, exist_ok=True)
        shutil.copy(os.path.join(self.template_dir, 'submit-thermo.sh'), thermalize_dir)
        shutil.copytree(self.tgap_files, os.path.join(thermalize_dir, 'gap_files'), dirs_exist_ok=True)
        with open(os.path.join(self.template_dir, 'input-thermalize-electronic'), 'r') as f:
            input_template = f.read()
        input_file = os.path.join(thermalize_dir, 'input')
        beta_file = os.path.join(self.template_dir, 'betafile', 'beta-old.dat')
        if atomsfile is None:
            thermalized_atoms = read(os.path.join(self.calculation_dir, 'thermalize_atomic', 'trajectory_out.xyz'), format='extxyz', index=-1)
            thermalized_atoms_file = os.path.join(self.calculation_dir, 'thermalize_atomic', 'thermalized_atoms.xyz')
            write(thermalized_atoms_file, thermalized_atoms, format='extxyz')
            atomsfile = thermalized_atoms_file
        with open(input_file, 'w') as f:
            f.write(input_template.format(atomsfile=atomsfile, ff_settings=self.potential.ff_settings, 
                                          num_species=len(self.bi.element), element=' '.join(self.bi.element), 
                                          mass=self.bi.mass, equ_md_steps=equ_md_steps, temp=temp, beta_file=beta_file, 
                                          xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh, zlow=zlow, zhigh=zhigh,
                                          eph_C_e=eph_C_e, eph_kappa_e=eph_kappa_e, eph_tout_file=eph_tout_file)) 
        subprocess.run('sbatch submit-thermo.sh', shell=True, check=True, cwd=thermalize_dir)

    # -------------------------------------- Cascade Steps --------------------------------------#
    def _get_PKA_directions(self, num_PKA_directions: int) -> np.ndarray:
        # random directions uniformly distributed over a unit sphere
        rng = np.random.default_rng(42)
        phi = rng.uniform(0.0, 2.0 * np.pi, num_PKA_directions)
        costheta = rng.uniform(-1.0, 1.0, num_PKA_directions)
        theta = np.arccos(costheta)

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        dirs = np.column_stack((x, y, z))  # shape (num_PKA_directions, 3)
        return dirs
    
    def _get_PKA_ids(self, 
                     thermalized_struct: object, 
                     dirs: np.ndarray, 
                     radius_frac: float=0.8) -> list[int]:
        # get PKA atom ids based on the given directions and the thermalized structure
        # radius_frac: fraction of half the minimum lattice constant to define the sphere radius
        atom_positions = thermalized_struct.get_positions()
        radius = 0.5*min(self.bi.alat[0], self.bi.alat[1], self.bi.alat[2]) * radius_frac
        cell = thermalized_struct.get_cell()
        center = 0.5 * (cell[0] + cell[1] + cell[2])
        PKA_ids = []
        for xyz in dirs:
            target_position = xyz * radius + center # shape (3,)
            dists = np.linalg.norm(atom_positions - target_position, axis=1)
            closest_idx = np.argmin(dists)
            PKA_ids.append(closest_idx)
        return PKA_ids
    

    ################################################## some logic issue here ##################################################
    
    def _set_PKA_velocity(self, 
                          thermalized_struct: object, 
                          PKA_ids: list[int], 
                          dirs: np.ndarray,
                          PKA_kin_eng: float) -> float:
        atom_velocities = thermalized_struct.get_array('velocities')
        velocity = np.sqrt(2 * PKA_kin_eng  / (self.mass * AMU_TO_KG * JOULE_TO_EV)) / (ANGSTROM_TO_METER/FS_TO_S) 
        eng_dir = os.path.join(self.calculation_dir, 'cascade', f'PKA_eng_{int(PKA_kin_eng)}eV')
        os.makedirs(eng_dir, exist_ok=True)
        for idx, dir, pka_id in enumerate(zip(dirs, PKA_ids)):
            PKA_thermalized_struct = copy.deepcopy(thermalized_struct)
            Vx = velocity * -dir[0]
            Vy = velocity * -dir[1]
            Vz = velocity * -dir[2]
            atom_velocities[pka_id]  = [Vx, Vy, Vz]  
            self.logger.info(f'PKA ID: {pka_id} with Velocities: {Vx:.2f}, {Vy:.2f}, {Vz:.2f} ang/fs')
            PKA_thermalized_struct.set_array('velocities', atom_velocities)
            eng_run_dir = os.path.join(eng_dir, f'PKA_{idx+1}')
            os.makedirs(eng_run_dir, exist_ok=True)
            PKA_thermalized_file = os.path.join(eng_run_dir, 'PKA_thermalized.xyz')
            write(PKA_thermalized_file, PKA_thermalized_struct, format='extxyz')



        


   







