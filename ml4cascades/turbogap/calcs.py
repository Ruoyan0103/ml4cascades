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

    # -------------------------------------- Atomic thermalize Steps --------------------------------------#
    def thermalize_atomic(self, input_config: dict) -> str:
        supercell_size = input_config["supercell_size"]
        equ_md_steps = input_config["equ_md_steps"]
        temp = input_config["temp"]
        taut = input_config["taut"]

        thermalize_dir = os.path.join(self.calculation_dir, 'thermalize_atomic', f'{supercell_size[0]}-{supercell_size[1]}-{supercell_size[2]}')
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

    # -------------------------------------- Electronic thermalize Steps --------------------------------------#
    def thermalize_electronic(self, input_config: dict):  
        supercell_size = input_config["supercell_size"]
        equ_md_steps = input_config["equ_md_steps"]
        temp = input_config["temp"]
        xlow = input_config["xlow"]
        xhigh = input_config["xhigh"]
        ylow = input_config["ylow"]
        yhigh = input_config["yhigh"]
        zlow = input_config["zlow"]
        zhigh = input_config["zhigh"]
        eph_C_e = input_config["eph_C_e"]
        eph_kappa_e = input_config["eph_kappa_e"]
        eph_tout_file = input_config["eph_tout_file"]
        atomsfile = input_config.get("atomsfile", None)

        thermalize_dir = os.path.join(self.calculation_dir, 'thermalize_electronic', f'{supercell_size[0]}-{supercell_size[1]}-{supercell_size[2]}')
        os.makedirs(thermalize_dir, exist_ok=True)
        shutil.copy(os.path.join(self.template_dir, 'submit-thermo.sh'), thermalize_dir)
        shutil.copytree(self.tgap_files, os.path.join(thermalize_dir, 'gap_files'), dirs_exist_ok=True)
        with open(os.path.join(self.template_dir, 'input-thermalize-electronic'), 'r') as f:
            input_template = f.read()
        input_file = os.path.join(thermalize_dir, 'input')
        beta_file = os.path.join(self.template_dir, 'betafile', 'beta_Ge.dat')
        if atomsfile is None:
            thermalized_atoms = read(os.path.join(self.calculation_dir, 'thermalize_atomic', f'{supercell_size[0]}-{supercell_size[1]}-{supercell_size[2]}', 'trajectory_out.xyz'), format='extxyz', index=-1)
            thermalized_atoms_file = os.path.join(self.calculation_dir, 'thermalize_atomic', f'{supercell_size[0]}-{supercell_size[1]}-{supercell_size[2]}', 'thermalized_atoms.xyz')
            write(thermalized_atoms_file, thermalized_atoms, format='extxyz')
            atomsfile = thermalized_atoms_file
        with open(input_file, 'w') as f:
            f.write(input_template.format(atomsfile=atomsfile, ff_settings=self.potential.ff_settings, 
                                          num_species=len(self.bi.element), element=' '.join(self.bi.element), 
                                          mass=self.bi.mass, equ_md_steps=equ_md_steps, temp=temp, beta_file=beta_file, 
                                          xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh, zlow=zlow, zhigh=zhigh,
                                          eph_C_e=eph_C_e, eph_kappa_e=eph_kappa_e, eph_tout_file=eph_tout_file)) 
        subprocess.run('sbatch submit-thermo.sh', shell=True, check=True, cwd=thermalize_dir)

    # -------------------------------------- PKA direction --------------------------------------#
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
    
    # -------------------------------------- Cascade simulation --------------------------------------#
    def run_cascade(self,
                    num_PKA_directions: int,
                    radius_frac: float,
                    PKA_kin_eng: float,
                    input_config: dict):
        supercell_size = input_config["supercell_size"]
        cascade_steps = input_config["cascade_steps"]
        temp = input_config["temp"]
        xlow = input_config["xlow"]
        xhigh = input_config["xhigh"]
        ylow = input_config["ylow"]
        yhigh = input_config["yhigh"]
        zlow = input_config["zlow"]
        zhigh = input_config["zhigh"]
        gsx = input_config["gsx"]
        gsy = input_config["gsy"]
        gsz = input_config["gsz"]
        eph_C_e = input_config["eph_C_e"]
        eph_kappa_e = input_config["eph_kappa_e"]
        eph_tout_file = input_config["eph_tout_file"]
        PKA_kin_eng_dir = os.path.join(self.calculation_dir, 'cascade', f'PKA_{int(PKA_kin_eng)}eV')
        os.makedirs(PKA_kin_eng_dir, exist_ok=True)

        thermalized_struct = read(os.path.join(self.calculation_dir, 'thermalize_electronic', f'{supercell_size[0]}-{supercell_size[1]}-{supercell_size[2]}', 'trajectory_out.xyz'), format='extxyz', index=-1)
        dirs = self._get_PKA_directions(num_PKA_directions)
        atom_positions = thermalized_struct.get_positions()
        atom_velocities = thermalized_struct.get_array('velocities')
        cell_lengths = thermalized_struct.cell.lengths()
        radius = 0.5 * min(cell_lengths) * radius_frac
        center = thermalized_struct.get_center_of_mass()
        self.logger.info(f'#--------------------------------- Ekin: {int(PKA_kin_eng)} eV size: {supercell_size[0]}*{supercell_size[1]}*{supercell_size[2]}---------------------------------#')
        for idx, xyz in enumerate(dirs):
            # PKA id
            target_position = xyz * radius + center # shape (3,)
            dists = np.linalg.norm(atom_positions - target_position, axis=1)
            PKA_id = np.argmin(dists)

            # PKA velocity
            velocity_value = np.sqrt(2 * PKA_kin_eng  / (self.bi.mass * AMU_TO_KG * JOULE_TO_EV)) / (ANGSTROM_TO_METER/FS_TO_S)
            velocity = velocity_value * -xyz        # shape (3,)

            # set PKA
            cascade_struct = copy.deepcopy(thermalized_struct)
            new_velocities = copy.deepcopy(atom_velocities)
            new_velocities[PKA_id] = velocity
            self.logger.info(f'PKA ID: {PKA_id}, direction: {xyz}, velocity: {velocity} ang/fs')
            cascade_struct.set_array('velocities', new_velocities)
            cascade_dir = os.path.join(PKA_kin_eng_dir, f'{idx+1}')
            os.makedirs(cascade_dir, exist_ok=True)
            cascade_file = os.path.join(cascade_dir, 'cascade_initial.xyz')
            write(cascade_file, cascade_struct, format='extxyz')

            # prepare input file
            shutil.copytree(self.tgap_files, os.path.join(cascade_dir, 'gap_files'), dirs_exist_ok=True)
            with open(os.path.join(self.template_dir, 'submit-cascade.sh'), 'r') as f:
                submit_template = f.read()
            submit_file = os.path.join(cascade_dir, 'submit-cascade.sh')
            with open(submit_file, 'w') as f:
                f.write(submit_template.format(num=idx+1))
            beta_file = os.path.join(self.template_dir, 'betafile', 'beta_Ge.dat')
            with open(os.path.join(self.template_dir, 'input-cascade'), 'r') as f:
                input_template = f.read()
            input_file = os.path.join(cascade_dir, 'input')
            with open(input_file, 'w') as f:
                f.write(input_template.format(atomsfile=cascade_file, ff_settings=self.potential.ff_settings, 
                                              num_species=len(self.bi.element), element=' '.join(self.bi.element), 
                                              mass=self.bi.mass, cascade_steps=cascade_steps, temp=temp, beta_file=beta_file,
                                              xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh, zlow=zlow, zhigh=zhigh,
                                              gsx=gsx, gsy=gsy, gsz=gsz,
                                              eph_C_e=eph_C_e, eph_kappa_e=eph_kappa_e, eph_tout_file=eph_tout_file))
            # subprocess.run('sbatch submit-cascade.sh', shell=True, check=True, cwd=cascade_dir)





        


   







