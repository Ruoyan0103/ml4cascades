import os, shutil, subprocess, copy
import numpy as np
from pandas import cut  
from .calcs_base import LMPSCalculator
from ml4cascades.utils import BasicCellInfo, ParameterGetter
from ml4cascades.potentials import IPotential
from ase.build import bulk
from ase.io import write, read

AMU_TO_KG = 1.66053906660E-27 # Atomic mass unit to kg conversion factor
JOULE_TO_EV = 6.241509074E18  # Joule to eV conversion factor
ANGSTROM_TO_METER = 1E-10     # Angstroms/picosecond to meters/second conversion factor
FS_TO_S = 1E-15               # Picoseconds to seconds conversion factor
PS_TO_S = 1E-12               # Picoseconds to seconds conversion factor
kB = 8.617333262145e-5        # Boltzmann constant, eV/K

module_dir = os.path.dirname(__file__)

class CascadeCalculator(LMPSCalculator): 
    def __init__(
        self,
        potential: IPotential,
        basicCellInfo: BasicCellInfo,
        task_name='cascade',
        model_name='EPH'
    ): 
        super().__init__(task_name, model_name)
        self.potential = potential  
        self.pot_files = potential.get_pot_files_path()
        self.bi = basicCellInfo

    # -------------------------------------- Thermalize Steps --------------------------------------#
    def thermalize(self, input_config: dict):  
        supercell_size = input_config["supercell_size"]
        equ_md_steps = input_config["equ_md_steps"]
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
        atomsfile = input_config.get("atomsfile", None)
        tinfile = input_config.get("tinfile", None)
        thermalize_dir = os.path.join(self.calculation_dir, 'thermalize', f'{supercell_size[0]}-{supercell_size[1]}-{supercell_size[2]}')
        os.makedirs(thermalize_dir, exist_ok=True)
        shutil.copy(os.path.join(self.template_dir, 'submit-thermo.sh'), thermalize_dir)
        shutil.copytree(self.pot_files, thermalize_dir, dirs_exist_ok=True)

        with open(os.path.join(self.template_dir, 'thermalize.lmp'), 'r') as f:
            input_template = f.read()
        input_file = os.path.join(thermalize_dir, 'input.lmp')
        beta_file = os.path.join(self.template_dir, 'betafile', 'beta_Ge.dat')
        if atomsfile is None:
            unit_cell = bulk(''.join(self.bi.element), self.bi.lattice, 
                             a=self.bi.alat[0], b=self.bi.alat[1], c=self.bi.alat[2], cubic=True)
            supercell = unit_cell * supercell_size
            atomsfile = os.path.join(thermalize_dir, 'data.input')
            write(atomsfile, supercell, format='lammps-data')
        if tinfile is None:
            tinfile = os.path.join(thermalize_dir, 'T.in')
            with open(tinfile, 'w') as f:
                f.write('# comment1 \n# comment2 \n# comment3 \n')
                f.write(f'{gsx} {gsy} {gsz} 10\n')
                f.write(f'{xlow} {xhigh} \n')
                f.write(f'{ylow} {yhigh} \n')
                f.write(f'{zlow} {zhigh} \n')
                f.write('NULL\n')
                for iz in range(gsz):
                    for iy in range(gsy):
                        for ix in range(gsx):
                            f.write(f'{ix} {iy} {iz} {temp} 0 1 {eph_C_e} {eph_kappa_e} 1 0\n')
        with open(input_file, 'w') as f:
            f.write(input_template.format(atomsfile=atomsfile, ff_settings=self.potential.ff_settings, tinfile=tinfile, 
                                          eph_C_e=eph_C_e, eph_kappa_e=eph_kappa_e, gsx=gsx, gsy=gsy, gsz=gsz,
                                          mass=self.bi.mass, equ_md_steps=equ_md_steps, temp=temp, beta_file=beta_file)) 
        subprocess.run('sbatch submit-thermo.sh', shell=True, check=True, cwd=thermalize_dir)

    # -------------------------------------- PKA direction --------------------------------------#
    def _get_PKA_directions(self, 
                            rng: np.random.Generator,
                            num_PKA_directions: int) -> np.ndarray:
        # random directions uniformly distributed over a unit sphere
        # rng = np.random.default_rng(42)
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
                    running_directions: list[int], # directions to run, example [1, 2, 3], start from 1 to num_PKA_directions
                    radius_frac: float,
                    PKA_kin_eng: float,
                    input_config: dict,
                    running_dir: str=None) -> str:
        # Common parameters for both modes
        supercell_size = input_config["supercell_size"]
        cascade_steps = input_config["cascade_steps"]
        xlow = input_config["xlow"]
        xhigh = input_config["xhigh"]
        ylow = input_config["ylow"]
        yhigh = input_config["yhigh"]
        zlow = input_config["zlow"]
        zhigh = input_config["zhigh"]
        border_thickness = input_config["border_thickness"]
        temp = input_config["temp"]
        
        # Mode-specific parameters
        if self.model_name == 'EPH':
            gsx = input_config["gsx"]
            gsy = input_config["gsy"]
            gsz = input_config["gsz"]
            eph_C_e = input_config["eph_C_e"]
            eph_kappa_e = input_config["eph_kappa_e"]
            tinfile = input_config.get("tinfile", None)                              
            temperature_dependent = input_config.get("temperature_dependent", False)
            if running_dir is not None:
                PKA_kin_eng_dir = running_dir
            else:
                PKA_kin_eng_dir = os.path.join(self.calculation_dir, 'cascade', f'PKA_{int(PKA_kin_eng)}eV', f'{radius_frac}-{gsx}')
            os.makedirs(PKA_kin_eng_dir, exist_ok=True)
        elif self.model_name == 'STOPPING':
            # gsx = input_config["gsx"]  # needed for directory path
            cutoff_eng = input_config["cutoff_eng"]
            if running_dir is not None:
                PKA_kin_eng_dir = running_dir
            else:
                PKA_kin_eng_dir = os.path.join(self.calculation_dir, 'cascade', f'PKA_{int(PKA_kin_eng)}eV', f'{radius_frac}-{cutoff_eng}')
            os.makedirs(PKA_kin_eng_dir, exist_ok=True)
        elif self.model_name == 'STOPPING-0K':
            # gsx = input_config["gsx"]  # needed for directory path
            cutoff_eng = input_config["cutoff_eng"]
            if running_dir is not None:
                PKA_kin_eng_dir = running_dir
            else:
                PKA_kin_eng_dir = os.path.join(self.calculation_dir, 'cascade', f'PKA_{int(PKA_kin_eng)}eV', f'{radius_frac}-{cutoff_eng}')
            os.makedirs(PKA_kin_eng_dir, exist_ok=True)

        PKA_id_list = []
        atomsfile = os.path.join(self.calculation_dir, 'thermalize', f'{supercell_size[0]}-{supercell_size[1]}-{supercell_size[2]}', 'data.output')
        thermalized_struct = read(atomsfile, format='lammps-data')
        rng = np.random.default_rng(42)
        dirs = self._get_PKA_directions(rng, num_PKA_directions)

        atom_positions = thermalized_struct.get_positions()
        cell_lengths = thermalized_struct.cell.lengths()
        radius = 0.5 * min(cell_lengths) * radius_frac
        center = thermalized_struct.get_center_of_mass()
        # for idx, xyz in enumerate(dirs):
        for running_dir in running_directions:
            if running_dir > num_PKA_directions:
                self.logger.error(f'Running direction {running_dir} exceeds the number of generated PKA directions {num_PKA_directions}.')
                continue
            xyz = dirs[running_dir-1]  # running_dir starts from 1
            target_position = xyz * radius + center
            dists = np.linalg.norm(atom_positions - target_position, axis=1)
            PKA_id = np.argmin(dists)
            PKA_id_list.append(PKA_id)
            velocity_value = np.sqrt(2 * PKA_kin_eng / (self.bi.mass * AMU_TO_KG * JOULE_TO_EV)) / (ANGSTROM_TO_METER/PS_TO_S)
            velocity = velocity_value * -xyz
            
            self.logger.info(f'Ekin: {int(PKA_kin_eng)} eV, size: {supercell_size[0]}*{supercell_size[1]}*{supercell_size[2]}')
            self.logger.info(f'PKA ID: {PKA_id}, direction: {xyz}, velocity: {velocity} ang/ps')

            cascade_dir = os.path.join(PKA_kin_eng_dir, f'{running_dir}')
            os.makedirs(cascade_dir, exist_ok=True)
            shutil.copytree(self.pot_files, cascade_dir, dirs_exist_ok=True)
            
            # Setup and run appropriate cascade mode
            if self.model_name == 'EPH':
                self._run_eph_cascade(cascade_dir, atomsfile, PKA_id, velocity,
                                     gsx, gsy, gsz, xlow, xhigh, ylow, yhigh, 
                                     zlow, zhigh, temp, eph_C_e, eph_kappa_e,
                                     border_thickness, cascade_steps,
                                     tinfile, temperature_dependent)
                with open(os.path.join(self.template_dir, 'submit-cascade.sh'), 'r') as f:
                    submit_template = f.read()

            elif self.model_name == 'STOPPING':
                self._run_stopping_cascade(cascade_dir, atomsfile, PKA_id, velocity,
                                          xlow, xhigh, ylow, yhigh, zlow, zhigh, temp, 
                                          border_thickness, cascade_steps, cutoff_eng)
                with open(os.path.join(self.template_dir, 'submit-cascade-stopping.sh'), 'r') as f:
                    submit_template = f.read()
            elif self.model_name == 'STOPPING-0K':
                atomsfile = os.path.join(self.calculation_dir, 'thermalize', f'{supercell_size[0]}-{supercell_size[1]}-{supercell_size[2]}', 'data.input')
                self._run_stopping_0K_cascade(cascade_dir, atomsfile, PKA_id, velocity,
                                              xlow, xhigh, ylow, yhigh, zlow, zhigh, temp, 
                                              border_thickness, cascade_steps, cutoff_eng)
                with open(os.path.join(self.template_dir, 'submit-cascade-stopping.sh'), 'r') as f:
                    submit_template = f.read()

            submit_file = os.path.join(cascade_dir, 'submit-cascade.sh')
            with open(submit_file, 'w') as f:
                f.write(submit_template.format(num=running_dir))
            
            self.logger.info(f'Prepared cascade simulation direction {running_dir}.')
            # Submit job
            subprocess.run('sbatch submit-cascade.sh', shell=True, check=True, cwd=cascade_dir)
        return PKA_kin_eng_dir
    
    def _write_tinfile(self, tinfile_path: str, gsx: int, gsy: int, gsz: int, 
                       xlow: float, xhigh: float, ylow: float, yhigh: float, 
                       zlow: float, zhigh: float, temp: float, 
                       eph_C_e: float, eph_kappa_e: float, temperature_dependent: bool, 
                       cascade_dir: str) -> None:
        """Write temperature input file for EPH mode."""
        with open(tinfile_path, 'w') as f:
            f.write('# comment1 \n# comment2 \n# comment3 \n')
            f.write(f'{gsx} {gsy} {gsz} 1\n')
            f.write(f'{xlow} {xhigh} \n')
            f.write(f'{ylow} {yhigh} \n')
            f.write(f'{zlow} {zhigh} \n')
            if not temperature_dependent:
                f.write('NULL\n')
                for iz in range(gsz):
                    for iy in range(gsy):
                        for ix in range(gsx):
                            f.write(f'{ix} {iy} {iz} {temp} 0 1 {eph_C_e} {eph_kappa_e} 1 0\n')
            else:
                f.write('Parameters.data\n')
                for iz in range(gsz):
                    for iy in range(gsy):
                        for ix in range(gsx):
                            f.write(f'{ix} {iy} {iz} {temp} 0 1 {eph_C_e} {eph_kappa_e} 1 1\n')
                param_file = os.path.join(cascade_dir, 'Parameters.data')
                param_getter = ParameterGetter()
                param_getter.write_data1(outputfile=param_file)
    
    def _run_eph_cascade(self, cascade_dir: str, atomsfile: str, PKA_id: int, velocity: np.ndarray,
                         gsx: int, gsy: int, gsz: int, xlow: float, xhigh: float, 
                         ylow: float, yhigh: float, zlow: float, zhigh: float,
                         temp: float, eph_C_e: float, eph_kappa_e: float,
                         border_thickness: float, cascade_steps: int,
                         tinfile: str, temperature_dependent: bool) -> None:
        """Prepare and write input file for EPH cascade mode."""
        betafile = os.path.join(self.template_dir, 'betafile', 'beta_Ge.dat')
        with open(os.path.join(self.template_dir, 'cascade-eph.lmp'), 'r') as f:
            input_template = f.read()
        input_file = os.path.join(cascade_dir, 'input.lmp')
        
        if tinfile is None:
            tinfile = os.path.join(cascade_dir, 'T.in')
            self._write_tinfile(tinfile, gsx, gsy, gsz, xlow, xhigh, ylow, yhigh, 
                               zlow, zhigh, temp, eph_C_e, eph_kappa_e, 
                               temperature_dependent, cascade_dir)
            tinfile = 'T.in'
        
        T_out_folder = os.path.join(cascade_dir, 'T_out')
        os.makedirs(T_out_folder, exist_ok=True)
        
        with open(input_file, 'w') as f:
            f.write(input_template.format(
                atomsfile=atomsfile, ff_settings=self.potential.ff_settings, 
                border_thickness=border_thickness, pka_id=PKA_id, 
                v_x=velocity[0], v_y=velocity[1], v_z=velocity[2],
                mass=self.bi.mass, cascade_steps=cascade_steps, temp=temp, 
                betafile=betafile, eph_C_e=eph_C_e, eph_kappa_e=eph_kappa_e,
                xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh, 
                zlow=zlow, zhigh=zhigh, gsx=gsx, gsy=gsy, gsz=gsz, tinfile=tinfile))
    
    def _run_stopping_cascade(self, cascade_dir: str, atomsfile: str, PKA_id: int, 
                              velocity: np.ndarray, xlow: float, xhigh: float,
                              ylow: float, yhigh: float, zlow: float, zhigh: float, temp: float,
                              border_thickness: float, cascade_steps: int, cutoff_eng: float) -> None:
        """Prepare and write input file for stopping cascade mode."""
        with open(os.path.join(self.template_dir, 'cascade-stopping.lmp'), 'r') as f:
            input_template = f.read()
        input_file = os.path.join(cascade_dir, 'input.lmp')
        stoppingfile = os.path.join(self.template_dir, 'stopping', 'Ge_Ge_elstop.txt')
        
        with open(input_file, 'w') as f:
            f.write(input_template.format(
                atomsfile=atomsfile, ff_settings=self.potential.ff_settings, 
                border_thickness=border_thickness, pka_id=PKA_id,
                v_x=velocity[0], v_y=velocity[1], v_z=velocity[2],
                mass=self.bi.mass, cascade_steps=cascade_steps,
                xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh, 
                zlow=zlow, zhigh=zhigh, temp=temp, cutoff_eng=cutoff_eng, 
                stoppingfile=stoppingfile))
            
    def _run_stopping_0K_cascade(self, cascade_dir: str, atomsfile: str, PKA_id: int, 
                                    velocity: np.ndarray, xlow: float, xhigh: float,
                                    ylow: float, yhigh: float, zlow: float, zhigh: float, temp: float,
                                    border_thickness: float, cascade_steps: int, cutoff_eng: float) -> None:
        """Prepare and write input file for stopping cascade mode."""
        with open(os.path.join(self.template_dir, 'cascade-stopping-0K.lmp'), 'r') as f:
            input_template = f.read()
        input_file = os.path.join(cascade_dir, 'input.lmp')
        stoppingfile = os.path.join(self.template_dir, 'stopping', 'Ge_Ge_elstop_srim96.txt')
        
        with open(input_file, 'w') as f:
            f.write(input_template.format(
                atomsfile=atomsfile, ff_settings=self.potential.ff_settings, 
                border_thickness=border_thickness, pka_id=PKA_id,
                v_x=velocity[0], v_y=velocity[1], v_z=velocity[2],
                mass=self.bi.mass, cascade_steps=cascade_steps,
                xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh, 
                zlow=zlow, zhigh=zhigh, temp=temp, cutoff_eng=cutoff_eng, 
                stoppingfile=stoppingfile))
