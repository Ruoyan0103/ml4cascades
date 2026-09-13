import os, shutil, subprocess, copy, glob, re
import numpy as np  
from .calcs_base import TurboGAPCalculator
from ml4cascades.utils import BasicCellInfo
from ml4cascades.potentials import IPotential
from ase.build import bulk
from ase.io import write, read
from .utils import lammps_dump_to_extxyz

AMU_TO_KG = 1.66053906660E-27 # Atomic mass unit to kg conversion factor
JOULE_TO_EV = 6.241509074E18  # Joule to eV conversion factor
ANGSTROM_TO_METER = 1E-10     # Angstrom to meter conversion factor
FS_TO_S = 1E-15               # Picosecond to second conversion factor
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

    # -------------------------------------- Slurm submission helper --------------------------------------#
    def _submit_job(self, cwd: str, dependency: str = None, script: str = 'submit-thermo.sh') -> str:
        cmd = 'sbatch --parsable'
        if dependency:
            cmd += f' --dependency=afterok:{dependency}'
        cmd += f' {script}'
        result = subprocess.run(cmd, shell=True, check=True, cwd=cwd, capture_output=True, text=True)
        return result.stdout.strip().split(';')[0]

    # -------------------------------------- Extract last frame of a multi-frame trajectory --------------------------------------#
    # TurboGAP's atoms_file reader always consumes the FIRST frame of an extxyz file, never the last,
    # so a multi-frame trajectory_out.xyz can't be fed directly as the next stage's atoms_file.
    # This submits a tiny dependent job that slices out the last (n_atoms + 2)-line block once the
    # producing job has actually finished (submission is async, so this can't be done in Python at
    # submit time - the trajectory file doesn't exist yet).
    def _submit_last_frame_extraction(self, cwd: str, src_filename: str, dst_filename: str,
                                       n_atoms: int, dependency: str) -> str:
        script_name = f'extract-{dst_filename}.sh'
        script_path = os.path.join(cwd, script_name)
        with open(script_path, 'w') as f:
            f.write(
                "#!/bin/bash\n"
                "#SBATCH --time=00:05:00\n"
                "#SBATCH --partition=sumo\n"
                "#SBATCH --account=sumo\n"
                "#SBATCH --ntasks=1\n"
                "#SBATCH --job-name=extract\n"
                "#SBATCH --error=extract.error\n"
                "#SBATCH --output=extract.output\n\n"
                f"tail -n {n_atoms + 2} {src_filename} > {dst_filename}\n"
            )
        os.chmod(script_path, 0o755)
        return self._submit_job(cwd, dependency=dependency, script=script_name)

    # -------------------------------------- Thermalize (atomic + electronic) --------------------------------------#
    def thermalize(self, atomic_config: dict, electronic_config: dict) -> None:
        supercell_size = atomic_config["supercell_size"]
        equ_md_steps = atomic_config["atomic_equ_md_steps"]
        temp = atomic_config["temp"]
        taut = atomic_config["taut"]
        press = atomic_config["press"]
        taup = atomic_config["taup"]
        gammap = atomic_config["gammap"]

        relax_dir = os.path.join(self.calculation_dir, 'thermalize_atomic', f'{supercell_size[0]}-{supercell_size[1]}-{supercell_size[2]}', 'relax')
        thermalize_atomic_dir = os.path.join(self.calculation_dir, 'thermalize_atomic', f'{supercell_size[0]}-{supercell_size[1]}-{supercell_size[2]}', 'thermalize')
        os.makedirs(relax_dir, exist_ok=True)
        os.makedirs(thermalize_atomic_dir, exist_ok=True)
        shutil.copy(os.path.join(self.template_dir, 'submit-thermo.sh'), relax_dir)
        shutil.copytree(self.tgap_files, os.path.join(relax_dir, 'gap_files'), dirs_exist_ok=True)
        shutil.copytree(self.tgap_files, os.path.join(thermalize_atomic_dir, 'gap_files'), dirs_exist_ok=True)
        with open(os.path.join(self.template_dir, 'input-relax'), 'r') as f:
            input_template = f.read()
        unit_cell = bulk(''.join(self.bi.element), self.bi.lattice,
                         a=self.bi.alat[0], b=self.bi.alat[1], c=self.bi.alat[2], cubic=True)
        supercell = unit_cell.repeat(supercell_size)
        n_atoms = len(supercell)
        atomsfile = os.path.join(thermalize_atomic_dir, 'data.input')
        write(atomsfile, supercell, format='extxyz')
        with open(os.path.join(relax_dir, 'input'), 'w') as f:
            f.write(input_template.format(atomsfile=atomsfile, ff_settings=self.potential.ff_settings,
                                          num_species=len(self.bi.element), element=' '.join(self.bi.element), mass=self.bi.mass))
        self.logger.info(f'#--------------------------------- RELAX ---------------------------------#')
        relax_job = self._submit_job(relax_dir)
        self.logger.info(f'Submitted relax job {relax_job}')

        extract_relax_job = self._submit_last_frame_extraction(
            relax_dir, 'trajectory_out.xyz', 'relaxed.data', n_atoms, dependency=relax_job)
        self.logger.info(f'Submitted relax last-frame extraction job {extract_relax_job} (after {relax_job})')
        relaxed_atomsfile = os.path.join(relax_dir, 'relaxed.data')

        shutil.copy(os.path.join(self.template_dir, 'submit-thermo.sh'), thermalize_atomic_dir)
        with open(os.path.join(self.template_dir, 'input-thermalize-atomic'), 'r') as f:
            input_template = f.read()
        with open(os.path.join(thermalize_atomic_dir, 'input'), 'w') as f:
            f.write(input_template.format(atomsfile=relaxed_atomsfile, ff_settings=self.potential.ff_settings,
                                          num_species=len(self.bi.element), element=' '.join(self.bi.element),
                                          mass=self.bi.mass, equ_md_steps=equ_md_steps, temp=temp, taut=taut,
                                          press=press, taup=taup, gammap=gammap))
        self.logger.info(f'#--------------------------------- THERMALIZE ATOMIC ---------------------------------#')
        atomic_job = self._submit_job(thermalize_atomic_dir, dependency=extract_relax_job)
        self.logger.info(f'Submitted atomic thermalize job {atomic_job} (after {extract_relax_job})')

        extract_atomic_job = self._submit_last_frame_extraction(
            thermalize_atomic_dir, 'trajectory_out.xyz', 'thermalized.data', n_atoms, dependency=atomic_job)
        self.logger.info(f'Submitted atomic thermalize last-frame extraction job {extract_atomic_job} (after {atomic_job})')
        atomsfile = os.path.join(thermalize_atomic_dir, 'thermalized.data')

        equ_md_steps = electronic_config["elec_equ_md_steps"]
        temp = electronic_config["temp"]
        xlow = electronic_config["xlow"]
        xhigh = electronic_config["xhigh"]
        ylow = electronic_config["ylow"]
        yhigh = electronic_config["yhigh"]
        zlow = electronic_config["zlow"]
        zhigh = electronic_config["zhigh"]
        eph_C_e = electronic_config["eph_C_e"]
        eph_kappa_e = electronic_config["eph_kappa_e"]
        eph_tout_file = electronic_config["eph_tout_file"]
        gx = electronic_config.get("gx", 1)
        gy = electronic_config.get("gy", 1)
        gz = electronic_config.get("gz", 1)

        thermalize_electronic_dir = os.path.join(self.calculation_dir, 'thermalize_electronic', f'{supercell_size[0]}-{supercell_size[1]}-{supercell_size[2]}')
        os.makedirs(thermalize_electronic_dir, exist_ok=True)
        shutil.copy(os.path.join(self.template_dir, 'submit-thermo.sh'), thermalize_electronic_dir)
        shutil.copytree(self.tgap_files, os.path.join(thermalize_electronic_dir, 'gap_files'), dirs_exist_ok=True)
        with open(os.path.join(self.template_dir, 'input-thermalize-electronic'), 'r') as f:
            input_template = f.read()
        beta_file = os.path.join(self.template_dir, 'betafile', 'beta_Ge.dat')
        with open(os.path.join(thermalize_electronic_dir, 'input'), 'w') as f:
            f.write(input_template.format(atomsfile=atomsfile, ff_settings=self.potential.ff_settings,
                                          num_species=len(self.bi.element), element=' '.join(self.bi.element),
                                          mass=self.bi.mass, equ_md_steps=equ_md_steps, temp=temp, beta_file=beta_file,
                                          xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh, zlow=zlow, zhigh=zhigh,
                                          eph_C_e=eph_C_e, eph_kappa_e=eph_kappa_e, eph_tout_file=eph_tout_file,
                                          gx=gx, gy=gy, gz=gz))
        self.logger.info(f'#--------------------------------- THERMALIZE ELECTRONIC ---------------------------------#')
        electronic_job = self._submit_job(thermalize_electronic_dir, dependency=extract_atomic_job)
        self.logger.info(f'Submitted electronic thermalize job {electronic_job} (after {extract_atomic_job})')

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
    def _write_tinfile(self, tinfile_path: str, gsx: int, gsy: int, gsz: int, 
                       xlow: float, xhigh: float, ylow: float, yhigh: float, 
                       zlow: float, zhigh: float, temp: float, 
                       eph_C_e: float, eph_kappa_e: float, temperature_dependent: bool, 
                       cascade_dir: str) -> None:
        """Write temperature input file for EPH mode."""
        t_dyn_flag = 1 if temperature_dependent else 0
        with open(tinfile_path, 'w') as f:
            f.write('# comment1 \n# comment2 \n# comment3 \n')
            f.write(f'{gsx} {gsy} {gsz} 1\n')
            f.write(f'{xlow} {xhigh} \n')
            f.write(f'{ylow} {yhigh} \n')
            f.write(f'{zlow} {zhigh} \n')
            f.write('i j k T_e S_e rho_e C_e K_e flag T_dyn_flag\n')
            for iz in range(1, gsz + 1):
                for iy in range(1, gsy + 1):
                    for ix in range(1, gsx + 1):
                        f.write(f'{ix} {iy} {iz} {temp} 0 1 {eph_C_e} {eph_kappa_e} 1 {t_dyn_flag}\n')
        if temperature_dependent:
            param_file = os.path.join(cascade_dir, 'Te-dependent_e-parameters.txt')
            k_ge_file = os.path.join(os.path.dirname(module_dir), 'utils', 'parameters', 'K_Ge.dat')
            data = np.loadtxt(k_ge_file, skiprows=2)
            Te_grid = data[:, 0]
            PS_TO_S = 1E-12
            kappa_e = data[:, 1] * JOULE_TO_EV / (1 / ANGSTROM_TO_METER * (1 / PS_TO_S))
            C_e_raw = data[:, 5] * JOULE_TO_EV / ((1 / ANGSTROM_TO_METER) ** 3)

            C_e_floor = 5e-7
            K_e_floor = 2.52e-4
            below_floor = C_e_raw < C_e_floor
            C_e_data = np.where(below_floor, C_e_floor, C_e_raw)
            K_e_data = np.where(below_floor, K_e_floor, kappa_e)

            with open(param_file, 'w') as f:
                f.write('# First 3 comment lines\n')
                f.write('# Te-dependent_e-parameters.txt file\n')
                f.write('# First N C_e(T_e) with T_e, no line gap, then M K_e(T_e) with T_e\n')
                f.write(f'{len(Te_grid)}\n')
                for T_e, C_e_val in zip(Te_grid, C_e_data):
                    f.write(f'{T_e:.1f} {C_e_val:.6e}\n')
                f.write(f'{len(Te_grid)}\n')
                for T_e, K_e_val in zip(Te_grid, K_e_data):
                    f.write(f'{T_e:.1f} {K_e_val:.6e}\n')
    
    def run_cascade(self,
                    num_PKA_directions: int,
                    running_directions: list[int], # directions to run, example [1, 2, 3], start from 1 to num_PKA_directions
                    radius_frac: float,
                    PKA_kin_eng: float,
                    input_config: dict,
                    tinfile: str = None,
                    running_dir: str = None,
                    lammps_thermalize: str = None) -> str:
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
        temperature_dependent = input_config.get("temperature_dependent", False)
        if running_dir is not None:
            PKA_kin_eng_dir = running_dir
        else:
            PKA_kin_eng_dir = os.path.join(self.calculation_dir, 'cascade', f'PKA_{int(PKA_kin_eng)}eV-{radius_frac}')
        os.makedirs(PKA_kin_eng_dir, exist_ok=True)

        if lammps_thermalize is not None:
            converted_file = os.path.join(PKA_kin_eng_dir, 'thermalized_from_lammps.xyz')
            thermalized_struct = lammps_dump_to_extxyz(lammps_thermalize, converted_file, specorder=self.bi.element, index=-1)
        else:
            thermalized_struct = read(os.path.join(self.calculation_dir, 'thermalize_atomic', f'{supercell_size[0]}-{supercell_size[1]}-{supercell_size[2]}', 'thermalize', 'thermalized.data'), format='extxyz', index=-1)

        dirs = self._get_PKA_directions(num_PKA_directions)
        atom_positions = thermalized_struct.get_positions()
        atom_velocities = thermalized_struct.get_array('velocities')
        cell_lengths = thermalized_struct.cell.lengths()
        radius = 0.5 * min(cell_lengths) * radius_frac
        center = thermalized_struct.get_center_of_mass()
        self.logger.info(f'#--------------------------------- Ekin: {int(PKA_kin_eng)} eV size: {supercell_size[0]}*{supercell_size[1]}*{supercell_size[2]}---------------------------------#')
        for running_direction in running_directions:
            if running_direction > num_PKA_directions:
                self.logger.error(f'Running direction {running_direction} exceeds the number of generated PKA directions {num_PKA_directions}.')
                continue
            idx = running_direction - 1  # running_direction starts from 1
            xyz = dirs[idx]

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
            cascade_file = os.path.join(cascade_dir, 'atoms_file_s1.xyz')
            write(cascade_file, cascade_struct, format='extxyz')

            # prepare tinfile
            tinfile_path = os.path.join(cascade_dir, 'T_input_s1.fdm')
            if tinfile is None:
                self._write_tinfile(tinfile_path, gsx, gsy, gsz, xlow, xhigh, ylow, yhigh, zlow, zhigh, temp, eph_C_e, eph_kappa_e, temperature_dependent, cascade_dir)
            else:
                shutil.copy(tinfile, tinfile_path)

            # prepare input file
            shutil.copytree(self.tgap_files, os.path.join(cascade_dir, 'gap_files'), dirs_exist_ok=True)
            with open(os.path.join(self.template_dir, 'submit-cascade.sh'), 'r') as f:
                submit_template = f.read()
            submit_file = os.path.join(cascade_dir, 'submit-cascade.sh')
            with open(submit_file, 'w') as f:
                f.write(submit_template.replace('{num}', str(idx+1)))
            beta_file = os.path.join(self.template_dir, 'betafile', 'beta_Ge.dat')

            # one input_s{n} file per cascade stage, found by globbing the
            # 'input-cascade_s*' templates so adding/removing stages doesn't
            # require touching this code
            stage_templates = sorted(
                glob.glob(os.path.join(self.template_dir, 'input-cascade_s*')),
                key=lambda p: int(re.search(r's(\d+)$', p).group(1)))

            prev_stage_name = None
            for template_path in stage_templates:
                stage_name = os.path.basename(template_path).rsplit('_', 1)[-1]  # 's1', 's2', ...
                with open(template_path, 'r') as f:
                    input_template = f.read()

                # stage 1 starts from the PKA structure; later stages continue
                # from the previous stage's final frame (extracted by
                # submit-cascade.sh, since atoms_file only reads frame 1)
                if prev_stage_name is None:
                    stage_atomsfile = cascade_file
                else:
                    stage_atomsfile = f'atoms_file_{stage_name}.xyz'

                input_file = os.path.join(cascade_dir, f'input_{stage_name}')
                with open(input_file, 'w') as f:
                    f.write(input_template.format(atomsfile=stage_atomsfile, ff_settings=self.potential.ff_settings,
                                                  num_species=len(self.bi.element), element=' '.join(self.bi.element),
                                                  mass=self.bi.mass, temp=temp, beta_file=beta_file,
                                                  xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh, zlow=zlow, zhigh=zhigh,
                                                  gsx=gsx, gsy=gsy, gsz=gsz,
                                                  eph_C_e=eph_C_e, eph_kappa_e=eph_kappa_e, eph_tout_file=eph_tout_file))
                prev_stage_name = stage_name

            # subprocess.run('sbatch submit-cascade.sh', shell=True, check=True, cwd=cascade_dir)
        return PKA_kin_eng_dir
