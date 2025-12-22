import os, shutil, subprocess
from .calcs_base import TurboGAPCalculator
from ml4cascades.utils import BasicCellInfo
from ml4cascades.potentials import IPotential
from ase.build import bulk
from ase.io import write


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

    def thermalize_atomic(self, supercell_size: list[int], equ_md_steps: int, temp: float, taut: float):
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
            f.write(input_template.format(atomsfile=atomsfile, ff_settings=self.bi.potential.ff_settings, 
                                          num_species=len(self.bi.element), element=' '.join(self.bi.element), 
                                          mass=self.bi.mass, equ_md_steps=equ_md_steps, temp=temp, taut=taut)) 
        subprocess.run('sbatch submit-thermo.sh', shell=True, check=True, cwd=thermalize_dir)

    def thermalize_electronic():
        pass


