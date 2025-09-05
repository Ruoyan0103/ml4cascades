import os
import xml.etree.ElementTree as ET
from ml4cascades.potentials import IPotential
from ml4cascades.utils import BasicInput
from ml4cascades.turbogap import CascadeCalculatorEPH
# from ml4cascades.lammps.calcs import CascadeCalculator
import numpy as np

module_dir = os.path.dirname(__file__)

class GAPotential(IPotential):
    pair_style = 'pair_style        quip'
    pair_coeff_template = 'pair_coeff        * * {} {} {}'

    def __init__(self, params: dict):
        self.name = 'GAP'
        self.params = params
        self.element = None
        self.ff_settings = None

    @staticmethod
    def from_config(filename: str) -> 'GAPotential':
        if not filename.endswith('.xml'):
            raise ValueError(f"Only XML files are supported: {filename}")

        tree = ET.parse(filename)
        root = tree.getroot()
        potential_label = root.tag
        pairpot = root.find('pairpot')
        glue_params = pairpot.find('Glue_params')
        per_type_data = glue_params.find('per_type_data')
        specie_z = per_type_data.get('atomic_num')

        params = dict(xml_file=filename, potential_label=potential_label, specie_z=specie_z)
        return GAPotential(params)

    def write_param(self, filename: str):
        self.element = self.params.get('specie_z')
        pair_coeff = self.pair_coeff_template.format(
            filename,
            f"\"Potential xml_label={self.params.get('potential_label')}\"",
            self.params.get('specie_z')
        )
        self.ff_settings = [self.pair_style, pair_coeff]


class TGAPotential(IPotential):
    ff_template = '{}'  

    def __init__(self):
        self.name = 'TGAP'
        self.ff_settings = None

    def write_param(self, param_file: str):
        self.ff_settings = self.ff_template.format(param_file)


if __name__ == "__main__":
    gap_file = os.path.join(module_dir, 'params', 'GAP', 'Ge-v10.xml')
    gap = GAPotential.from_config(gap_file)
    gap.write_param(gap_file)
    gap_ff_settings = gap.ff_settings

    gap_file_folder = os.path.join(module_dir, 'params', 'TGAP') 
    gap_file = 'Ge-v10-gap'
    tgap = TGAPotential()
    tgap.write_param(gap_file)
    tgap_ff_settings = tgap.ff_settings

    bi = BasicInput(ff_settings=tgap_ff_settings, mass=[72.64], element=['Ge'], lattice=['diamond'], alat=[[5.76]*3])
    temperature = 300
    num_directions = 30
    equ_md_steps = 30000
    cascade_md_steps = 30500





    energies, num_directions = [100], 500
    # energies, num_directions = [2000], 500
    # energies, num_directions = [5000], 500
    # energies, num_directions = [10e3, 20e3], 500
    sizes = [] 
    radius_fracs = []
    for e in energies:
        if e < 400:
            sizes.append(int(np.ceil(np.cbrt(e*20/8))))
            radius_fracs.append(0.5)
        elif e >= 400 and e < 2000:
            sizes.append(int(np.ceil(np.cbrt(e*30/8))))
            radius_fracs.append(0.5)
        elif e >= 2000 and e <= 5000:
            sizes.append(int(np.ceil(np.cbrt(e*40/8))))
            radius_fracs.append(0.8)
        elif e > 5000 and e <= 20e3:
            sizes.append(int(np.ceil(np.cbrt(e*55/8))))
            radius_fracs.append(0.9)
    # thicknesses = [alat] * len(energies)
    thicknesses = [alat/2] * len(energies)

    cas_calc = CascadeCalculator(gap, mass, element, lattice, alat, sizes, thicknesses, radius_fracs, temperature,
                                energies, num_directions)
    cas_calc.calculate(relax_flag=False, simulation_flag=False, check_flag=False, postprocess_flag=True) 




   
    # energies, num_sampling_points = [100, 400], 500
    # energies, num_sampling_points = [1000, 2000, 5000], 500
    # # energies, num_directions = [100], 500
    # # energies, num_directions = [5000], 500
    # # energies, num_directions = [10e3, 20e3], 500
    # sizes = [] 
    # radius_fracs = []
    # for e in energies:
    #     if e <= 400:
    #         sizes.append(int(np.ceil(np.cbrt(e*20/8))))
    #         radius_fracs.append(0.5)
    #     elif e > 400 and e <= 2000:
    #         sizes.append(int(np.ceil(np.cbrt(e*30/8))))
    #         radius_fracs.append(0.5)
    #     elif e > 2000 and e <= 5000:
    #         sizes.append(int(np.ceil(np.cbrt(e*40/8))))
    #         radius_fracs.append(0.8)
    #     elif e > 5000 and e <= 20e3:
    #         sizes.append(int(np.ceil(np.cbrt(e*55/8))))
    #         radius_fracs.append(0.9)
    # thicknesses = [alat] * len(energies)
    # equilibration_steps = 30000
    # cascade_steps = 30500
    # cas_calc = CascadeCalculator(tgap, num_species, mass, element, lattice, alat, sizes, thicknesses, radius_fracs, temperature,
    #                             energies, num_sampling_points, equilibration_steps, cascade_steps, gap_file_folder)
    # cas_calc.calculate(relaxflag=True, cascadeflag=False)
    # cas_calc.postProcess() 
