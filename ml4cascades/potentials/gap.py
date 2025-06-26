import os
import xml.etree.ElementTree as ET
from ml4cascades.potentials import IPotential
from ml4cascades.turbogap.calcs import CascadeCalculator


module_dir = os.path.dirname(__file__)


class GAPotential(IPotential):
    pair_style = 'pair_style        quip'
    pair_coeff = 'pair_coeff        * * {} {} {}'


    def __init__(self, params):
        self.name = 'GAP'
        self.params = params
        self.ff_settings = None


    @staticmethod
    def from_config(filename):
        """
        Initialize potentials with parameters file.

        ARgs:
            filename (str): The file storing parameters of potentials.

        Returns:
            GAPotential.
        """
        if filename.endswith('.xml'):
            def get_xml(xml_file):
                tree = ET.parse(xml_file)
                root = tree.getroot()
                potential_label = root.tag
                pairpot = root.find('pairpot')
                glue_params = pairpot.find('Glue_params')
                per_type_data = glue_params.find('per_type_data')
                specie_z = per_type_data.get('atomic_num')
                return filename, potential_label, specie_z

            filename, potential_label, specie_z = get_xml(filename)
            params = dict(xml_file=filename, potential_label=potential_label, specie_z=specie_z)
            return GAPotential(params)
        

    def write_param(self, xml_filename=None):
        """
        Write potential parameters for lammps calculation.

        Args:
            xml_filename (str): Filename to store xml formatted parameters.
        """
        self.element = self.params.get('specie_z', None)
        self.pair_coeff = self.pair_coeff.format(xml_filename, 
                                            '\"Potential xml_label={}\"'.format(self.params.get('potential_label', None)),
                                            self.params.get('specie_z', None))
        self.ff_settings = [self.pair_style, self.pair_coeff]


class TGAPotential(IPotential):
    def __init__(self):
        self.name = 'TGAP'
        self.ff_settings = '{}'

    def write_param(self, params):
        self.ff_settings = self.ff_settings.format(params)


if __name__ == "__main__":
    # gap_file = os.path.join(module_dir, 'params', 'GAP', 'Ge-v10.xml')
    # gap = GAPotential.from_config(gap_file)
    # gap.write_param(gap_file)

    # mass, element, lattice, alat, temperature = 72.56, 'Ge', 'diamond', 5.76, 300    
    # energies, num_directions = [100, 400, 1000, 2000, 5000, 10e3, 20e3, 50e3], 30
    # energies, num_directions = [100, 400, 1000], 30
    # sizes = [9, 9, 9]     
    # pka_ids = [1202, 1202, 1202]  
    # cas_calc = CascadeCalculator(gap, mass, element, lattice, alat, sizes, temperature,
    #                              pka_ids, energies, num_directions)
    # cas_calc.calculate()

    gap_file_folder = os.path.join(module_dir, 'params', 'TGAP') 
    gap_file = 'Ge-v10-gap'
    tgap = TGAPotential()
    tgap.write_param(gap_file)
    num_species, mass, element, lattice, alat, temperature = 1, 72.64, 'Ge', 'diamond', 5.76, 300 
    simulation_steps = 1000
    energies, num_sampling_points = [15, 20], 10
    sizes = [3, 3] 
    cas_calc = CascadeCalculator(tgap, num_species, mass, element, lattice, alat, sizes, temperature,
                                energies, num_sampling_points, simulation_steps, gap_file_folder)
    cas_calc.calculate()
