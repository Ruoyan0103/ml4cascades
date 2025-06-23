import os
import xml.etree.ElementTree as ET
from ml4cascades.potentials import IPotential
from ml4cascades.lammps.calcs import RelaxationCalculator, CascadeCalculator

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


if __name__ == "__main__":
    gap_file = os.path.join(module_dir, 'params', 'GAP', 'Ge-v10.xml')
    gap = GAPotential.from_config(gap_file)
    gap.write_param(gap_file)

    mass, element, lattice, alat, size, temperature = 72.56, 'Ge', 'diamond', 5.76, 9, 300    
    relax_calc = RelaxationCalculator(gap, mass, element, lattice, alat, size, temperature)
    # relax_calc.calculate()

    pka_id = 10
    energies, num_directions = [100, 400, 1000, 2000, 5000, 10e3, 20e3, 50e3], 30
    cas_calc = CascadeCalculator(gap, mass, element, lattice, alat, size, temperature,
                                 pka_id, energies, num_directions)
    cas_calc.calculate()