import numpy as np
class TCeKappa:
    '''
    Ce: electronic heat capacity
    Kappa: electronic thermal conductivity
    '''
    def __init__(
        self, 
        parameters_in_file: str,
        parameters_out_file: str,
        tin_file: Optional[str] = None,
        grids: Optional[list[int]] = None,
        boxsize: Optional[list[float]] = None,
        constant_Temp: Optional[float] = None
    ):
        if tin_file is None and constant_Temp is None:
            raise ValueError(
                "You must provide either `tin_file` or `constant_Temp`.\n"
                "Ce and Kappa must either come from a file or be constant values."
            )
        if tin_file is not None and (grids or boxsize is None):
            raise ValueError("If `tin_file` is provided, `grids` must also be set and non-empty.")
        self.parameters_in_file = parameters_in_file
        self.parameters_out_file = parameters_out_file
        self.tin_file = tin_file
        self.grids = grids
        self.boxsize = boxsize
        self.constant_Temp = constant_Temp
    
    def _convert_Ce_unit(
        self,
        values: list
    ):
    '''
        Convert Ce (heat capacity) unit from J/m^3/K to eV/Å^3/K
        1 J = 6.242e18 eV
        1 m^3 = 1e30 Å^3
        1 J/m^3/K = 6.242e-12 eV/Å^3/K
    '''
        converted_values = [value * 6.242e-12 for value in values]
        return converted_values

    def _convert_Kappa_unit(
        self,
        values: list
    ):
    '''
        Convert Kappa (heat conductivity) unit from W/m/K to eV/Å/K/ps
        1 W = 1 J/s = 6.242e18 eV/s
        1 W/m/K = 6.242e18 eV/m/K/s
        1 m = 1e10 Å
        1 s = 1e12 ps
        1 W/m/K = 6.242e-4 eV/Å/K/ps
    '''
        converted_values = [value * 6.242e-4 for value in values]
        return converted_values
    
    def _read_file(self):
        k_ge = np.loadtxt(self.parameters_file, skiprows=2)
        Temp = k_ge[:, 0]
        Ce_org = k_ge[:, 1]
        Kappa_org = k_ge[:, 5]
        Ce_converted = self._convert_Ce_unit(Ce_org)
        Kappa_converted = self._convert_Kappa_unit(Kappa_org)
        return Temp, Ce_converted, Kappa_converted
    
    def _write_parameters_file(self):
        Temp, Ce, Kappa = self._read_file()
        with open(self.parameters_out_file, 'w') as f:
            f.write(f'# Ce & Kappa for Ge: {len(Temp)} data, 100 K dT\n')
            f.write('# https://github.com/N-Medvedev/XTANT-3_coupling_data/blob/main/K_semiconductors/K_Ge.dat\n')
            f.write('# Ce (eV/Ang^3/K), Kappa (eV/Ang/K/ps)\n')
            f.write(f'{len(Temp)} 100\n')
            for C, K in zip(Ce, Kappa):
                f.write(f'{C:.6e} {K:.6e}\n')
        print('Parameters out file is written.')

    def constant_Ce_Kappa(self):
        Ce_constant = None
        Kappa_constant = None
        if self.constant_Temp is not None:
            Temp, Ce, Kappa = self._read_file()
            for t, c, k in zip(Temp, Ce, Kappa):
                if self.constant_Temp == t:
                    Ce_constant = c
                    Kappa_constant = k
                    break
        if Ce_constant is None or Kappa_constant is None:
            raise ValueError(f'Ce and Kappa at {self.constant_Temp} is not found.')
        return Ce_constant, Kappa_constant

    def tinfile_Ce_Kappa(self):
        if self.tin_file is not None:
            self.write_parameters_file()
            with open(self.tin_file, 'w') as f:
                f.write('# Tin file for Ge\n')
                f.write(f'# Read parameters from {self.parameters_out_file}\n')
                f.write('# i j k T_e S rho_e C_e kappa_e f\n')
                f.write(f'{self.grids[0]} {self.grids[1]} {self.grids[2]} 1\n')
                f.write(f'{self.boxsize[0]} {self.boxsize[1]}\n')
                f.write(f'{self.boxsize[2]} {self.boxsize[3]}\n')
                f.write(f'{self.boxsize[4]} {self.boxsize[5]}\n')
                for x in range(self.grids[0]):
                    for y in range(self.grids[1]):
                        for z in range(self.grids[2]):
                            f.write(f'{x} {y} {z} 3.000000e+02 0.000000e+00 1.000000e+00 1.000000e-05 1.000000e-09 1 1\n')
                            # only x, y, z matters, Te, Ce, Kappa will be read from parameters file





