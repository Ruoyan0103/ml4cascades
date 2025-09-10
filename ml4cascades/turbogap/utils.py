import numpy as np
import warnings 

class TCeKappa:
    '''
    Ce: electronic heat capacity
    Kappa: electronic thermal conductivity
    '''
    def __init__(
        self, 
        parameters_in_file: str,
        parameters_out_file: str,
        tin_file: str,
        grids: list[int],
        boxsize: list[float],
        T_e: float,
        read_from_param_file: int, # 1: read from parameters file, 0: do not read
        C_e: float=1,
        K_e: float=1,
        N_C_e: int=50,
        M_K_e: int=50
    ):
        self.parameters_in_file = parameters_in_file
        self.parameters_out_file = parameters_out_file
        self.tin_file = tin_file
        self.grids = grids
        self.boxsize = boxsize
        self.read_from_param_file = read_from_param_file
        self.T_e = T_e
        if C_e == 1:
            warnings.warn("C_e is set to the default placeholder (1). Please provide a correct value.")
        if K_e == 1:
            warnings.warn("K_e is set to the default placeholder (1). Please provide a correct value.")
        self.C_e = C_e
        self.K_e = K_e
        self.N_C_e = N_C_e
        self.M_K_e = M_K_e
    
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
        k_ge = np.loadtxt(self.parameters_in_file, skiprows=2)
        Temp = k_ge[:, 0]
        Kappa_org = k_ge[:, 1]
        Ce_org = k_ge[:, 5]
        Kappa_converted = self._convert_Kappa_unit(Kappa_org)
        Ce_converted = self._convert_Ce_unit(Ce_org)
        return Temp, Ce_converted, Kappa_converted
    
    def _write_parameters_file(self):
        Temp, Ce, Kappa = self._read_file()
        with open(self.parameters_out_file, 'w') as f:
            f.write(f'# Ce & Kappa for Ge: {len(Temp)} data, 100 K dT\n')
            f.write('# https://github.com/N-Medvedev/XTANT-3_coupling_data/blob/main/K_semiconductors/K_Ge.dat\n')
            f.write('# First N line: T_e C_e(eV/Ang^3/K), no line gap, then M line: T_e K_e(eV/Ang/K/ps)\n')
            f.write(f'{self.N_C_e}\n')
            for T, C, _ in zip(Temp, Ce, range(self.N_C_e)):
                f.write(f'{T} {C:.6e}\n')
            f.write(f'{self.M_K_e}\n')
            for T, K, _ in zip(Temp, Kappa, range(self.M_K_e)):
                f.write(f'{T} {K:.6e}\n')
        print('Parameters out file is written.')

    def _get_Ce_Ke_for_T(self):
        Temp, Ce, Kappa = self._read_file()
        for T, C, K in zip(Temp, Ce, Kappa):
            if T == self.T_e:
                return C, K
        raise ValueError(f'Temperature {self.T_e} K not found in the parameters file.') 

    def write_tinfile(self):
        self._write_parameters_file()
        with open(self.tin_file, 'w') as f:
            f.write('# Tin file for Ge\n')
            f.write('# \n')
            f.write('# \n')
            f.write(f'{self.grids[0]+1} {self.grids[1]+1} {self.grids[2]+1} 1\n')
            f.write(f'{self.boxsize[0]} {self.boxsize[1]}\n')
            f.write(f'{self.boxsize[2]} {self.boxsize[3]}\n')
            f.write(f'{self.boxsize[4]} {self.boxsize[5]}\n')
            f.write('i j k T_e S_e rho_e C_e K_e flag T_dyn_flag \n')
            for x in range(self.grids[0]+1):
                for y in range(self.grids[1]+1):
                    for z in range(self.grids[2]+1):
                        f.write(f'{x+1} {y+1} {z+1} {self.T_e} 0.000000e+00 1.000000e+00 {self.C_e} {self.K_e} 0 {self.read_from_param_file}\n')
                        # only x, y, z matters, Te, Ce, Kappa will be read from parameters file





