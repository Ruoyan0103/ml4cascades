import numpy as np 
import matplotlib.pyplot as plt
import os 

AMU_TO_KG = 1.66053906660E-27 # Atomic mass unit to kg conversion factor
JOULE_TO_EV = 6.241509074E18  # Joule to eV conversion factor
ANGSTROM_TO_METER = 1E-10     # Angstrom to meter conversion factor
PS_TO_S = 1E-12               # Picosecond to second conversion factor

module_dir = os.path.dirname(__file__)

class ParameterGetter:
    def __init__(self, temp):
        self.temp = temp 

    def get_data1(self, filename: str=os.path.join(module_dir, 'parameters', 'K_Ge.dat')) -> tuple[float, float]:
        # data from https://github.com/N-Medvedev/XTANT-3_coupling_data/blob/main/K_semiconductors/K_Ge.dat
        data = np.loadtxt(filename, skiprows=2)
        Te = data[:, 0]  
        kappa_e = np.array(data[:, 1]) * JOULE_TO_EV / (1/ANGSTROM_TO_METER * (1/PS_TO_S))  
        C_e = np.array(data[:, 5]) * JOULE_TO_EV / ((1/ANGSTROM_TO_METER)**3)

        if self.temp < Te[0] or self.temp > Te[-1]:
            raise ValueError(f"Temperature {self.temp} K is out of range ({Te[0]} K to {Te[-1]} K)")
        else:
            kappa_e_temp = np.interp(
                self.temp, 
                Te, 
                np.array(data[:, 1]) * JOULE_TO_EV / (1/ANGSTROM_TO_METER * (1/PS_TO_S))
            )
            C_e_temp = np.interp(
                self.temp, 
                Te, 
                np.array(data[:, 5]) * JOULE_TO_EV / ((1/ANGSTROM_TO_METER)**3)
            )
            
        fig, ax = plt.subplots(2, 1, figsize=(8, 10))
        # Left subplot: Electron heat capacity
        ax[0].set_yscale('log')
        ax[0].set_xscale('log')
        ax[0].plot(Te, C_e, color='orange')
        # ax[0].set_xlabel(r'Te (K)')
        ax[0].set_ylabel(
            r'Ce '
            r'$\left(\mathrm{eV\,K^{-1}\,\AA^{-3}}\right)$', fontsize=15
        )
        ax[0].axvline(x=self.temp, color='gray', linestyle='--', linewidth=1)
        ax[0].axhline(y=C_e_temp, color='gray', linestyle='--', linewidth=1)
        # Right subplot: Electron thermal conductivity
        ax[1].set_yscale('log')
        ax[1].set_xscale('log')
        ax[1].plot(Te, kappa_e, color='blue')
        ax[1].set_xlabel(r'Te (K)', fontsize=15)
        ax[1].set_ylabel(
            r'Ke '
            r'$\left(\mathrm{eV\,K^{-1}\,\AA^{-1}\,ps^{-1}}\right)$', fontsize=15
        )
        ax[1].axvline(x=self.temp, color='gray', linestyle='--', linewidth=1)
        ax[1].axhline(y=kappa_e_temp, color='gray', linestyle='--', linewidth=1)

        ax[0].tick_params(axis='both', which='major', labelsize=12, length=6, width=1.2)
        ax[0].tick_params(axis='both', which='minor', labelsize=10, length=3, width=1.0)
        ax[1].tick_params(axis='both', which='major', labelsize=12, length=6, width=1.2)
        ax[1].tick_params(axis='both', which='minor', labelsize=10, length=3, width=1.0)
        ax[0].set_title(f'Temp: {self.temp} K, Ce: {C_e_temp:.3e} eV/K/A^3, kappa_e: {kappa_e_temp:.3e} eV/K/A/ps', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(module_dir, 'parameters', 'thermal_properties.png'), dpi=300)
        return C_e_temp, kappa_e_temp

    def get_data2(self, filename: str=os.path.join(module_dir, 'parameters', 'Ce_Ge.txt')):
        # data from https://github.com/N-Medvedev/XTANT-3_coupling_data/blob/main/Ce_semiconductors/Ce_Ge.txt
        data = np.loadtxt(filename, skiprows=2)
        Te = data[:, 0]
        C_e = np.array(data[:, 1]) * JOULE_TO_EV / ((1/ANGSTROM_TO_METER)**3)
        if self.temp < Te[0] or self.temp > Te[-1]:
            raise ValueError(f"Temperature {self.temp} K is out of range ({Te[0]} K to {Te[-1]} K)")
        else:
            C_e_temp = np.interp(
                self.temp, 
                Te, 
                np.array(data[:, 1]) * JOULE_TO_EV / ((1/ANGSTROM_TO_METER)**3)
            )

        # one plot 
        plt.figure(figsize=(6, 5))
        plt.yscale('log')
        plt.xscale('log')
        plt.plot(Te, C_e, color='orange')
        plt.xlabel(r'Te (K)', fontsize=15)
        plt.ylabel(
            r'Ce '
            r'$\left(\mathrm{eV\,K^{-1}\,\AA^{-3}}\right)$', fontsize=15
        )
        plt.tick_params(axis='both', which='major', labelsize=12, length=6, width=1.2)
        plt.tick_params(axis='both', which='minor', labelsize=10, length=3, width=1.0)  
        plt.axvline(x=self.temp, color='gray', linestyle='--', linewidth=1)
        plt.axhline(y=C_e_temp, color='gray', linestyle='--', linewidth=1)
        plt.title(f'Temp: {self.temp} K, Ce: {C_e_temp:.3e} eV/K/A^3', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(module_dir, 'parameters', 'heat_capacity.png'), dpi=300)
        return C_e_temp
    
