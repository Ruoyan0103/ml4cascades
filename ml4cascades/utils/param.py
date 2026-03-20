import numpy as np 
import matplotlib.pyplot as plt
import os 
import math 

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
TICK_FONTSIZE = 15
LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 14

AMU_TO_KG = 1.66053906660E-27 # Atomic mass unit to kg conversion factor
JOULE_TO_EV = 6.241509074E18  # Joule to eV conversion factor
ANGSTROM_TO_METER = 1E-10     # Angstrom to meter conversion factor
PS_TO_S = 1E-12               # Picosecond to second conversion factor

module_dir = os.path.dirname(__file__)

class ParameterGetter:
    def __init__(self):
        pass

    def get_data1(self, temp: float, 
                  filename: str=os.path.join(module_dir, 'parameters', 'K_Ge.dat')) -> tuple[float, float]:
        # data from https://github.com/N-Medvedev/XTANT-3_coupling_data/blob/main/K_semiconductors/K_Ge.dat
        # linear interpolation
        element = os.path.basename(filename).split('_')[1].split('.')[0]
        data = np.loadtxt(filename, skiprows=2)
        Te = data[:, 0]  
        kappa_e = np.array(data[:, 1]) * JOULE_TO_EV / (1/ANGSTROM_TO_METER * (1/PS_TO_S))  
        C_e = np.array(data[:, 5]) * JOULE_TO_EV / ((1/ANGSTROM_TO_METER)**3)

        if temp < Te[0] or temp > Te[-1]:
            raise ValueError(f"Temperature {temp} K is out of range ({Te[0]} K to {Te[-1]} K)")
        else:
            kappa_e_temp = np.interp(
                temp, 
                Te, 
                np.array(data[:, 1]) * JOULE_TO_EV / (1/ANGSTROM_TO_METER * (1/PS_TO_S))
            )
            C_e_temp = np.interp(
                temp, 
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
            r'$\left(\mathrm{eV\,K^{-1}\,\AA^{-3}}\right)$', fontsize=15,
            fontname='DejaVu Serif'
        )
        ax[0].axvline(x=temp, color='gray', linestyle='--', linewidth=1)
        ax[0].axhline(y=C_e_temp, color='gray', linestyle='--', linewidth=1)
        # add text for largest and smallest Ce, and the corresponding temperature (Te, Ce)
        ax[0].text(
            Te[0],
            C_e[0],
            f"{Te[0]} K, {C_e[0]:.2e} " + r"$\mathrm{eV\,K^{-1}\,\AA^{-3}}$",
            fontsize=13,
            verticalalignment='top',
            horizontalalignment='left'
        )
        ax[0].text(
            Te[-1],
            C_e[-1],
            f"{Te[-1]} K, {C_e[-1]:.2e} " + r"$\mathrm{eV\,K^{-1}\,\AA^{-3}}$",
            fontsize=13,
            verticalalignment='bottom',
            horizontalalignment='right'
        )
        
        # Right subplot: Electron thermal conductivity
        ax[1].set_yscale('log')
        ax[1].set_xscale('log')
        ax[1].plot(Te, kappa_e, color='blue')
        ax[1].set_xlabel(r'Te (K)', fontsize=15)
        ax[1].set_ylabel(
            r'$\kappa_e$ '
            r'$\left(\mathrm{eV\,K^{-1}\,\AA^{-1}\,ps^{-1}}\right)$', fontsize=15,
            fontname='DejaVu Serif'
        )
        ax[1].axvline(x=temp, color='gray', linestyle='--', linewidth=1)
        ax[1].axhline(y=kappa_e_temp, color='gray', linestyle='--', linewidth=1)
        # add text for largest and smallest kappa_e, and the corresponding temperature (Te, ke)
        # ax[1].text(Te[0], kappa_e[0], f"{Te[0]}, {kappa_e[0]:.2e}", fontsize=10, verticalalignment='bottom', horizontalalignment='left')
        # ax[1].text(Te[-1], kappa_e[-1], f"{Te[-1]}, {kappa_e[-1]:.2e}", fontsize=10, verticalalignment='bottom', horizontalalignment='right')
        ax[1].text(
            Te[0],
            kappa_e[0],
            f"{Te[0]} K, {kappa_e[0]:.2e} " + r"$\left(\mathrm{eV\,K^{-1}\,\AA^{-1}\,ps^{-1}}\right)$",
            fontsize=13,
            verticalalignment='top',
            horizontalalignment='left'
        )
        ax[1].text(
            Te[-1],
            kappa_e[-1],
            f"{Te[-1]} K, {kappa_e[-1]:.2e} " + r"$\left(\mathrm{eV\,K^{-1}\,\AA^{-1}\,ps^{-1}}\right)$",
            fontsize=13,
            verticalalignment='bottom',
            horizontalalignment='right'
        )

        ax[0].tick_params(axis='both', which='major', labelsize=12, length=6, width=1.2)
        ax[0].tick_params(axis='both', which='minor', labelsize=10, length=3, width=1.0)
        ax[1].tick_params(axis='both', which='major', labelsize=12, length=6, width=1.2)
        ax[1].tick_params(axis='both', which='minor', labelsize=10, length=3, width=1.0)
        # ax[0].set_title(f'Temp: {temp} K, Ce: {C_e_temp:.3e} eV/K/A^3, kappa_e: {kappa_e_temp:.3e} eV/K/A/ps', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(module_dir, 'parameters', 'figs', f'thermal_properties_{element}.png'), dpi=300)
        return C_e_temp, kappa_e_temp

    def write_data1(self, outputfile: str,
                    filename: str=os.path.join(module_dir, 'parameters', 'K_Ge.dat')):
        # data from https://github.com/N-Medvedev/XTANT-3_coupling_data/blob/main/K_semiconductors/K_Ge.dat
        data = np.loadtxt(filename, skiprows=2)
        Te = data[:, 0]
        const_dT = Te[1] - Te[0]
        min_temp = 0
        epsilon = 1e-9
        start_idx = math.floor((Te[0]-min_temp) / const_dT + epsilon)
        value_num_points = math.floor((Te[-1]-Te[0])/const_dT + epsilon) + 1

        kappa_e = np.array(data[:, 1]) * JOULE_TO_EV / (1/ANGSTROM_TO_METER * (1/PS_TO_S))  
        C_e = np.array(data[:, 5]) * JOULE_TO_EV / ((1/ANGSTROM_TO_METER)**3)
        
        with open(outputfile, 'w') as f:
            f.write("#\n#\n#\n")
            f.write(f"{start_idx+value_num_points} {const_dT}\n")
            for _ in range(start_idx):                # [0, start_idx-1])
                f.write("5e-08 0.0002529\n")          # fake values for temps below the Te[0]
            for i in range(0, value_num_points):      # [start_idx, start_idx + value_num_points -1]
                if C_e[i] < 5e-8:
                    f.write(f"5e-8 0.0002529\n")
                else:
                    f.write(f"{C_e[i]:.6e} {kappa_e[i]:.6e}\n")

    def get_data2(self, temp: float, 
                  filename: str=os.path.join(module_dir, 'parameters', 'Ce_W.txt')):
        # data from https://github.com/N-Medvedev/XTANT-3_coupling_data/blob/main/Ce_semiconductors/Ce_Ge.txt
        # linear interpolation
        element = os.path.basename(filename).split('_')[1].split('.')[0]
        data = np.loadtxt(filename, skiprows=2)
        Te = data[:, 0]
        C_e = np.array(data[:, 1]) * JOULE_TO_EV / ((1/ANGSTROM_TO_METER)**3)
        if temp < Te[0] or temp > Te[-1]:
            raise ValueError(f"Temperature {temp} K is out of range ({Te[0]} K to {Te[-1]} K)")
        else:
            C_e_temp = np.interp( 
                temp, 
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
        plt.axvline(x=temp, color='gray', linestyle='--', linewidth=1)
        plt.axhline(y=C_e_temp, color='gray', linestyle='--', linewidth=1)
        # add text for smallest and largest Ce, and the corresponding temperature (Te, Ce)
        plt.text(Te[0], C_e[0], f"{Te[0]}, {C_e[0]:.2e}", fontsize=10, verticalalignment='bottom', horizontalalignment='left')
        plt.text(Te[-1], C_e[-1], f"{Te[-1]}, {C_e[-1]:.2e}", fontsize=10, verticalalignment='bottom', horizontalalignment='right')

        plt.title(f'Temp: {temp} K, Ce: {C_e_temp:.3e} eV/K/A^3', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(module_dir, 'parameters', 'figs', f'heat_capacity_{element}.png'), dpi=300)
        return C_e_temp

    def estimate_Ce(self, PKA_energy_eV, band_gap_eV, volume_A3, fe=1.0):
        kB_eV_per_K = 8.617333e-5 
        E_elec = fe * PKA_energy_eV 
        n_eh = E_elec / band_gap_eV  
        n_density = n_eh / volume_A3  
        Ce = n_density * kB_eV_per_K
        return Ce
    
    
if __name__ == "__main__":
    getter = ParameterGetter()
    temp = 450
    Ce, ke = getter.get_data1(temp)
    print(f"At {temp} K: Ce: {Ce:.3e} eV/K/A^3, ke: {ke:.3e} eV/K/A/ps")
    # Ce = getter.get_data2(900)
    # Ce = getter.estimate_Ce(PKA_energy_eV=1000, band_gap_eV=0.67, volume_A3=778688, fe=1)
    # print(f"Estimated Ce: {Ce:.3e} eV/K/A^3")

   
