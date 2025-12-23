import numpy as np
import matplotlib.pyplot as plt

class CascadePloter:
    def __init__(self):
        pass

    def plot_eph_results(self, datafile: str, figfile: str):
        Time, E_fric, E_rand, E_net_cum, T_e, T_a, Kin_a, Pot_a = np.loadtxt(datafile, skiprows=1, unpack=True)
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 10))

        axes[0].plot(Time, E_fric, label='E_fric')
        axes[0].plot(Time, E_rand, label='E_rand')
        axes[0].plot(Time, E_net_cum, label='E_net_cum')
        axes[0].set_ylabel('Energy (eV)')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(Time, T_e, label='T_e')
        axes[1].plot(Time, T_a, label='T_a')
        axes[1].set_ylabel('Temperature (K)')
        axes[1].legend()
        axes[1].grid(True)

        pot_shift = -np.min(Pot_a) + np.min(Kin_a)
        axes[2].plot(Time, Kin_a, label='Kin_a')
        axes[2].plot(Time, Pot_a+pot_shift, label='Pot_a')
        axes[2].plot(Time, Kin_a+Pot_a+pot_shift, label='Tot_a')
        axes[2].set_xlabel('Time (fs)')
        axes[2].set_ylabel('Energy (eV)')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        fig.savefig(figfile, dpi=300)

    def plot_thermo_results(self, datafile: str, figfile: str):
        Step, Time, Temp, E_kin, E_pot, Pressure = np.loadtxt(datafile, skiprows=1, unpack=True)
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
        axes[0].plot(Time, Temp, label='Temperature')
        axes[0].set_ylabel('Temperature (K)')
        axes[0].legend()
        axes[0].grid(True)

        pot_shift = -np.min(E_pot) + np.min(E_kin)
        axes[1].plot(Time, E_kin, label='E_kin')
        axes[1].plot(Time, E_pot+pot_shift, label='E_pot')
        axes[1].plot(Time, E_kin + E_pot+pot_shift, label='E_tot')
        axes[1].set_ylabel('Energy (eV)')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(Time, Pressure, label='Pressure')
        axes[2].set_xlabel('Time (fs)')
        axes[2].set_ylabel('Pressure (bar)')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        fig.savefig(figfile, dpi=300)
        

