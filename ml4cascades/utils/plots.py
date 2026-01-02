import numpy as np
import matplotlib.pyplot as plt

class CascadePloter:
    def __init__(self):
        pass

    def plot_eph_results(self, datafile: str, figfile: str):
        Time, E_fric, E_rand, E_net_cum, T_e, T_a, Kin_a, Pot_a = np.loadtxt(datafile, skiprows=1, unpack=True)
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))

        # axes[0].plot(Time, E_fric, label='E_fric')
        # axes[0].plot(Time, E_rand, label='E_rand')
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
        
        axes[2].set_xscale('log')
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
        # axes[1].plot(Time, E_kin, label='E_kin')
        axes[1].plot(Time, E_pot-E_pot[0], label='E_pot')
        # axes[1].plot(Time, E_kin + E_pot+pot_shift, label='E_tot')
        axes[1].set_ylabel('Energy (eV)')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(Time, Pressure, label='Pressure')
        axes[2].set_xlabel('Time (fs)')
        axes[2].set_ylabel('Pressure (bar)')
        axes[2].legend()
        axes[2].grid(True)

        for i in range(3):
            axes[i].set_xscale('log')
        plt.tight_layout()
        fig.savefig(figfile, dpi=300)
        

    def plot_mesh_Te(self, ni: int, nj: int, datafile: str, figfile: str):
        blocks = {}
        current_block = None
        current_data = []
        with open(datafile, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 1 and parts[0].isdigit(): # Block header: single integer
                    if current_block is not None:
                        blocks[current_block] = np.array(current_data, dtype=float)
                    current_block = int(parts[0])
                    current_data = []
                    continue
                if len(parts) >= 10 and parts[0].isdigit():    # Data line
                    current_data.append(parts)
        if current_block is not None:                      # Save last block
            blocks[current_block] = np.array(current_data, dtype=float)

        # plot last block
        last_block = max(blocks.keys())
        print(f"Plotting block {last_block}")
        data = blocks[last_block]
        T = np.zeros((ni, nj))
        k_slice = 1
        mask = data[:, 2] == k_slice
        slice_data = data[mask]
        for row in slice_data:
            i, j, Te = int(row[0]) - 1, int(row[1]) - 1, row[3]
            T[i, j] = Te
        
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(ni + 1)
        y = np.arange(nj + 1)
        pcm = ax.pcolormesh(
            x, y, T.T,    # transpose so i→x, j→y
            shading="flat"
        )
        plt.colorbar(pcm, ax=ax, label="Electron Temperature (K)")
        ax.set_xlabel("i")
        ax.set_ylabel("j")
        ax.set_title(f"T_e at {last_block} fs")
        ax.set_aspect("equal")
        plt.tight_layout()
        fig.savefig(figfile, dpi=300)


