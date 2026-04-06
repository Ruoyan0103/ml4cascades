import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os, math
from ovito.io import import_file

TICK_FONTSIZE = 14
LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 13

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

kB = 8.617333262145e-5      # Boltzmann constant, eV/K
class TurbogapCascadePlotter:
    def __init__(self):
        pass

    def plot_eph_results(self, datafile: str, figfile: str):
        Time, E_fric, E_rand, E_net_cum, T_e, T_a, Kin_a, Pot_a = np.loadtxt(datafile, skiprows=1, unpack=True)
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        Time_fs = Time*1000
        # axes[0].plot(Time_fs, E_fric+E_rand, label='E_fric')
        # axes[0].plot(Time_fs, E_rand, label='E_rand')
        pot_shift = -np.min(Pot_a) + np.min(Kin_a) 
        # axes[0].plot(Time_fs, E_net_cum+Kin_a+Pot_a+pot_shift, label='Tot_a')
        axes[0].plot(Time_fs, E_net_cum, label='E_net_cum')

        # axes[0].plot(Time_fs, E_net_cum+Kin_a+Pot_a+pot_shift, label='E_net_cum + Tot_a')
        axes[0].set_ylabel('Energy (eV)')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(Time_fs*1000, T_e, label='T_e')
        axes[1].plot(Time_fs*1000, T_a, label='T_a')
        axes[1].set_ylabel('Temperature (K)')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(Time_fs*1000, Kin_a, label='Kin_a')
        axes[2].plot(Time_fs*1000, Pot_a+pot_shift, label='Pot_a')
        axes[2].plot(Time_fs*1000, Kin_a+Pot_a+pot_shift, label='Tot_a')
        axes[2].set_xlabel('Time (fs)')
        axes[2].set_ylabel('Energy (eV)')
        axes[2].legend()
        axes[2].grid(True)
        
        # axes[2].set_xscale('log')
        plt.tight_layout()
        fig.savefig(figfile, dpi=300)

    def plot_thermo_results(self, datafile: str, figfile: str):
        Step, Time, Temp, E_kin, E_pot, Pressure = np.loadtxt(datafile, skiprows=1, unpack=True)
        Time_fs = Time*1000
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
        axes[0].plot(Time_fs*1000, Temp, label='Temperature')
        axes[0].set_ylabel('Temperature (K)')
        axes[0].legend()
        axes[0].grid(True)

        pot_shift = -np.min(E_pot) + np.min(E_kin)
        # axes[1].plot(Time_fs, E_kin, label='E_kin')
        axes[1].plot(Time_fs*1000, E_pot-E_pot[0], label='E_pot')
        # axes[1].plot(Time_fs, E_kin + E_pot+pot_shift, label='E_tot')
        axes[1].set_ylabel('Energy (eV)')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(Time_fs*1000, Pressure, label='Pressure')
        axes[2].set_xlabel('Time (fs)')
        axes[2].set_ylabel('Pressure (bar)')
        axes[2].legend()
        axes[2].grid(True)

        # for i in range(3):
        #     axes[i].set_xscale('log')
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

class LammpsCascadePlotter:
    def __init__(self):
        pass

    def plot_stopping_results(self, datafile: str, figfile: str, ecut: float=1.0):
        data = np.loadtxt(datafile, skiprows=1)
        step, Time, temp, pe, ke, etotal, ekmaxall, e_stopping_loss = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], data[:,7]
        Time_fs = Time*1000
        fig, axes = plt.subplots(2, 1, figsize=(6, 6))
        axes[0].plot(Time_fs, temp, label='Ta')
        axes[0].set_ylabel('Temperature (K)', fontsize=15)
        axes[0].legend(fontsize=15)
        axes[0].text(0.05, 0.9, f"Last 10 ps average Ta: {np.mean(temp[int(0.9*len(temp)):]):.2f} K", transform=axes[0].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        # axes[0].set_yscale('log')
        # add horizontal line for ekmaxall == 1, and a corresponding vertical line for the Time_fs when ekmaxall first drops below 1 eV
        # axes[0].axhline(ecut, color='gray', linestyle='--', linewidth=1)
        below_1_indices = np.where(ekmaxall < ecut)[0]
        if len(below_1_indices) > 0:
            first_below_1_Time_fs = Time_fs[below_1_indices[0]]
            # axes[0].axvline(first_below_1_Time_fs, color='gray', linestyle='--', linewidth=1)
            axes[1].axvline(first_below_1_Time_fs, color='gray', linestyle='--', linewidth=1)
            # axes[2].axvline(first_below_1_Time_fs, color='gray', linestyle='--', linewidth=1)

        init_etotal = etotal[0]
        energy_loss = [init_etotal - e for e in etotal]
        axes[1].plot(Time_fs*1000, energy_loss, label='Energy loss')
        axes[1].plot(Time_fs*1000, e_stopping_loss, label='E_stopping_loss')
        # axes[1].set_ylabel('Energy (eV)', fontsize=15)
        axes[1].legend(fontsize=15)
        axes[1].set_xlabel('Time (fs)', fontsize=15)
        

        # axes[2].plot(Time_fs, temp, label='Temperature')
        # axes[2].plot(Time_fs, border_temp, label='T_border')
        # axes[2].plot(Time_fs, inside_temp, label='T_inside')
        # axes[2].set_xlabel('Time_fs (ps)', fontsize=15)
        # axes[2].set_ylabel('Temperature (K)', fontsize=15)
        # axes[2].legend(fontsize=15)
        # axes[2].text(0.05, 0.9, f"Last 10 ps average Tborder: {np.mean(border_temp[int(0.9*len(border_temp)):]):.2f} K", transform=axes[2].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        # axes[2].text(0.05, 0.8, f"Last 10 ps average Tinside: {np.mean(inside_temp[int(0.9*len(inside_temp)):]):.2f} K", transform=axes[2].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        # axes[3].plot(Time_fs, inside_etotal, label='E_inside')
        # axes[3].set_xlabel('Time_fs (ps)', fontsize=15)
        # axes[3].set_ylabel('Energy (eV)', fontsize=15)
        # axes[3].set_xscale('log')
        # axes[3].legend(fontsize=15)

        # axes[4].plot(Time_fs, border_etotal, label='E_border')
        # axes[4].set_xlabel('Time_fs (ps)', fontsize=15)
        # axes[4].set_ylabel('Energy (eV)', fontsize=15)
        # axes[4].set_xscale('log')
        # axes[4].legend(fontsize=15)

        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
            ax.set_xscale('log')
        
        plt.tight_layout()
        fig.savefig(figfile, dpi=300)
  
    def plot_eph_results(self, datafile1: str, datafile2: str, figfile: str):
        data = np.loadtxt(datafile1, skiprows=1)
        # step, Time_fs, Ta, Te = data[:,0], data[:,1], data[:,2], data[:,3]
        # Ee, E_random, E_friction = data[:,4], data[:,5], data[:,6]
        # dT_e, ddT_e, S_e = data[:,7], data[:,8], data[:,9]
        step, Time, Ta, transferred_eng, Te = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4] 
        Time_fs = Time*1000
        # Tborder, Tinside = data[:,5], data[:,6]
        # T_br, T_in = data[:,10], data[:,11]
        data = np.loadtxt(datafile2, skiprows=1)
        step, Epot, Ekin, Etotal = data[:,0], data[:,3], data[:,4], data[:,5]

        fig, axes = plt.subplots(4, 1, figsize=(8, 12))
        axes[0].plot(Time_fs, transferred_eng, color='red', label='E_transfer')
        # axes[0].plot(Time_fs, E_friction+E_random, color='blue', label='E_transfer2')
        axes[0].set_ylabel('E_transfer (eV)', fontsize=15)
        axes[0].legend(fontsize=15)
        axes[0].grid(True)

        #axes[1].plot(Time_fs, T_in, label='Ta-inside')
        #axes[1].plot(Time_fs, T_br, label='Ta-border')
        axes[1].plot(Time_fs, Ta, label='Ta')
        axes[1].plot(Time_fs, Te, label='Te')
        #axes[1].plot(Time_fs[1:], Te_after[1:], label='Te_after')
        #axes[1].plot(Time_fs[1:], Te_before[1:], label='Te_before')
        axes[1].set_ylabel('Temperature (K)', fontsize=15)
        axes[1].legend(fontsize=15)
        axes[1].grid(True)
        # add text, 2 decimal
        #axes[1].text(0.05, 0.9, f"Average Ta-inside: {np.mean(T_in[int(0.9*len(T_in)):]):.2f} K", transform=axes[1].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))  
        #axes[1].text(0.05, 0.8, f"Average Ta-border: {np.mean(T_br[int(0.9*len(T_br)):]):.2f} K", transform=axes[1].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        axes[1].text(0.05, 0.9, f"Average Ta: {np.mean(Ta[int(0.9*len(Ta)):]):.2f} K", transform=axes[1].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        axes[1].text(0.05, 0.8, f"Average Te: {np.mean(Te[int(0.9*len(Te)):]):.2f} K", transform=axes[1].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
        #pot_shift = -np.min(Epot) + np.min(Ekin)
        #axes[2].plot(Time_fs, Ekin, label='Ekin')
        #axes[2].plot(Time_fs, Epot+pot_shift, label='Epot')
        axes[2].plot(Time_fs, Etotal, label='Etotal(Atomic)')
        axes[2].plot(Time_fs, Etotal+transferred_eng, label='Etotal(Electronic+Atomic)')
        axes[2].text(0.05, 0.9, f"Energy loss: {abs(min(Etotal+transferred_eng)-max(Etotal+transferred_eng)):.2f} eV", transform=axes[2].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))  
        # record these two columns into a text file  
        # with open(figfile.replace('.png', '_dT_ddT.txt'), 'w') as fout:
        #     fout.write('Time_fs(ps) dT_e(eV/Angstrom^3/ps) ddT_e(eV/Angstrom^3/ps)\n')
        #     for t, dte, ddte in zip(Time_fs[1:], dT_e[1:], ddT_e[1:]):
        #         fout.write(f"{t} {dte} {ddte}\n")

        # axes[2].plot(Time_fs, S_e, label='S_e')
        # axes[2].set_yscale('log')
      
        # axes[2].set_xlabel('Time_fs (ps)', fontsize=15)
        # axes[2].set_ylabel('Power density (eV/Å³/ps)', fontsize=15)
        axes[2].set_ylabel('Energy (eV)', fontsize=15)
        axes[2].legend(fontsize=15)
        axes[2].grid(True)
        
        axes[3].plot(Time_fs, Epot-Epot[0], label='Epot')
        axes[3].set_xlabel('Time (fs)', fontsize=15)
        axes[3].set_ylabel('Energy (eV)', fontsize=15)
        axes[3].legend(fontsize=15)
        axes[3].grid(True)

        # increase ticks size
        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.set_xscale('log')
        
        # axes[1].set_xscale('log')
        # fig, ax = plt.subplots(figsize=(6, 6))
        # ax.plot(Time_fs, Ta, label='Ta')
        # ax.plot(Time_fs, Te, label='Te')
        # ax.set_ylabel('Temperature (K)')
        # ax.set_xlabel('Time (fs)')
        # ax.legend()
        # ax.grid(True)
        plt.tight_layout()
        fig.savefig(figfile, dpi=300)


    def plot_eph_coupling(self, 
                          folder_list: list[Path],
                          label_list: list[str],
                          figfile: str,
                          max_time: float=1.5):
        fig, ax = plt.subplots(figsize=(8, 12))
        for folder, label in zip(folder_list, label_list): 
            eng_file = folder / 'eng.out'
            thermo_file = folder / 'thermo.out'
            data = np.loadtxt(eng_file, skiprows=1)
            Time, transferred_eng = data[:,1], data[:,3]
            data = np.loadtxt(thermo_file, skiprows=1)
            Ekin = data[:,4]

            mask = Time <= max_time
            Time = Time[mask]
            Time_fs = Time*1000
            transferred_eng = transferred_eng[mask]
            Ekin = Ekin[mask]

            dEe_dt = np.gradient(transferred_eng, Time)
            tau = 2*Ekin/np.abs(dEe_dt)
            ax.plot(Time, tau, label=f'{label}')

        ax.set_ylabel('Coupling Time (ps)', fontsize=LABEL_FONTSIZE)
        ax.set_xlabel('Time (ps)', fontsize=LABEL_FONTSIZE)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
        # ax.set_xscale('log')
        ax.set_yscale('log')
        plt.tight_layout()
        fig.savefig(figfile, dpi=300)


    def get_grid_Ta_Te(self, 
                       dump_file: str,  
                       T_out_folder: str,
                       new_dump_file: str,
                       new_tout_file: str,
                       avg_Ta_file: str,
                       avg_Te_file: str,
                       flag: int):
        # ----------------------------- Delete existing files -------------------------
        if flag == 1 or flag == 5:
            if os.path.exists(new_dump_file):
                os.remove(new_dump_file)
        elif flag == 2 or flag == 5:
            if os.path.exists(new_tout_file):
                os.remove(new_tout_file)
        elif flag == 3 or flag == 5:
            if os.path.exists(avg_Ta_file):
                os.remove(avg_Ta_file)
        elif flag == 4 or flag == 5:
            if os.path.exists(avg_Te_file):
                os.remove(avg_Te_file)
        
        # ----------------------------- Get header info -------------------------
        with open(dump_file, 'r') as fin:
            dump_lines = fin.readlines()
        header = dump_lines[:8]

        # ----------------------------- Process T_out files -------------------------
        tout_files = [os.path.join(T_out_folder, f) for f in os.listdir(T_out_folder)]
        tout_files.sort(key=lambda f: int(os.path.basename(f).split('_')[-1]))
        epsilon = 1e-9

        # grids from electronic system
        data = np.loadtxt(tout_files[0], skiprows=1)
        initial_x, initial_y, initial_z = data[0, 0], data[0, 1], data[0, 2]
        # only one axis values are changing
        mask = (data[:, 0] == initial_x) & (data[:, 1] == initial_y)
        slice_data = data[mask]
        z_list = slice_data[:, 2]
        step_z = z_list[1] - z_list[0]
        mask = (data[:, 0] == initial_x) & (data[:, 2] == initial_z)
        slice_data = data[mask]
        y_list = slice_data[:, 1]
        step_y = y_list[1] - y_list[0]
        mask = (data[:, 1] == initial_y) & (data[:, 2] == initial_z)
        slice_data = data[mask]
        x_list = slice_data[:, 0]
        step_x = x_list[1] - x_list[0]
        min_x, min_y, min_z = min(x_list), min(y_list), min(z_list)
        max_x, max_y, max_z = max(x_list), max(y_list), max(z_list)
        x_length = max_x - min_x + step_x
        y_length = max_y - min_y + step_y
        z_length = max_z - min_z + step_z

        x_grids = math.floor(x_length/step_x + epsilon)
        y_grids = math.floor(y_length/step_y + epsilon)
        z_grids = math.floor(z_length/step_z + epsilon)

        with open(new_tout_file, 'w') as fout:
            for Time_fsstep, tout_file in enumerate(tout_files):
                print(f"======= Electronic system, processing Time_fsstep: {Time_fsstep} ========")
                avg_temp = 0
                if flag == 2 or flag == 5:
                    fout.write(header[0])  # ITEM: Time_fsSTEP
                    fout.write(f"{Time_fsstep}\n") 
                    fout.write(header[2])  # ITEM: NUMBER OF ATOMS
                    fout.write(f"{x_grids * y_grids * z_grids}\n")
                    fout.write(header[4])  # ITEM: BOX BOUNDS pp pp pp
                    fout.write(f"{min_x} {max_x+step_x}\n")
                    fout.write(f"{min_y} {max_y+step_y}\n")
                    fout.write(f"{min_z} {max_z+step_z}\n")
                    new_line = 'ITEM: ATOMS grid_id x y z Te\n'
                    fout.write(new_line)
                    with open(tout_file, 'r') as fin:
                        lines = fin.readlines()
                        for idx, line in enumerate(lines[1:]):
                            x, y, z, Te = line.strip().split()
                            fout.write(f'{idx} {float(x)+step_x/2} {float(y)+step_y/2} {float(z)+step_z/2} {Te}\n') # the center of the grid
                            avg_temp += float(Te) / (x_grids * y_grids * z_grids)
                if flag == 4 or flag == 5:
                    with open(avg_Te_file, 'a') as favg:
                        favg.write(f"{Time_fsstep} {avg_temp}\n")

        # ----------------------------- Process dump file -------------------------
        num_atoms = int(header[3].strip())
        atomic_xlo, atomic_xhi = header[5].split()
        atomic_ylo, atomic_yhi = header[6].split()
        atomic_zlo, atomic_zhi = header[7].split()

        atomic_x_grids = math.floor((float(atomic_xhi) - float(atomic_xlo)) / step_x + epsilon)
        atomic_y_grids = math.floor((float(atomic_yhi) - float(atomic_ylo)) / step_y + epsilon)
        atomic_z_grids = math.floor((float(atomic_zhi) - float(atomic_zlo)) / step_z + epsilon)

        shift_x = float(atomic_xlo) - min_x 
        shift_y = float(atomic_ylo) - min_y
        shift_z = float(atomic_zlo) - min_z

        shift_grid_x = math.floor(shift_x / step_x + epsilon)
        shift_grid_y = math.floor(shift_y / step_y + epsilon)
        shift_grid_z = math.floor(shift_z / step_z + epsilon)

        block_length = num_atoms + 9  
        num_blocks = len(dump_lines) // block_length
        for block_idx in range(1, num_blocks):
            grids_val = [[[-epsilon for _ in range(z_grids)] for _ in range(y_grids)] for _ in range(x_grids)]  # grid out of atomic region will have -epsilon value
            grids_cnt = [[[0 for _ in range(z_grids)] for _ in range(y_grids)] for _ in range(x_grids)]
            start_line = block_idx * block_length
            end_line = start_line + block_length
            block_lines = dump_lines[start_line:end_line]

            # ----------20260401-----------#
            grids_density = [[[-epsilon for _ in range(z_grids)] for _ in range(y_grids)] for _ in range(x_grids)]
            grids_coupling = [[[-epsilon for _ in range(z_grids)] for _ in range(y_grids)] for _ in range(x_grids)]
            # ----------20260401-----------#

            print(f"======= Atomic system, processing block: {block_idx}/{num_blocks-1} ========")
            for line in block_lines[9:]:
                data = line.strip().split()
                x = float(data[2])
                y = float(data[3])
                z = float(data[4])
                ek = float(data[9])
                # ----------20260401-----------#
                site_density = float(data[11])
                site_coupling = float(data[12])
                # ----------20260401-----------#

                ix = min(math.floor((x - min_x) / step_x + epsilon), shift_grid_x + atomic_x_grids - 1)
                iy = min(math.floor((y - min_y) / step_y + epsilon), shift_grid_y + atomic_y_grids - 1)
                iz = min(math.floor((z - min_z) / step_z + epsilon), shift_grid_z + atomic_z_grids - 1)
                ix = max(ix, shift_grid_x)
                iy = max(iy, shift_grid_y)
                iz = max(iz, shift_grid_z)
                grids_val[ix][iy][iz] += ek
                grids_cnt[ix][iy][iz] += 1
                # ----------20260401-----------#
                grids_density[ix][iy][iz] += site_density
                grids_coupling[ix][iy][iz] += site_coupling
                # ----------20260401-----------#

            # convert to temperature, average temp for one grid
            for ix in range(x_grids):
                for iy in range(y_grids):
                    for iz in range(z_grids):
                        factor = 2.0 / (3.0 * kB)
                        grids_val[ix][iy][iz] *= factor/max(grids_cnt[ix][iy][iz], 1)
                        # ----------20260401-----------#
                        grids_density[ix][iy][iz] /= max(grids_cnt[ix][iy][iz], 1)
                        grids_coupling[ix][iy][iz] /= max(grids_cnt[ix][iy][iz], 1)
                        # ----------20260401-----------#

            # with open(cnt_atom_file, 'a') as fout:
            #     fout.write(f"block: {block_idx} \n")
            #     fout.write("grid_id atom_count\n")
            #     for ix in range(x_grids):
            #         for iy in range(y_grids):
            #             for iz in range(z_grids):
            #                 grid_id = iz * (x_grids * y_grids) + iy * x_grids + ix
            #                 fout.write(f"{grid_id} {grids_cnt[ix][iy][iz]} \n")

            # average temperature for the whole system
            if flag == 3 or flag == 5:
                avg_temp = 0
                for ix in range(x_grids):
                    for iy in range(y_grids):
                        for iz in range(z_grids):
                            avg_temp += grids_val[ix][iy][iz] / (x_grids * y_grids * z_grids)
                        with open(avg_Ta_file, 'a') as favg:
                            favg.write(f"{block_idx} {avg_temp}\n")
            
            # write to newe dump file
            if flag == 1 or flag == 5: 
                with open(new_dump_file, 'a') as fout:
                    fout.write(header[0])  # ITEM: Time_fsSTEP
                    fout.write(f"{block_idx}\n")
                    fout.write(header[2])  # ITEM: NUMBER OF ATOMS
                    fout.write(f"{x_grids * y_grids * z_grids}\n")
                    fout.write(header[4])  # ITEM: BOX BOUNDS pp pp pp
                    fout.write(f"{min_x} {max_x+step_x}\n")
                    fout.write(f"{min_y} {max_y+step_y}\n")
                    fout.write(f"{min_z} {max_z+step_z}\n")
                    new_line = 'ITEM: ATOMS grid_id idx idy idz x y z Ta density coupling\n'
                    fout.write(new_line)
                    for iz in range(z_grids):
                        for iy in range(y_grids):
                            for ix in range(x_grids):
                                grid_id = iz * (x_grids * y_grids) + iy * x_grids + ix
                                x_center = min_x + ix * step_x + step_x / 2
                                y_center = min_y + iy * step_y + step_y / 2
                                z_center = min_z + iz * step_z + step_z / 2
                                Ta = grids_val[ix][iy][iz]
                                # ----------20260401-----------#
                                density = grids_density[ix][iy][iz]
                                coupling = grids_coupling[ix][iy][iz]
                                # ----------20260401-----------#
                                new_line = f"{grid_id} {ix} {iy} {iz} {x_center} {y_center} {z_center} {Ta:.6f} {density:.6f} {coupling:.6f}\n"
                                fout.write(new_line)
    
    def get_extreme_Te_Ta(self, 
                         tout_file: str,
                         dump_file: str,
                         frame_idx_list: list[int],
                         Time_fs_list: list[float],
                         grid_list: list[int],
                         electron_grid_list: list[int],
                         figfile: str):
        assert len(frame_idx_list) == len(Time_fs_list)
        hottest_Te = []
        coldest_Te = []
        hottest_Ta = []
        coldest_Ta = []
        avg_Ta = []
        avg_Te = []
        len_avg_grid_ta = len(grid_list)
        len_avg_grid_te = len(electron_grid_list)
        higher_Te_than_Ta = []
        for frame_idx in frame_idx_list:
            elec_pipeline = import_file(tout_file)
            elec_data = elec_pipeline.compute(frame_idx)
            elec_grids = elec_data.particles
            te_list = elec_grids['te']

            atom_pipeline = import_file(dump_file)
            atom_data = atom_pipeline.compute(frame_idx)
            atom_grids = atom_data.particles
            ta_list = atom_grids['ta']

            avg_ta_val = 0
            avg_te_val = 0
            for grid_id in grid_list:
                avg_ta_val += atom_grids['ta'][grid_id] / len_avg_grid_ta
            for grid_id in electron_grid_list:
                avg_te_val += elec_grids['te'][grid_id] / len_avg_grid_te

            avg_Ta.append(avg_ta_val)
            avg_Te.append(avg_te_val)
            descend_Te_grid_ids = np.argsort(-te_list)
            hottest_Te.append(te_list[descend_Te_grid_ids[0]])
            coldest_Te.append(te_list[descend_Te_grid_ids[-1]])
            print('Te: ')
            print(descend_Te_grid_ids[:8])

            ta_array = np.array(ta_list)
            positive_ids = np.where(ta_array > 0)[0]
            positive_values = ta_array[positive_ids]
            descend_ids = positive_ids[np.argsort(-positive_values)]
            hottest_Ta.append(ta_array[descend_ids[0]])
            coldest_Ta.append(ta_array[descend_ids[-1]])
            print('Ta: ')
            print(descend_ids[:8])

            cnt = 0
            high_avg_te = 0
            low_avg_ta = 0
            for id in positive_ids:
                ta = ta_list[id]
                te = te_list[id]
                if te > ta:
                    high_avg_te += te
                    low_avg_ta += ta
                    if id in grid_list:
                        higher_Te_than_Ta.append({frame_idx: id})
            # higher_Te_than_Ta.append(cnt)

        figure, ax = plt.subplots(figsize=(10, 6))
        ax.plot(Time_fs_list, hottest_Te, label='Highest Te', linestyle='--', marker='o', color='orange')
        ax.plot(Time_fs_list, coldest_Te, label='Lowest Te', linestyle='--',marker='D', color='blue')
        ax.set_ylabel('Temperature (K)', fontsize=LABEL_FONTSIZE)

        ax.plot(Time_fs_list, hottest_Ta, label='Highest Ta', marker='o', color='orange')
        ax.plot(Time_fs_list, coldest_Ta, label='Lowest Ta', marker='D', color='blue')

        # ax.plot(Time_fs_list, avg_Ta, label='Avg Ta', marker='^', color='green')
        # ax.plot(Time_fs_list, avg_Te, label='Avg Te', linestyle='--', marker='^', color='green')

        print(higher_Te_than_Ta)
        ax.set_xticks(Time_fs_list)
        ax.set_xticklabels(Time_fs_list, fontsize=TICK_FONTSIZE)
        # ax.set_yscale('log')
        # ax.axhline(y=300, color='red', label='300 K')
        ax.set_xlabel('Time (fs)', fontsize=LABEL_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.grid(True)
        plt.tight_layout()
        figure.savefig(figfile, dpi=300)

    def get_extreme_Ta(self, 
                       dump_file: str,
                       frame_idx_list: list[int],
                       Time_fs_list: list[float],
                       figfile: str):
        assert len(frame_idx_list) == len(Time_fs_list)
        hottest = [] 
        coldest = []
        for frame_idx in frame_idx_list:
            all_pipeline = import_file(dump_file)
            data = all_pipeline.compute(frame_idx)
            grids = data.particles
            ta_list = grids['ta']
            decend_T_grid_ids = np.argsort(-ta_list)
            ascend_T_grid_ids = np.argsort(ta_list)
            hottest.append(ta_list[decend_T_grid_ids[0]])
            coldest.append(ta_list[ascend_T_grid_ids[0]])
            # for i in range(1):
            #     print(sorted_grid_ids[i], te_list[sorted_grid_ids[i]], grids.positions[sorted_grid_ids[i]])
        figure, ax = plt.subplots(figsize=(6, 6))
        ax.plot(Time_fs_list, hottest, label='Hottest', marker='o')
        ax.plot(Time_fs_list, coldest, label='Coldest', marker='D')
        ax.axhline(y=300, color='red', linestyle='--', label='300 K')
        ax.set_xlabel('Time (fs)', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Ta (K)', fontsize=LABEL_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.grid(True)
        plt.tight_layout()
        figure.savefig(figfile, dpi=300)

    def plot_te_ta_along_x(self, 
                           tout_file: str, 
                           dump_file: str,
                           frame_idx_list: list[int],
                           Time_fs_list: list[float],
                           grid_list: list[int],
                           electron_grid_list: list[int],
                           figfile: str,
                           tout_file2: str=None,
                           dump_file2: str=None):
        # list of colors for different frames, the same length as frame_idx_list
        length = len(frame_idx_list)
        colors = plt.cm.viridis(np.linspace(0, 1, length+1))  
        min_limit = 10000
        max_limit = 0
        min_te_limit = 10000
        max_te_limit = 0
        min_ta_limit = 10000
        max_ta_limit = 0
        fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharey=False)
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        for i, frame_idx in enumerate(frame_idx_list):
            elec_pipeline = import_file(tout_file)
            elec_data = elec_pipeline.compute(frame_idx)
            elec_grids = elec_data.particles
            atmo_pipeline = import_file(dump_file)
            atmo_data = atmo_pipeline.compute(frame_idx)
            atmo_grids = atmo_data.particles

            te_list = []
            ta_list = []
            x_list = []
            electron_x_list = []
            for grid_id in grid_list:
                x_list.append(elec_grids.positions[grid_id][0])
                ta_list.append(atmo_grids['ta'][grid_id])
            for grid_id in electron_grid_list:
                te_list.append(elec_grids['te'][grid_id])
                electron_x_list.append(elec_grids.positions[grid_id][0])

            if tout_file2 is not None and dump_file2 is not None:
                elec_pipeline2 = import_file(tout_file2)
                elec_data2 = elec_pipeline2.compute(frame_idx)
                elec_grids2 = elec_data2.particles
                atmo_pipeline2 = import_file(dump_file2)
                atmo_data2 = atmo_pipeline2.compute(frame_idx)
                atmo_grids2 = atmo_data2.particles

                te_list2 = []
                ta_list2 = []
                x_list2 = []
                electron_x_list2 = []
                for grid_id in grid_list:
                    x_list2.append(atmo_grids2.positions[grid_id][0])
                    ta_list2.append(atmo_grids2['ta'][grid_id])
                for grid_id in electron_grid_list:  
                    te_list2.append(elec_grids2['te'][grid_id])
                    electron_x_list2.append(elec_grids2.positions[grid_id][0])
            
            ax[0].plot(x_list, ta_list, color=colors[i], marker='D', markersize=3, label=f'{Time_fs_list[i]} fs')
            ax[1].plot(electron_x_list, te_list, linestyle='--', color=colors[i], marker='D', markersize=3, label=f'{Time_fs_list[i]} fs')

            ax2.plot(x_list, ta_list, color=colors[i], marker='D', markersize=3, label=f'{Time_fs_list[i]} fs Ta')
            ax2.plot(electron_x_list, te_list, linestyle='--',color=colors[i], marker='D', markersize=3, label=f'{Time_fs_list[i]} fs Te')

            if tout_file2 is not None and dump_file2 is not None:
                ax[0].plot(
                    electron_x_list2, te_list2,
                    linestyle='None',
                    marker='o',
                    markersize=4,
                    markerfacecolor='none',
                    color=colors[i],
                )
                ax[1].plot(
                    x_list2, ta_list2,
                    linestyle='None',
                    marker='o',
                    markerfacecolor='none',
                    markersize=4,
                    color=colors[i],
                )

                ax2.plot(
                    electron_x_list2, te_list2,
                    linestyle='None',
                    marker='o',
                    markersize=4,
                    markerfacecolor='none',
                    color=colors[i],
                )
                ax2.plot(
                    x_list2, ta_list2,
                    linestyle='None',
                    marker='o',
                    markerfacecolor='none',
                    markersize=4,
                    color=colors[i],
                )

            # for sharey limit 
            min_limit = min(min(te_list), min(ta_list), min_limit, 295) 
            max_limit = max(max(te_list), max(ta_list), max_limit)

            min_te_limit = min(min(te_list), min_te_limit, 295)
            max_te_limit = max(max(te_list), max_te_limit,305)
            min_ta_limit = min(min(ta_list), min_ta_limit, 295)
            max_ta_limit = max(max(ta_list), max_ta_limit, 305)
            share_min_limit = min(min_te_limit, min_ta_limit)
            share_max_limit = max(max_te_limit, max_ta_limit)

        # ax[0].set_ylim(share_min_limit, share_max_limit)
        # ax[1].set_ylim(share_min_limit, share_max_limit)
        ax[0].set_ylim(min_ta_limit, max_ta_limit)
        ax[1].set_ylim(min_te_limit, max_te_limit)
        ax[1].set_ylabel('')

        # set a horizontal line at y=300K
        ax[0].axhline(y=300, color='red', label='300 K')
        ax[1].axhline(y=300, color='red', label='300 K')
        ax2.axhline(y=300, color='red', label='300 K')
    
        ax[0].legend(fontsize=LEGEND_FONTSIZE)
        # ax[1].legend()
        ax[0].set_xlabel('X Position (Angstrom)', fontsize=LABEL_FONTSIZE)
        ax[1].set_xlabel('X Position (Angstrom)', fontsize=LABEL_FONTSIZE)
        ax[0].set_ylabel('Temperature (K)', fontsize=LABEL_FONTSIZE)

        ax[0].set_title(f'Atomic system ')
        ax[1].set_title(f'Electronic system ')

        ax[0].grid(True)
        ax[1].grid(True)
        for i in range(2):
            ax[i].tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)


        ax2.set_xlabel('X Position (Angstrom)', fontsize=LABEL_FONTSIZE)
        ax2.set_ylabel('Temperature (K)', fontsize=LABEL_FONTSIZE)
        ax2.legend(fontsize=LEGEND_FONTSIZE)
        ax2.grid(True)
        ax2.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)

        # set y axis to log scale
        # ax[0].set_yscale('log')
        # ax[1].set_yscale('log')
  

        fig.subplots_adjust(hspace=0)
        fig.tight_layout()
        fig2.tight_layout()
        fig.savefig(figfile, dpi=300)
        fig2.savefig(figfile.replace('.png', '_combined.png'), dpi=300)

    def plot_xy_heatmap(self, 
                    tout_file: str, 
                    dump_file: str,
                    frame_idx_list: list[int],
                    Time_fs_list: list[float],
                    figfile1: str,
                    figfile2: str,
                    figfile3: str,
                    z: int,
                    gridx: int,
                    gridy: int,
                    border: int):

        elec_pipeline = import_file(tout_file)
        atom_pipeline = import_file(dump_file)

        te_maps = []      # full Te maps
        ta_maps = []      # inner Ta maps
        tdiff_maps = []   # inner (Te - Ta) maps

        starting_grid = z * gridx * gridy
        ending_grid = (z + 1) * gridx * gridy

        nx = gridx - 2 * border
        ny = gridy - 2 * border

        if nx <= 0 or ny <= 0:
            raise ValueError("border is too large for the grid size")

        for i, frame_idx in enumerate(frame_idx_list):
            elec_data = elec_pipeline.compute(frame_idx)
            elec_grids = elec_data.particles
            atom_data = atom_pipeline.compute(frame_idx)
            atom_grids = atom_data.particles

            # ---- full Te map ----
            te_list_full = []
            for grid_id in range(starting_grid, ending_grid):
                te_list_full.append(elec_grids['te'][grid_id])

            te_map_full = np.array(te_list_full).reshape(gridy, gridx)
            te_maps.append(te_map_full)

            # ---- inner Ta map ----
            ta_list_inner = []
            for y in range(border, gridy - border):
                for x in range(border, gridx - border):
                    grid_id = z * gridx * gridy + y * gridx + x
                    ta_list_inner.append(atom_grids['ta'][grid_id])

            ta_map_inner = np.array(ta_list_inner).reshape(ny, nx)
            ta_maps.append(ta_map_inner)

            # ---- inner Te map for diff ----
            te_map_inner = te_map_full[border:gridy - border, border:gridx - border]

            # ---- diff on inner region only ----
            tdiff_map = te_map_inner - ta_map_inner
            tdiff_maps.append(tdiff_map)

        # ---- plot Te (full grid) ----
        te_vmin = min(m.min() for m in te_maps)
        te_vmax = max(m.max() for m in te_maps)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        axes = axes.ravel()

        images = []
        for i, ax in enumerate(axes[:len(te_maps)]):
            im = ax.imshow(
                te_maps[i],
                origin='lower',
                cmap='coolwarm',
                vmin=300,
                vmax=2000,
                aspect='equal'
            )
            images.append(im)
            ax.set_title(f"t = {Time_fs_list[i]} fs", fontsize=LABEL_FONTSIZE-2)
            if i % 2 == 0:
                ax.set_ylabel("Y", fontsize=LABEL_FONTSIZE-1)
            if i >= 2:
                ax.set_xlabel("X", fontsize=LABEL_FONTSIZE-1)
            ax.tick_params(labelsize=TICK_FONTSIZE-1)

        for j in range(len(te_maps), len(axes)):
            axes[j].axis('off')

        fig.text(
            0.02, 0.99, 
            f"z = {z}",
            fontsize=LABEL_FONTSIZE,
            ha='left',
            va='top'
        )
        cbar = fig.colorbar(images[0], ax=axes, label='Te (K)', extend='both')
        cbar.set_label('Te (K)', fontsize=LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE-1)  
        fig.savefig(figfile1, dpi=300)
        plt.close(fig)

        # ---- plot Ta (inner grid) ----
        ta_vmin = min(m.min() for m in ta_maps)
        ta_vmax = max(m.max() for m in ta_maps)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        axes = axes.ravel()

        images = []
        for i, ax in enumerate(axes[:len(ta_maps)]):
            im = ax.imshow(
                ta_maps[i],
                origin='lower',
                cmap='coolwarm',
                vmin=300,
                vmax=2000,
                aspect='equal'
            )
            images.append(im)
            ax.set_title(f"t = {Time_fs_list[i]} fs", fontsize=LABEL_FONTSIZE-2)
            if i % 2 == 0:
                ax.set_ylabel("Y", fontsize=LABEL_FONTSIZE-1)
            if i >= 2:
                ax.set_xlabel("X", fontsize=LABEL_FONTSIZE-1)
            ax.tick_params(labelsize=TICK_FONTSIZE-1)

        for j in range(len(te_maps), len(axes)):
            axes[j].axis('off')

        fig.text(
            0.02, 0.99, 
            f"z = {z}",
            fontsize=LABEL_FONTSIZE,
            ha='left',
            va='top'
        )
        cbar = fig.colorbar(images[0], ax=axes, label='Ta (K)', extend='both')
        cbar.set_label('Ta (K)', fontsize=LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE-1)  
        fig.savefig(figfile2, dpi=300)
        plt.close(fig)

        # ---- plot Te - Ta (inner grid only) ----
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        axes = axes.ravel()

        images = []
        for i, ax in enumerate(axes[:len(tdiff_maps)]):
            im = ax.imshow(
                tdiff_maps[i],
                origin='lower',
                cmap='coolwarm',
                vmin=-1000,
                vmax=1000,
                aspect='equal'
            )
            images.append(im)
            ax.set_title(f"t = {Time_fs_list[i]} fs", fontsize=LABEL_FONTSIZE-2)
            if i % 2 == 0:
                ax.set_ylabel("Y", fontsize=LABEL_FONTSIZE-1)
            if i >= 2:
                ax.set_xlabel("X", fontsize=LABEL_FONTSIZE-1)
            ax.tick_params(labelsize=TICK_FONTSIZE-1)

        for j in range(len(tdiff_maps), len(axes)):
            axes[j].axis('off')

        fig.text(
            0.02, 0.99, 
            f"z = {z}",
            fontsize=LABEL_FONTSIZE,
            ha='left',
            va='top'
        )
        cbar = fig.colorbar(images[0], ax=axes, label='Te - Ta (K)', extend='both')
        cbar.set_label('Te - Ta (K)', fontsize=LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE-1)  
        fig.savefig(figfile3, dpi=300)
        plt.close(fig)

























































        



          

        









