import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os, math
from ovito.io import import_file
from matplotlib.colors import SymLogNorm, LogNorm, PowerNorm

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
        segments = []                                                                                                                                                                                           
        current_segment = []                                                                                                                                                                                    
        max_time = 700                                                                                                                                                                                                     
        with open(datafile, 'r') as f:                                                                                                                                                                              
            for line in f:                                                                                                                                                                                      
                line = line.strip()                                                                                                                                                                             
                if '### reset stopping_power' in line:                                                                                                                                                          
                    if current_segment:                                                                                                                                                                         
                        segments.append(current_segment)                                                                                                                                                        
                    current_segment = []                                                                                                                                                                        
                elif line.startswith('#') or not line:                                                                                                                                                          
                    continue
                else:                                                                                     
                    parts = line.split()              
                    if len(parts) == 7:                                                                   
                        current_segment.append([float(x) for x in parts])
        if current_segment:                                                                               
            segments.append(current_segment)
        all_rows = []                                 
        sp_offset = 0.0
        for segment in segments:
            seg = np.array(segment)           
            _, first_idx = np.unique(seg[:, 1], return_index=True)                                        
            seg = seg[np.sort(first_idx)]
            if all_rows:
                last_time = all_rows[-1][1]                                                               
                seg = seg[seg[:, 1] > last_time]
            if seg.size == 0:                         
                continue
            seg[:, 6] += sp_offset                                                                        
            sp_offset = seg[-1, 6]
            all_rows.extend(seg.tolist())                                        
        data = np.array(all_rows) if all_rows else np.empty((0, 7))                                       
        if data.ndim == 1:
            data = data.reshape(1, -1)                                                                    
        mask = data[:, 1] <= max_time                                                                     
        data = data[mask]                                                                                 
        Time, temp, pe, ke, etotal, e_stopping_loss = (
            data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6]
        )     
        Time_fs = Time*1000
        fig, axes = plt.subplots(3, 1, figsize=(6, 10))
        axes[0].plot(Time_fs, temp, label='Ta')
        axes[0].set_ylabel('Temperature (K)', fontsize=15)
        axes[0].legend(fontsize=15)
        axes[0].text(0.05, 0.9, f"Last 10 ps average Ta: {np.mean(temp[int(0.9*len(temp)):]):.2f} K", transform=axes[0].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        init_etotal = etotal[0]
        energy_loss = [init_etotal - e for e in etotal]
        axes[1].plot(Time_fs, energy_loss, label='Energy loss')
        axes[1].plot(Time_fs, e_stopping_loss, label='E_stopping_loss')
        axes[1].set_ylabel('Energy (eV)', fontsize=15)
        axes[1].legend(fontsize=15)
        axes[1].set_xlabel('Time (fs)', fontsize=15)
        
        axes[2].plot(Time_fs, pe, label='potential energy')
        axes[2].set_xlabel('Time (fs)', fontsize=15)
        axes[2].set_ylabel('Potential Energy (eV)', fontsize=15)
        axes[2].legend(fontsize=15)
        
        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
            ax.set_xscale('log')
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        fig.savefig(figfile, dpi=300)
  
    def plot_eph_results(self, datafile: str, figfile: str):
        # thermo.out: $(step) $(time) $(temp) $(pe) $(ke) $(etotal) $(f_friction[1]) $(f_friction[2])
        #data = np.loadtxt(datafile, skiprows=1)
        segments = []
        current_segment = []
        max_time = 700
        with open(datafile, 'r') as f:
            for line in f:
                line = line.strip()
                if '### reset eph_power' in line:
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = []
                elif line.startswith('#') or not line:
                    continue
                else:
                    parts = line.split()
                    if len(parts) == 8:
                        current_segment.append([float(x) for x in parts])
        if current_segment:
            segments.append(current_segment)
        all_rows = []
        sp_offset = 0.0
        for segment in segments:
            seg = np.array(segment)
            _, first_idx = np.unique(seg[:, 1], return_index=True)
            seg = seg[np.sort(first_idx)]
            if all_rows:
                last_time = all_rows[-1][1]
                seg = seg[seg[:, 1] > last_time]
            if seg.size == 0:
                continue
            seg[:, 6] += sp_offset
            sp_offset = seg[-1, 6]
            all_rows.extend(seg.tolist())
        data = np.array(all_rows) if all_rows else np.empty((0, 9))
        if data.ndim == 1:
            data = data.reshape(1, -1)
        mask = data[:, 1] <= max_time
        data = data[mask]
        step, Time, Ta, pot, kin, etotal, transferred_eng, Te = (
            data[:, 0], data[:, 1], data[:, 2], data[:, 3],
            data[:, 4], data[:, 5], data[:, 6], data[:, 7]
        )
        #step, Time, Ta, Epot, Ekin, Etotal, transferred_eng, Te = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], data[:,7]
        Time_fs = Time*1000

        fig, axes = plt.subplots(3, 1, figsize=(8, 12))
        axes[0].plot(Time_fs, Ta, label='Ta')
        axes[0].plot(Time_fs, Te, label='Te')
        axes[0].set_ylabel('Temperature (K)', fontsize=15)
        axes[0].legend(fontsize=15)
        axes[0].grid(True)
        axes[0].text(0.05, 0.9, f"Average Ta: {np.mean(Ta[int(0.9*len(Ta)):]):.2f} K", transform=axes[1].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        axes[0].text(0.05, 0.8, f"Average Te: {np.mean(Te[int(0.9*len(Te)):]):.2f} K", transform=axes[1].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
        axes[1].plot(Time_fs, pot-pot[0], label='epot')
        axes[1].set_ylabel('Energy (eV)', fontsize=15)
        axes[1].legend(fontsize=15)
        axes[1].grid(True)
        
        axes[2].plot(Time_fs, transferred_eng, label='transfer')
        axes[2].set_xlabel('Time (fs)', fontsize=15)
        axes[2].set_ylabel('Energy (eV)', fontsize=15)
        axes[2].legend(fontsize=15)
        axes[2].grid(True)

        for i in range(3):
            axes[i].set_xscale('log')

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


    def get_Ta_time(self, dump_file: str, out_ta_time_file: str):
        with open(dump_file, 'r') as fin, open(out_ta_time_file, 'a') as ftime:
            while True:
                header = [fin.readline() for _ in range(11)]
                if not header[0]:
                    break
                time = float(header[1].strip())
                step = int(header[3].strip())
                num_atoms = int(header[5].strip())
                ftime.write(f"{step} {time}\n")
                for _ in range(num_atoms):
                    fin.readline()

    def get_grid_Ta_Te(self,
                       dump_file: str,
                       T_out_folder: str,
                       new_dump_file: str,
                       new_tout_file: str,
                       yes_dump: bool,
                       yes_tout: bool,
                       te_time_file: str,
                       block_id_list: list[int],
                       tout_step_id_list: list[int]):
        # ----------------------------- Delete existing files -------------------------
        if yes_dump and os.path.exists(new_dump_file):
            os.remove(new_dump_file)
        if yes_tout and os.path.exists(new_tout_file):
            os.remove(new_tout_file)

        rows = []
        with open(te_time_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                vals = [float(x) for x in line.split()]
                if vals == [0.0, 0.0]:
                    continue
                rows.append(vals)
        thermo_data = np.array(rows)
        step, time = thermo_data[:, 0], thermo_data[:, 1]

        tout_files = [os.path.join(T_out_folder, f) for f in os.listdir(T_out_folder)]
        tout_files.sort(key=lambda f: int(os.path.basename(f).split('_')[-1]))
        epsilon = 1e-1

        # grids from electronic system
        data = np.loadtxt(tout_files[0], skiprows=1, ndmin=2)
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

        if yes_tout:  
        # ----------------------------- Process T_out files -------------------------
            tout_step_id_set = set(tout_step_id_list) if tout_step_id_list is not None else None
            with open(new_tout_file, 'w') as fout:
                for tout_step, tout_file in enumerate(tout_files):
                    if tout_step_id_set is not None and tout_step not in tout_step_id_set:
                        continue
                    print(f"======= Electronic system, processing Time_fsstep: {tout_step} ========")
                    avg_temp = 0
                    fout.write("ITEM: TIME\n")
                    fout.write(f"{time[tout_step-1]}\n")
                    fout.write("ITEM: TIMESTEP\n")
                    fout.write(f"{int(step[tout_step-1])}\n")
                    fout.write("ITEM: NUMBER OF ATOMS\n")
                    fout.write(f"{x_grids * y_grids * z_grids}\n")
                    fout.write("ITEM: BOX BOUNDS pp pp pp\n")
                    fout.write(f"{min_x} {max_x+step_x}\n")
                    fout.write(f"{min_y} {max_y+step_y}\n")
                    fout.write(f"{min_z} {max_z+step_z}\n")
                    fout.write('ITEM: ATOMS grid_id x y z Te\n')
                    with open(tout_file, 'r') as fin:
                        lines = fin.readlines()
                        for idx, line in enumerate(lines[1:]):
                            x, y, z, Te = line.strip().split()
                            fout.write(f'{idx} {float(x)+step_x/2} {float(y)+step_y/2} {float(z)+step_z/2} {Te}\n') # the center of the grid
                            avg_temp += float(Te) / (x_grids * y_grids * z_grids)

        if yes_dump:
            # ----------------------------- Process dump file -------------------------
            with open(dump_file, 'r') as fin:
                header = [next(fin) for _ in range(10)]
            time = float(header[1].strip())
            num_atoms = int(header[5].strip())
            atomic_xlo, atomic_xhi = header[7].split()
            atomic_ylo, atomic_yhi = header[8].split()
            atomic_zlo, atomic_zhi = header[9].split()

            # step_x /= 2
            # step_y /= 2
            # step_z /= 2
            # x_grids = math.floor(x_length/step_x + epsilon)
            # y_grids = math.floor(y_length/step_y + epsilon)
            # z_grids = math.floor(z_length/step_z + epsilon)

            atomic_x_grids = math.floor((float(atomic_xhi) - float(atomic_xlo)) / step_x + epsilon)
            atomic_y_grids = math.floor((float(atomic_yhi) - float(atomic_ylo)) / step_y + epsilon)
            atomic_z_grids = math.floor((float(atomic_zhi) - float(atomic_zlo)) / step_z + epsilon)

            shift_x = float(atomic_xlo) - min_x
            shift_y = float(atomic_ylo) - min_y
            shift_z = float(atomic_zlo) - min_z

            shift_grid_x = math.floor(shift_x / step_x + epsilon)
            shift_grid_y = math.floor(shift_y / step_y + epsilon)
            shift_grid_z = math.floor(shift_z / step_z + epsilon)

            block_id_set = set(block_id_list) if block_id_list is not None else None
            block_idx = 0
            with open(dump_file, 'r') as fin:
                while True:
                    header_lines = [fin.readline() for _ in range(11)]
                    if not header_lines[0]:
                        break
                    block_idx += 1
                    if block_id_set is not None and block_idx not in block_id_set:
                        for _ in range(num_atoms):
                            fin.readline()
                        continue
                    atom_lines = [fin.readline() for _ in range(num_atoms)]
                    block_lines = header_lines + atom_lines

                    grids_val = [[[-epsilon for _ in range(z_grids)] for _ in range(y_grids)] for _ in range(x_grids)]
                    grids_cnt = [[[0 for _ in range(z_grids)] for _ in range(y_grids)] for _ in range(x_grids)]
                    grids_density = [[[-epsilon for _ in range(z_grids)] for _ in range(y_grids)] for _ in range(x_grids)]
                    grids_coupling = [[[-epsilon for _ in range(z_grids)] for _ in range(y_grids)] for _ in range(x_grids)]

                    print(f"======= Atomic system, processing block: {block_idx} ========")
                    for line in block_lines[11:]:
                        data = line.strip().split()
                        x = float(data[2])
                        y = float(data[3])
                        z = float(data[4])
                        ek = float(data[9])
                        site_density = float(data[11])
                        site_coupling = float(data[12])

                        ix = min(math.floor((x - min_x) / step_x + epsilon), shift_grid_x + atomic_x_grids - 1)
                        iy = min(math.floor((y - min_y) / step_y + epsilon), shift_grid_y + atomic_y_grids - 1)
                        iz = min(math.floor((z - min_z) / step_z + epsilon), shift_grid_z + atomic_z_grids - 1)
                        ix = max(ix, shift_grid_x)
                        iy = max(iy, shift_grid_y)
                        iz = max(iz, shift_grid_z)
                        grids_val[ix][iy][iz] += ek
                        grids_cnt[ix][iy][iz] += 1
                        grids_density[ix][iy][iz] += site_density
                        grids_coupling[ix][iy][iz] += site_coupling

                    # convert to temperature, average temp for one grid
                    for ix in range(x_grids):
                        for iy in range(y_grids):
                            for iz in range(z_grids):
                                factor = 2.0 / (3.0 * kB)
                                grids_val[ix][iy][iz] *= factor/max(grids_cnt[ix][iy][iz], 1)
                                grids_density[ix][iy][iz] /= max(grids_cnt[ix][iy][iz], 1)
                                grids_coupling[ix][iy][iz] /= max(grids_cnt[ix][iy][iz], 1)

                    # write to new dump file
                    block_time = float(block_lines[1].strip())
                    block_step = int(block_lines[3].strip())
                    with open(new_dump_file, 'a') as fout:
                        fout.write("ITEM: TIME\n")
                        fout.write(f"{block_time}\n")
                        fout.write("ITEM: TIMESTEP\n")
                        fout.write(f"{block_step}\n")
                        fout.write("ITEM: NUMBER OF ATOMS\n")
                        fout.write(f"{x_grids * y_grids * z_grids}\n")
                        fout.write("ITEM: BOX BOUNDS pp pp pp\n")
                        fout.write(f"{min_x} {max_x+step_x}\n")
                        fout.write(f"{min_y} {max_y+step_y}\n")
                        fout.write(f"{min_z} {max_z+step_z}\n")
                        fout.write('ITEM: ATOMS grid_id idx idy idz x y z Ta density coupling atoms_in_grid\n')
                        for iz in range(z_grids):
                            for iy in range(y_grids):
                                for ix in range(x_grids):
                                    grid_id = iz * (x_grids * y_grids) + iy * x_grids + ix
                                    x_center = min_x + ix * step_x + step_x / 2
                                    y_center = min_y + iy * step_y + step_y / 2
                                    z_center = min_z + iz * step_z + step_z / 2
                                    Ta = grids_val[ix][iy][iz]
                                    density = grids_density[ix][iy][iz]
                                    coupling = grids_coupling[ix][iy][iz]
                                    new_line = f"{grid_id} {ix} {iy} {iz} {x_center} {y_center} {z_center} {Ta:.6f} {density:.6f} {coupling:.6f} {grids_cnt[ix][iy][iz]}\n"
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
                       num_of_frames: int, 
                       figfile: str,
                       txtfile: str):
        threshold = []
        threshold_atoms = []
        time = []
        threshold_temp = 1230
        for frame_idx in range(num_of_frames):
            all_pipeline = import_file(dump_file)
            data = all_pipeline.compute(frame_idx)
            t = data.attributes.get('Time', frame_idx)
            time.append(t)
            grids = data.particles
            ta_list = grids['ta']
            atoms_in_grid = grids['atoms_in_grid']
            mask = ta_list > threshold_temp
            threshold.append(np.sum(mask))
            threshold_atoms.append(int(np.sum(atoms_in_grid[mask])))
        figure, ax = plt.subplots(figsize=(6, 6))
        ax.plot(time, threshold_atoms, label='threshold_count', marker='o')
        ax.set_xlabel('Time (ps)', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Number of liquid atoms', fontsize=LABEL_FONTSIZE)
        ax.set_xscale('log')
        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.grid(True)
        plt.tight_layout()
        figure.savefig(figfile, dpi=300)
        with open(txtfile, 'w') as f:
            for ti, ta, na in zip(time, threshold, threshold_atoms):
                f.write(f'{ti} {ta} {na}\n')

    def get_extreme_Te(self, 
                       tout_file: str,
                       num_of_frames: int, 
                       figfile: str,
                       txtfile: str):
        hottest = [] 
        coldest = []
        for frame_idx in range(num_of_frames):
            all_pipeline = import_file(tout_file)
            data = all_pipeline.compute(frame_idx)
            grids = data.particles
            te_list = grids['te']
            decend_T_grid_ids = np.argsort(-te_list)
            ascend_T_grid_ids = np.argsort(te_list)
            hottest.append(te_list[decend_T_grid_ids[0]])
            coldest.append(te_list[ascend_T_grid_ids[0]])
        figure, ax = plt.subplots(figsize=(6, 6))
        ax.plot(range(num_of_frames), hottest, label='Hottest', marker='o')
        ax.plot(range(num_of_frames), coldest, label='Coldest', marker='D')
        for idx, t in enumerate(coldest):
            if t < 200:
                print(f'{idx}, {t}')
        ax.axhline(y=300, color='red', linestyle='--', label='300 K')
        ax.set_xlabel('Number of frames', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('Te (K)', fontsize=LABEL_FONTSIZE)
        ax.set_yscale('log')
        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.grid(True)
        plt.tight_layout()
        figure.savefig(figfile, dpi=300)
        with open(txtfile, 'w') as f:
            for ti, te in zip(time, hottest):
                f.write(f'{ti} {te}\n')

    def plot_te_ta_along_x(self,
                           tout_file: str,
                           dump_file: str,
                           atomic_grid_list: list[int],
                           electron_grid_list: list[int],
                           figfile: str):
        min_te_limit = 10000
        max_te_limit = 0
        min_ta_limit = 10000
        max_ta_limit = 0
        fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharey=False)  # separate Ta and Te 
        fig2, ax2 = plt.subplots(figsize=(6, 6))                     # combine Ta and Te

        elec_pipeline = import_file(tout_file)
        atmo_pipeline = import_file(dump_file)

        n_ta = atmo_pipeline.source.num_frames
        n_te = elec_pipeline.source.num_frames
        ta_colors = plt.cm.viridis(np.linspace(0, 1, n_ta + 1))
        te_colors = plt.cm.viridis(np.linspace(0, 1, n_te + 1))

        shift = 0 # move atomic system to the center along x axis
        ta_handles = []
        te_handles = []
        for i in range(0, n_ta):
            atmo_data = atmo_pipeline.compute(i)
            atmo_grids = atmo_data.particles
            t_label = atmo_data.attributes.get('Time', i)

            ta_list = []
            x_list = []
            for grid_id in atomic_grid_list:
                x_list.append(atmo_grids.positions[grid_id][0])
                ta_list.append(atmo_grids['ta'][grid_id])

            shift = abs(x_list[-1]-x_list[0])/2
            ax[0].plot(x_list-shift, ta_list, color=ta_colors[i], marker='o', markersize=3, label=f'{t_label:.3f} ps')
            h, = ax2.plot(x_list-shift, ta_list, color=ta_colors[i], marker='o', markersize=3, label=f'Ta: {t_label:.3f} ps')
            ta_handles.append(h)
            min_ta_limit = min(min(ta_list), min_ta_limit, 295)
            max_ta_limit = max(max(ta_list), max_ta_limit, 305)

        for i in range(0, n_te):
            elec_data = elec_pipeline.compute(i)
            elec_grids = elec_data.particles
            t_label = elec_data.attributes.get('Time', i)

            te_list = []
            electron_x_list = []
            for grid_id in electron_grid_list:
                te_list.append(elec_grids['te'][grid_id])
                electron_x_list.append(elec_grids.positions[grid_id][0])

            ax[1].plot(electron_x_list-shift, te_list, linestyle='--', color=te_colors[i], marker='o', markersize=3, label=f'{t_label:.3f} ps')
            h, = ax2.plot(electron_x_list-shift, te_list, linestyle='--', color=te_colors[i], marker='o', markersize=3, label=f'Te {t_label:.3f} ps')
            te_handles.append(h)

            min_te_limit = min(min(te_list), min_te_limit, 295)
            max_te_limit = max(max(te_list), max_te_limit, 305)
  
        for i in range(2):
            ax[i].tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
            ax[i].grid(True)
            ax[i].set_xlabel('Coordinate (Å)', fontsize=LABEL_FONTSIZE-2)
            #ax[i].axhline(y=300, color='red', label='300 K')
            ax[i].legend(fontsize=LEGEND_FONTSIZE)

        # atomic: 
        ax[0].set_ylim(min_ta_limit, max_ta_limit)
        ax[0].set_ylabel('Temperature (K)', fontsize=LABEL_FONTSIZE)
        ax[0].set_title(f'Atomic system')
        
        # electronic:
        ax[1].set_ylim(min_te_limit, max_te_limit)
        ax[1].set_ylabel('')
        ax[1].set_title(f'Electronic system')
        

        #ax2.axhline(y=300, color='red', label='300 K')
        ax2.set_xlabel('Coordinate (Å)', fontsize=LABEL_FONTSIZE-2)
        ax2.set_ylabel('Temperature (K)', fontsize=LABEL_FONTSIZE-2)
        ax2.set_yscale('log')
        paired_handles = [h for pair in zip(ta_handles, te_handles) for h in pair]
        paired_handles += ta_handles[len(te_handles):] + te_handles[len(ta_handles):]
        ax2.legend(handles=paired_handles, fontsize=LEGEND_FONTSIZE)
        ax2.grid(True)
        ax2.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)

        # # inset: top-left, zoomed to 300–2500 K
        # ax2_ins = ax2.inset_axes([0.02, 0.55, 0.44, 0.42])
        # for line in ax2.get_lines():
        #     ax2_ins.plot(line.get_xdata(), line.get_ydata(),
        #                  color=line.get_color(), linestyle=line.get_linestyle(),
        #                  marker=line.get_marker(), markersize=line.get_markersize(),
        #                  linewidth=line.get_linewidth())
        # ax2_ins.set_ylim(250, 350)
        # ax2_ins.set_facecolor('white')
        # ax2_ins.grid(True, linewidth=0.5)
        # ax2_ins.tick_params(labelsize=TICK_FONTSIZE - 2)
        # ax2_ins.set_zorder(5)

        fig.subplots_adjust(hspace=0)
        fig.tight_layout()
        fig2.tight_layout()
        fig.savefig(figfile, dpi=300)
        fig2.savefig(figfile.replace('.png', '_combined.png'), dpi=300)

    def plot_xy_heatmap(self,
                        tout_file: str,
                        dump_file: str,
                        figfile1: str,
                        figfile2: str,
                        figfile3: str,
                        z: int,
                        gridx: int,
                        gridy: int,
                        border: int):
        elec_pipeline = import_file(tout_file)
        atom_pipeline = import_file(dump_file)

        te_maps = []
        te_Time_fs_list = []
        ta_maps = []
        ta_Time_fs_list = []

        starting_grid = z * gridx * gridy
        ending_grid = (z + 1) * gridx * gridy

        nx = gridx - 2 * border
        ny = gridy - 2 * border

        if nx <= 0 or ny <= 0:
            raise ValueError("border is too large for the grid size")

        n_te = elec_pipeline.source.num_frames
        for i in range(n_te):
            elec_data = elec_pipeline.compute(i)
            elec_grids = elec_data.particles
            t = elec_data.attributes.get('Time', i)

            te_list_full = []
            for grid_id in range(starting_grid, ending_grid):
                te_list_full.append(elec_grids['te'][grid_id])

            te_map_full = np.array(te_list_full).reshape(gridy, gridx)
            te_maps.append(te_map_full)
            t = f"{t:.3f}"
            te_Time_fs_list.append(t)

        n_ta = atom_pipeline.source.num_frames
        for i in range(n_ta):
            atom_data = atom_pipeline.compute(i)
            atom_grids = atom_data.particles
            t = atom_data.attributes.get('Time', i)

            ta_list_inner = []
            for y in range(border, gridy - border):
                for x in range(border, gridx - border):
                    grid_id = z * gridx * gridy + y * gridx + x
                    ta_list_inner.append(atom_grids['ta'][grid_id])

            ta_map_inner = np.array(ta_list_inner).reshape(ny, nx)
            ta_maps.append(ta_map_inner)
            t = f"{t:.3f}"
            ta_Time_fs_list.append(t)

        # ---- build tdiff maps by matching times ----
        tdiff_maps = []
        tdiff_Time_fs_list = []
        te_time_map = {t: te_maps[i] for i, t in enumerate(te_Time_fs_list)}
        ta_time_map = {t: ta_maps[i] for i, t in enumerate(ta_Time_fs_list)}
        for te, ta in zip(te_Time_fs_list, ta_Time_fs_list):
            te_inner = te_time_map[te][border:gridy - border, border:gridx - border]
            tdiff_maps.append(te_inner - ta_time_map[ta])
            tdiff_Time_fs_list.append(te)

        # ---- plot Te (full grid) ----
        te_vmin = 300
        te_vmax = 1000

        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        axes = axes.ravel()
        images = []
        for i, ax in enumerate(axes[:len(te_maps)]):
            norm = PowerNorm(gamma=0.5, vmin=te_vmin, vmax=te_vmax)
            im = ax.imshow(
                te_maps[i],
                origin='lower',
                cmap='coolwarm',
                norm=norm,
                aspect='equal'
            )
            images.append(im)
            # two decimals for time 
            ax.set_title(f"t = {te_Time_fs_list[i]} ps", fontsize=LABEL_FONTSIZE-2)
            if i % 2 == 0:
                ax.set_ylabel("Y", fontsize=LABEL_FONTSIZE-1)
            if i >= 2:
                ax.set_xlabel("X", fontsize=LABEL_FONTSIZE-1)
            ax.tick_params(labelsize=TICK_FONTSIZE-1)
            # add text for max and min values
            max_val = te_maps[i].max()
            min_val = te_maps[i].min()
            ax.text(
                0.02, 0.98, 
                f"max: {max_val:.1f} K\nmin: {min_val:.1f} K",
                fontsize=LABEL_FONTSIZE-3,
                ha='left',
                va='top',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )

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
        cbar.set_ticks([300, 400, 500, 600, 700, 800, 900, 1000])
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE-1)
        fig.savefig(figfile1, dpi=300)
        plt.close(fig)

        # ---- plot Ta (inner grid) ----
        ta_vmin = 300
        ta_vmax = 1000

        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        axes = axes.ravel()
        images = []
        for i, ax in enumerate(axes[:len(ta_maps)]):
            norm = PowerNorm(gamma=0.5, vmin=ta_vmin, vmax=ta_vmax)
            im = ax.imshow(
                ta_maps[i],
                origin='lower',
                cmap='coolwarm',
                norm=norm,
                aspect='equal'
            )
            images.append(im)
            ax.set_title(f"t = {ta_Time_fs_list[i]} ps", fontsize=LABEL_FONTSIZE-2)
            if i % 2 == 0:
                ax.set_ylabel("Y", fontsize=LABEL_FONTSIZE-1)
            if i >= 2:
                ax.set_xlabel("X", fontsize=LABEL_FONTSIZE-1)
            ax.tick_params(labelsize=TICK_FONTSIZE-1)
            # add text for max and min values
            max_val = ta_maps[i].max()
            min_val = ta_maps[i].min()
            ax.text(
                0.02, 0.98, 
                f"max: {max_val:.1f} K\nmin: {min_val:.1f} K",
                fontsize=LABEL_FONTSIZE-3,
                ha='left',
                va='top',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )

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
        cbar.set_ticks([300, 400, 500, 600, 700, 800, 900, 1000])
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE-1)
        fig.savefig(figfile2, dpi=300)
        plt.close(fig)

        # ---- plot Te - Ta (inner grid only) ----
        # if len(tdiff_maps) == 0:
        #     print("No matching time steps between Te and Ta for tdiff maps.")
        #     return
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        axes = axes.ravel()
        images = []
        for i, ax in enumerate(axes[:len(tdiff_maps)]):
            im = ax.imshow(
                tdiff_maps[i],
                origin='lower',
                cmap='coolwarm',
                vmin=-100,
                vmax=100,
                aspect='equal'
            )
            images.append(im)
            ax.set_title(f"t = {tdiff_Time_fs_list[i]} fs", fontsize=LABEL_FONTSIZE-2)
            if i % 2 == 0:
                ax.set_ylabel("Y", fontsize=LABEL_FONTSIZE-1)
            if i >= 2:
                ax.set_xlabel("X", fontsize=LABEL_FONTSIZE-1)
            ax.tick_params(labelsize=TICK_FONTSIZE-1)
            # add text for max and min values
            max_val = tdiff_maps[i].max()
            min_val = tdiff_maps[i].min()
            ax.text(
                0.02, 0.98, 
                f"max: {max_val:.1f} K\nmin: {min_val:.1f} K",
                fontsize=LABEL_FONTSIZE-3,
                ha='left',
                va='top',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )

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

























































        



          

        









