import numpy as np
import matplotlib.pyplot as plt
import os, re, math
from ovito.io import import_file
from ovito.pipeline import StaticSource, Pipeline

class TurbogapCascadePlotter:
    def __init__(self):
        pass

    def plot_eph_results(self, datafile: str, figfile: str):
        Time, E_fric, E_rand, E_net_cum, T_e, T_a, Kin_a, Pot_a = np.loadtxt(datafile, skiprows=1, unpack=True)
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))

        # axes[0].plot(Time, E_fric+E_rand, label='E_fric')
        # axes[0].plot(Time, E_rand, label='E_rand')
        pot_shift = -np.min(Pot_a) + np.min(Kin_a) 
        # axes[0].plot(Time, E_net_cum+Kin_a+Pot_a+pot_shift, label='Tot_a')
        axes[0].plot(Time, E_net_cum, label='E_net_cum')

        # axes[0].plot(Time, E_net_cum+Kin_a+Pot_a+pot_shift, label='E_net_cum + Tot_a')
        axes[0].set_ylabel('Energy (eV)')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(Time, T_e, label='T_e')
        axes[1].plot(Time, T_a, label='T_a')
        axes[1].set_ylabel('Temperature (K)')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(Time, Kin_a, label='Kin_a')
        axes[2].plot(Time, Pot_a+pot_shift, label='Pot_a')
        axes[2].plot(Time, Kin_a+Pot_a+pot_shift, label='Tot_a')
        axes[2].set_xlabel('Time (fs)')
        axes[2].set_ylabel('Energy (eV)')
        axes[2].legend()
        axes[2].grid(True)
        
        # axes[2].set_xscale('log')
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

class LammpsCascadePlotter:
    def __init__(self):
        pass

    def plot_eph_results(self, datafile1: str, datafile2: str, figfile: str):
        step, Time, Ta, friction1, Te, Tbr, Tin = np.loadtxt(datafile1, skiprows=1, unpack=True)
        data = np.loadtxt(datafile2, skiprows=1)
        step, Epot, Ekin, Etotal = data[:,0], data[:,3], data[:,4], data[:,5]
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        axes[0].plot(Time, friction1, color='red')
        axes[0].set_ylabel('E_transfer (eV)', fontsize=15)
        axes[0].legend(fontsize=15)
        axes[0].grid(True)

        # axes[1].plot(Time, Ta, label='Ta')
        axes[1].plot(Time, Te, label='Te')
        axes[1].plot(Time, Tbr, label='Tborder')
        axes[1].plot(Time, Ta, label='Tinside')
        axes[1].set_ylabel('Temperature (K)', fontsize=15)
        axes[1].legend(fontsize=15)
        axes[1].grid(True)
        # add text, 2 decimal
        # axes[1].text(0.05, 0.9, f"Average Ta: {np.mean(Ta[int(0.9*len(Ta)):]):.2f} K", transform=axes[1].transAxes, fontsize=15, bbox=dict(facecolor='white', alpha=0.5))  
        # axes[1].text(0.05, 0.8, f"Average Te: {np.mean(Te[int(0.9*len(Te)):]):.2f} K", transform=axes[1].transAxes, fontsize=15, bbox=dict(facecolor='white', alpha=0.5))
        # axes[1].text(0.05, 0.7, f"Average Tbr: {np.mean(Tbr[int(0.9*len(Tbr)):]):.2f} K", transform=axes[1].transAxes, fontsize=15, bbox=dict(facecolor='white', alpha=0.5))
        
        pot_shift = -np.min(Epot) + np.min(Ekin)
        axes[2].plot(Time, Ekin, label='Ekin')
        axes[2].plot(Time, Epot+pot_shift, label='Epot')
        axes[2].plot(Time, Etotal+pot_shift, label='Etotal')
        axes[2].set_xlabel('Time (ps)', fontsize=15)
        axes[2].set_ylabel('Energy (eV)', fontsize=15)
        axes[2].legend(fontsize=15)
        axes[2].grid(True)
        
        # increase ticks size
        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=15)
        
        # axes[1].set_xscale('log')
        # fig, ax = plt.subplots(figsize=(6, 6))
        # ax.plot(Time, Ta, label='Ta')
        # ax.plot(Time, Te, label='Te')
        # ax.set_ylabel('Temperature (K)')
        # ax.set_xlabel('Time (ps)')
        # ax.legend()
        # ax.grid(True)
        plt.tight_layout()
        fig.savefig(figfile, dpi=300)
        # average Ta and Te over last 10% of time
        print(f"Average Ta: {np.mean(Ta[int(0.9*len(Ta)):])} K")
        print(f"Average Te: {np.mean(Te[int(0.9*len(Te)):])} K")    

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
            for timestep, tout_file in enumerate(tout_files):
                print(f"======= Electronic system, processing timestep: {timestep} ========")
                avg_temp = 0
                if flag == 2 or flag == 5:
                    fout.write(header[0])  # ITEM: TIMESTEP
                    fout.write(f"{timestep}\n") 
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
                        favg.write(f"{timestep} {avg_temp}\n")

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
            grids_val = [[[0 for _ in range(z_grids)] for _ in range(y_grids)] for _ in range(x_grids)]
            grids_cnt = [[[0 for _ in range(z_grids)] for _ in range(y_grids)] for _ in range(x_grids)]
            start_line = block_idx * block_length
            end_line = start_line + block_length
            block_lines = dump_lines[start_line:end_line]

            print(f"======= Atomic system, processing block: {block_idx}/{num_blocks-1} ========")
            for line in block_lines[9:]:
                data = line.strip().split()
                x = float(data[2])
                y = float(data[3])
                z = float(data[4])
                ek = float(data[9])

                ix = min(math.floor((x - min_x) / step_x + epsilon), shift_grid_x + atomic_x_grids - 1)
                iy = min(math.floor((y - min_y) / step_y + epsilon), shift_grid_y + atomic_y_grids - 1)
                iz = min(math.floor((z - min_z) / step_z + epsilon), shift_grid_z + atomic_z_grids - 1)
                ix = max(ix, shift_grid_x)
                iy = max(iy, shift_grid_y)
                iz = max(iz, shift_grid_z)
                grids_val[ix][iy][iz] += ek
                grids_cnt[ix][iy][iz] += 1

            # convert to temperature, average temp for one grid
            for ix in range(x_grids):
                for iy in range(y_grids):
                    for iz in range(z_grids):
                        factor = 2.0 / (3.0 * 8.617333262145e-5)
                        grids_val[ix][iy][iz] *= factor/max(grids_cnt[ix][iy][iz], 1)

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
                    fout.write(header[0])  # ITEM: TIMESTEP
                    fout.write(f"{block_idx}\n")
                    fout.write(header[2])  # ITEM: NUMBER OF ATOMS
                    fout.write(f"{x_grids * y_grids * z_grids}\n")
                    fout.write(header[4])  # ITEM: BOX BOUNDS pp pp pp
                    fout.write(f"{min_x} {max_x+step_x}\n")
                    fout.write(f"{min_y} {max_y+step_y}\n")
                    fout.write(f"{min_z} {max_z+step_z}\n")
                    new_line = 'ITEM: ATOMS grid_id idx idy idz x y z Ta\n'
                    fout.write(new_line)
                    for iz in range(z_grids):
                        for iy in range(y_grids):
                            for ix in range(x_grids):
                                grid_id = iz * (x_grids * y_grids) + iy * x_grids + ix
                                x_center = min_x + ix * step_x + step_x / 2
                                y_center = min_y + iy * step_y + step_y / 2
                                z_center = min_z + iz * step_z + step_z / 2
                                Ta = grids_val[ix][iy][iz]
                                new_line = f"{grid_id} {ix} {iy} {iz} {x_center} {y_center} {z_center} {Ta:.6f}\n"
                                fout.write(new_line)


                # for ix in range(x_grids):
                #     for iy in range(y_grids):
                #         for iz in range(z_grids):
                #             grid_id = ix * (y_grids * z_grids) + iy * z_grids + iz
                #             x_center = min_x + ix * step_x + step_x / 2
                #             y_center = min_y + iy * step_y + step_y / 2
                #             z_center = min_z + iz * step_z + step_z / 2
                #             Ta = grids_val[ix][iy][iz]
                #             new_line = f"{grid_id} {x_center} {y_center} {z_center} {Ta:.6f}\n"
                #             fout.write(new_line)
                # for line in block_lines[9:]:
                #     data = line.strip().split()
                #     x = float(data[2])
                #     y = float(data[3])
                #     z = float(data[4])
                #     ix = min(int((x - float(xlo)) / step_x), x_grids - 1)
                #     iy = min(int((y - float(ylo)) / step_y), y_grids - 1)
                #     iz = min(int((z - float(zlo)) / step_z), z_grids - 1)
                #     Ta = grids_val[ix][iy][iz]
                #     # new_line = ' '.join(data + [f"{Ta:.6f}"]) + '\n'
                #     new_line = f"{data[0]} {data[1]} {data[2]} {data[3]} {data[4]} {Ta:.6f}\n"
                #     fout.write(new_line)


        # # ----------------------------- Process dump file -------------------------
        # # grids from atomic system
        # num_atoms = int(header[3].strip())
        # atomic_xlo, atomic_xhi = header[5].split()
        # atomic_ylo, atomic_yhi = header[6].split()
        # atomic_zlo, atomic_zhi = header[7].split()
        # # electronic system is larger than atomic system by 2 steps in each direction
        # shift_x = float(atomic_xlo) - min_x 
        # shift_y = float(atomic_ylo) - min_y
        # shift_z = float(atomic_zlo) - min_z
        # xlo = float(atomic_xlo) + shift_x 
        # xhi = float(atomic_xhi) + shift_x
        # ylo = float(atomic_ylo) + shift_y
        # yhi = float(atomic_yhi) + shift_y
        # zlo = float(atomic_zlo) + shift_z
        # zhi = float(atomic_zhi) + shift_z
        # atomic_x_grids = int((float(xhi)-float(xlo))/step_x)
        # atomic_y_grids = int((float(yhi)-float(ylo))/step_y)
        # atomic_z_grids = int((float(zhi)-float(zlo))/step_z)
        
        # block_length = num_atoms + 9  
        # num_blocks = len(dump_lines) // block_length
        # for block_idx in range(1, num_blocks):
        #     grids_val = [[[0 for _ in range(z_grids)] for _ in range(y_grids)] for _ in range(x_grids)]
        #     grids_cnt = [[[0 for _ in range(z_grids)] for _ in range(y_grids)] for _ in range(x_grids)]
        #     start_line = block_idx * block_length
        #     end_line = start_line + block_length
        #     block_lines = dump_lines[start_line:end_line]
        #     print(f"======= Atomic system, processing block: {block_idx}/{num_blocks-1} ========")
        #     for line in block_lines[9:]:
        #         data = line.strip().split()
        #         x = float(data[2])
        #         y = float(data[3])
        #         z = float(data[4])
        #         ek = float(data[9])

        #         ix = min(int((x - float(xlo)) / step_x), atomic_x_grids - 1)
        #         iy = min(int((y - float(ylo)) / step_y), atomic_y_grids - 1)
        #         iz = min(int((z - float(zlo)) / step_z), atomic_z_grids - 1)
        #         grids_val[ix][iy][iz] += ek
        #         grids_cnt[ix][iy][iz] += 1
    
        #     # convert to temperature, average temp for one grid
        #     for ix in range(x_grids):
        #         for iy in range(y_grids):
        #             for iz in range(z_grids):
        #                 factor = 2.0 / (3.0 * 8.617333262145e-5)
        #                 grids_val[ix][iy][iz] *= factor/max(grids_cnt[ix][iy][iz], 1)
        #     # average temperature for the whole system
        #     if flag == 3 or flag == 5:
        #         avg_temp = 0
        #         for ix in range(x_grids):
        #             for iy in range(y_grids):
        #                 for iz in range(z_grids):
        #                     avg_temp += grids_val[ix][iy][iz] / (x_grids * y_grids * z_grids)
        #         with open(avg_Ta_file, 'a') as favg:
        #             favg.write(f"{block_idx} {avg_temp}\n")

        #     # write to newe dump file
        #     if flag == 1 or flag == 5:
        #         with open(new_dump_file, 'a') as fout:
        #             fout.writelines(header)
        #             new_line = 'ITEM: ATOMS grid_id x y z Ta\n'
        #             fout.write(new_line)
        #             for line in block_lines[9:]:
        #                 data = line.strip().split()
        #                 x = float(data[2])
        #                 y = float(data[3])
        #                 z = float(data[4])
        #                 ix = min(int((x - float(xlo)) / step_x), x_grids - 1)
        #                 iy = min(int((y - float(ylo)) / step_y), y_grids - 1)
        #                 iz = min(int((z - float(zlo)) / step_z), z_grids - 1)
        #                 Ta = grids_val[ix][iy][iz]
        #                 # new_line = ' '.join(data + [f"{Ta:.6f}"]) + '\n'
        #                 new_line = f"{data[0]} {data[1]} {data[2]} {data[3]} {data[4]} {Ta:.6f}\n"
        #                 fout.write(new_line)

    
    # def get_hottest_Ta_Te(self, 
    #                       tout_file: str,
    #                       frame_idx: int):
        # rank hottest grids based on Te
        # all_pipeline = import_file(tout_file)
        # data = all_pipeline.compute(frame_idx)
        # grids = data.particles
        # te_list = grids['te']
        # sorted_grid_ids = np.argsort(-te_list)
        # for i in range(10):
        #     print(sorted_grid_ids[i], te_list[sorted_grid_ids[i]], grids.positions[sorted_grid_ids[i]])

        # plot te along x axis 

    def plot_te_ta_along_x(self, 
                           tout_file: str, 
                           dump_file: str,
                           frame_idx_list: list[int],
                           time_list: list[float],
                           grid_list: list[int],
                           electron_grid_list: list[int],
                           figfile: str,
                           tout_file2: str=None,
                           dump_file2: str=None,
                           name_case2: str=None):
        # list of colors for different frames, the same length as frame_idx_list
        length = len(frame_idx_list)
        colors = plt.cm.viridis(np.linspace(0, 1, length+1))  
        min_limit = 10000
        max_limit = 0
        fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
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
            

            ax[0].plot(electron_x_list, te_list, color=colors[i], marker='D', markersize=4, label=f'{time_list[i]} ps')
            ax[1].plot(x_list, ta_list, linestyle='--', color=colors[i], marker='D', markersize=4, label=f'{time_list[i]} ps')
            if tout_file2 is not None and dump_file2 is not None:
                ax[0].plot(
                    electron_x_list2, te_list2,
                    linestyle='None',
                    marker='o',
                    markersize=4,
                    markerfacecolor='none',
                    color=colors[i],
                    # label=f'{time_list[i]} ps ({name_case2})'
                )
                ax[1].plot(
                    x_list2, ta_list2,
                    linestyle='None',
                    marker='o',
                    markerfacecolor='none',
                    markersize=4,
                    color=colors[i],
                    # label=f'{time_list[i]} ps ({name_case2})'
                )

            min_limit = min(min(te_list), min(ta_list), min_limit) 
            max_limit = max(max(te_list), max(ta_list), max_limit) 
        
        ax[0].set_ylim(min_limit, max_limit)
        ax[1].set_ylim(min_limit, max_limit)
        ax[1].set_ylabel('')

        # set a horizontal line at y=300K
        ax[0].axhline(y=300, color='red', label='300 K')
        ax[1].axhline(y=300, color='red', label='300 K')
    
        ax[0].legend()
        # ax[1].legend()
        ax[0].set_xlabel('X Position (Angstrom)')
        ax[1].set_xlabel('X Position (Angstrom)')
        ax[0].set_ylabel('Temperature (K)')

        ax[0].set_title(f'Electronic system ')
        ax[1].set_title(f'Atomic system ')

        ax[0].grid(True)
        ax[1].grid(True)

        # set y axis to log scale
        # ax[0].set_yscale('log')
        # ax[1].set_yscale('log')
  

        fig.subplots_adjust(hspace=0)
        plt.tight_layout()
        fig.savefig(figfile, dpi=300)
            









                  
                




   
                       
                        


      
                    
        
        


        


            


        

                        


            
        



          

        









