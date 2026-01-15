import numpy as np
import matplotlib.pyplot as plt
import os, re

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
        step, _, _, Epot, Ekin, Etotal = np.loadtxt(datafile2, skiprows=1, unpack=True)
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

        # grids from electronic system
        data = np.loadtxt(tout_files[0], skiprows=1)
        mask = (data[:, 0] == 0) & (data[:, 1] == 0)
        slice_data = data[mask]
        z = slice_data[:, 2]
        step_z = z[1] - z[0]
        mask = (data[:, 0] == 0) & (data[:, 2] == 0)
        slice_data = data[mask]
        y = slice_data[:, 1]
        step_y = y[1] - y[0]
        mask = (data[:, 1] == 0) & (data[:, 2] == 0)
        slice_data = data[mask]
        x = slice_data[:, 0]
        step_x = x[1] - x[0]

        x_length = max(x) - min(x) + step_x
        y_length = max(y) - min(y) + step_y
        z_length = max(z) - min(z) + step_z

        x_grids = int(x_length/step_x)
        y_grids = int(y_length/step_y)
        z_grids = int(z_length/step_z)

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
                    fout.write(f"0.0 {x_length}\n")
                    fout.write(f"0.0 {y_length}\n")
                    fout.write(f"0.0 {z_length}\n")
                    new_line = 'ITEM: ATOMS grid_id x y z Te\n'
                    fout.write(new_line)
                with open(tout_file, 'r') as fin:
                    lines = fin.readlines()
                    for idx, line in enumerate(lines[1:]):
                        x, y, z, Te = line.strip().split()
                        if flag == 2 or flag == 5:
                            fout.write(f'{idx} {float(x)+step_x/2} {float(y)+step_y/2} {float(z)+step_z/2} {Te}\n')
                        avg_temp += float(Te) / (x_grids * y_grids * z_grids)
                if flag == 4 or flag == 5:
                    with open(avg_Te_file, 'a') as favg:
                        favg.write(f"{timestep} {avg_temp}\n")

        # ----------------------------- Process dump file -------------------------
        # grids from atomic system
        num_atoms = int(header[3].strip())
        xlo, xhi = header[5].split()
        ylo, yhi = header[6].split()
        zlo, zhi = header[7].split()
        x_grids = int((float(xhi)-float(xlo))/step_x)
        y_grids = int((float(yhi)-float(ylo))/step_y)
        z_grids = int((float(zhi)-float(zlo))/step_z)
        
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

                ix = min(int((x - float(xlo)) / step_x), x_grids - 1)
                iy = min(int((y - float(ylo)) / step_y), y_grids - 1)
                iz = min(int((z - float(zlo)) / step_z), z_grids - 1)
                grids_val[ix][iy][iz] += ek
                grids_cnt[ix][iy][iz] += 1
    
            # convert to temperature, average temp for one grid
            for ix in range(x_grids):
                for iy in range(y_grids):
                    for iz in range(z_grids):
                        factor = 2.0 / (3.0 * 8.617333262145e-5)
                        grids_val[ix][iy][iz] *= factor/max(grids_cnt[ix][iy][iz], 1)
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
                    fout.writelines(header)
                    new_line = block_lines[8].rstrip('\n') + ' Ta\n'
                    fout.write(new_line)
                    for line in block_lines[9:]:
                        data = line.strip().split()
                        x = float(data[2])
                        y = float(data[3])
                        z = float(data[4])
                        ix = min(int((x - float(xlo)) / step_x), x_grids - 1)
                        iy = min(int((y - float(ylo)) / step_y), y_grids - 1)
                        iz = min(int((z - float(zlo)) / step_z), z_grids - 1)
                        Ta = grids_val[ix][iy][iz]
                        new_line = ' '.join(data + [f"{Ta:.6f}"]) + '\n'
                        fout.write(new_line)





                  
                




   
                       
                        


      
                    
        
        


        


            


        

                        


            
        



          

        









