import os
import numpy as np
from ase.io import read 
from matplotlib import pyplot as plt
from ml4cascades.utils import BasicCellInfo
from ovito.io import import_file
from ovito.pipeline import StaticSource, Pipeline
from ovito.modifiers import WignerSeitzAnalysisModifier, ExpressionSelectionModifier, ClusterAnalysisModifier
from collections import Counter

class CascadeProcessor:
    def __init__(self,
                 basicCellInfo: BasicCellInfo,
                 traj_folder: str):
        self.bi = basicCellInfo
        self.traj_folder = traj_folder
        
    '''
    lack of damage energy calculation
    '''
    def cal_ibm(self,
                num_trajs: int,
                n0: float=1, # atomic density in atoms/Å³
                ed: float=1  # damage energy in eV
                ):
        time_all = []
        R2_all = []
        R2_avg_time_all = []
        for num_traj in range(num_trajs):
            time = [0]
            R2 = [0]
            traj_file = os.path.join(self.traj_folder, f'{num_traj+1}', 'trajectory_out.xyz')
            traj_frames = read(traj_file, format='extxyz', index=":")
            init_pos = traj_frames[0].get_positions()
            for frame in traj_frames[1:]:              # loop over frames 
                R2_val = 0
                curr_pos = frame.get_positions()
                curr_time = frame.info['time']
                for i in range(len(curr_pos)):         # sum over all atoms
                    R2_val += sum((np.array(curr_pos[i])-np.array(init_pos[i]))**2)
                time.append(curr_time)
                R2.append(R2_val)
            time_all.append(time)
            R2_all.append(R2)
            # for Q calculation: averaging R^2 over time 
            R2_avg_time_val = np.trapz(R2, time) / (time[-1] - time[0])
            R2_avg_time_all.append(R2_avg_time_val)

        # for R^2 plot: averaging R^2 over trajectories (interpolating to the first trajectory's time points)
        R2_interp_all = []
        for time, R2 in zip(time_all, R2_all):
            R2_interp = np.interp(time_all[0], time, R2)
            R2_interp_all.append(R2_interp)
        R2_avg = np.mean(R2_interp_all, axis=0)
        with open(os.path.join(self.traj_folder, 'R2.txt'), 'w') as f:
            f.write('Time (fs)\tR^2 (Å²)\n')
            for t, r2 in zip(time_all[0], R2_avg):
                f.write(f'{t}\t{r2}\n')
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(time_all[0], R2_avg)
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel(r'$\mathrm{R}^2 (\mathrm{Å}^2)$')
        plt.tight_layout()
        fig.savefig(os.path.join(self.traj_folder, 'R2.png'), dpi=300)

        # for Q calculation: averaging R^2 over time and then over trajectories
        Q_avg = np.mean(R2_avg_time_val) / (6*n0*ed)
        Q_std = np.std(R2_avg_time_val) / (6*n0*ed)
        with open(os.path.join(self.traj_folder, 'Q.txt'), 'w') as f:
            f.write('Q_avg\tQ_std\n')
            f.write(f'{Q_avg}\t{Q_std}\n')

    '''
    lack of other methods for defect analysis
    '''
    # Wigner-Seitz method
    def cal_wsDefect(self, num_trajs: int):
        time_all = []
        num_vac_all = []
        num_int_all = []
        for num_traj in range(num_trajs):
            time = [0]
            num_vac = [0]
            num_int = [0]
            traj_file = os.path.join(self.traj_folder, f'{num_traj+1}', 'trajectory_out.xyz')
            traj_frames = read(traj_file, format='extxyz', index=":")
            all_pipeline = import_file(traj_file)
            init_frame = all_pipeline.compute(0)
            reference_pipeline = Pipeline(source=StaticSource(data=init_frame))
            for frame_idx in range(1, all_pipeline.source.num_frames):
                cur_pipeline = Pipeline(source=StaticSource(data=all_pipeline.compute(frame_idx)))
                wsam = WignerSeitzAnalysisModifier(per_type_occupancies=True, output_displaced=False)
                wsam.reference = reference_pipeline.source
                cur_pipeline.modifiers.append(wsam)
                data = cur_pipeline.compute(0)
                cnt_vacancies = 0
                cnt_interstitials = 0
                for occupancy in data.particles['Occupancy']:
                    if occupancy == 0:
                        cnt_vacancies += 1
                    if occupancy > 1:
                        cnt_interstitials += 1
                cur_pipeline.modifiers.remove(wsam)
                num_vac.append(cnt_vacancies)
                num_int.append(cnt_interstitials) 
                time.append(traj_frames[frame_idx].info['time'])
            time_all.append(time)
            num_vac_all.append(num_vac)
            num_int_all.append(num_int)

        # for defect plot: averaging defect over trajectories (interpolating to the first trajectory's time points)
        vac_interp_all = []
        int_interp_all = []
        for time, num_vac_val, num_int_val in zip(time_all, num_vac_all, num_int_all):
            vac_interp = np.interp(time_all[0], time, num_vac_val)
            int_interp = np.interp(time_all[0], time, num_int_val)
            vac_interp_all.append(vac_interp)
            int_interp_all.append(int_interp)
        vac_avg = np.mean(vac_interp_all, axis=0)
        int_avg = np.mean(int_interp_all, axis=0)
        vac_std = np.std(vac_interp_all, axis=0)
        int_std = np.std(int_interp_all, axis=0)
        with open(os.path.join(self.traj_folder, 'defect.txt'), 'w') as f:
            f.write('Time (fs)\tnum_vac\tstd_vac\tnum_int\tstd_int\n')
            for t, v, vstd, i, istd in zip(time_all[0], vac_avg, vac_std, int_avg, int_std):
                f.write(f'{t:.3f}\t{int(v)}\t{int(vstd)}\t{int(i)}\t{int(istd)}\n')
        fig, ax = plt.subplots(figsize=(6, 4))
        eps = 1e-12
        vac_low = np.clip(vac_avg - vac_std, eps, None)
        vac_high = vac_avg + vac_std
        int_low = np.clip(int_avg - int_std, eps, None)
        int_high = int_avg + int_std
        ax.plot(time_all[0], vac_avg, label='Vacancy')
        ax.fill_between(time_all[0], vac_low, vac_high, alpha=0.3)
        # ax.plot(time_all[0], int_avg, label='Interstitial')
        # ax.fill_between(time_all[0], int_low, int_high, alpha=0.3)
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('Number of Defects')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(self.traj_folder, 'defects.png'), dpi=300)

    '''
    cutoff in Å
    '''
    def cal_cluster(self, num_trajs: int, expression: str, cutoff: float=1.0):
        time_all = []
        max_cluster_size_all = []
        for num_traj in range(num_trajs):
            time = [0]
            max_cluster_size = [0]
            traj_file = os.path.join(self.traj_folder, f'{num_traj+1}', 'trajectory_out.xyz')
            traj_frames = read(traj_file, format='extxyz', index=":")
            all_pipeline = import_file(traj_file)
            init_frame = all_pipeline.compute(0)
            reference_pipeline = Pipeline(source=StaticSource(data=init_frame))
            for frame_idx in range(1, all_pipeline.source.num_frames):
                cur_pipeline = Pipeline(source=StaticSource(data=all_pipeline.compute(frame_idx)))
                wsam = WignerSeitzAnalysisModifier(per_type_occupancies=True, output_displaced=False)
                wsam.reference = reference_pipeline.source
                cur_pipeline.modifiers.append(wsam)
                sel = ExpressionSelectionModifier(expression=expression)
                cur_pipeline.modifiers.append(sel)
                cls = ClusterAnalysisModifier(cutoff=cutoff, sort_by_size=True, only_selected=True)
                cur_pipeline.modifiers.append(cls)
                data = cur_pipeline.compute(0)
                cluster_table = data.tables['clusters']
                cluster_sizes = cluster_table['Cluster Size']
                count_dict = Counter(cluster_sizes)  # {cluster size: count}
                cur_pipeline.modifiers.remove(wsam)
                time.append(traj_frames[frame_idx].info['time'])
                max_cluster_size.append(next(iter(count_dict)))
            time_all.append(time)
            max_cluster_size_all.append(max_cluster_size)
            
        # for max cluster size plot: averaging over trajectories (interpolating to the first trajectory's time points)
        max_cluster_interp_all = []
        for time, max_cluster_size in zip(time_all, max_cluster_size_all):
            max_cluster_interp = np.interp(time_all[0], time, max_cluster_size)
            max_cluster_interp_all.append(max_cluster_interp)
        max_cluster_avg = np.mean(max_cluster_interp_all, axis=0)
        max_cluster_std = np.std(max_cluster_interp_all, axis=0)
        with open(os.path.join(self.traj_folder, 'max_cluster.txt'), 'w') as f:
            f.write('Time (fs)\tMax Cluster Size\n')
            for t, mcs, mcs_std in zip(time_all[0], max_cluster_avg, max_cluster_std):
                f.write(f'{t}\t{int(mcs)}\t{int(mcs_std)}\n')
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.plot(time_all[0], max_cluster_avg)
        ax.set_xlabel('Time (fs)')
        ax.set_ylabel('Max Cluster Size')
        plt.tight_layout()
        fig.savefig(os.path.join(self.traj_folder, 'max_cluster.png'), dpi=300)
