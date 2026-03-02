import os
from typing import Optional
import numpy as np
from ase.io.lammpsrun import read_lammps_dump_text
from matplotlib import pyplot as plt
from ml4cascades.utils import BasicCellInfo
from ovito.io import import_file
from ovito.pipeline import StaticSource, Pipeline
from ovito.modifiers import WignerSeitzAnalysisModifier, ExpressionSelectionModifier, ClusterAnalysisModifier
from collections import Counter
from ml4cascades.loggers.logger import AppLogger

module_dir = os.path.dirname(__file__)

class CascadeProcessor:
    def __init__(self,
                 basicCellInfo: BasicCellInfo,
                 PKA_kin_eng: float,
                 traj_folder: str,
                 model_name: str,
                 task_name='processing'):
        self.bi = basicCellInfo
        self.PKA_kin_eng = PKA_kin_eng
        self.traj_folder = traj_folder
        self.log_dir = os.path.join(module_dir, 'logs', task_name)
        self.log_file = os.path.join(self.log_dir, f'{model_name}.log')
        self.logger = AppLogger(__name__, self.log_file, overwrite=True).get_logger()
        os.makedirs(self.log_dir, exist_ok=True)
        
    '''
    issue with positions. compare the positions of the same id atom.

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
            traj_file = os.path.join(self.traj_folder, f'{num_traj+1}', 'data.output')
            all_pipeline = import_file(traj_file)
            num_frames = all_pipeline.source.num_frames
            data = all_pipeline.compute(0)
            init_pos = init_frame.get_positions()
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
            f.write('Time (ps)\tR^2 (ang^2)\n')
            for t, r2 in zip(time_all[0], R2_avg):
                f.write(f'{t}\t{r2}\n')
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(time_all[0], R2_avg)
        ax.set_xlabel('Time (ps)')
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

    '''
    lack of other methods for defect analysis
    '''
    # Wigner-Seitz method
    def cal_WSDefect(
        self,
        start_traj: int,
        num_trajs: int,
        exclude_list: list[int],
        single_traj_dir: Optional[str] = None,
    ):
        time_all = []
        num_vac_all = []
        num_int_all = []
        num_def_all = []
        self.logger.info(f'#------------Defect analysis, PKA_kin_eng: {self.PKA_kin_eng} eV------------#')
        cnt = 0
        traj_dirs = []
        if single_traj_dir is not None:
            traj_dirs.append((single_traj_dir, os.path.basename(single_traj_dir.rstrip(os.sep)), None))
        else:
            for num_traj in range(num_trajs):
                traj_dir = os.path.join(self.traj_folder, f'{start_traj+num_traj}')
                traj_id = start_traj + num_traj
                traj_dirs.append((traj_dir, f'{traj_id}', traj_id))

        for traj_dir, traj_label, traj_id in traj_dirs:
            eng_out = os.path.join(traj_dir, 'thermo.out')
            if not os.path.exists(eng_out) or (traj_id is not None and traj_id in exclude_list):
                continue
            cnt += 1
            if cnt > num_trajs:
                break
            self.logger.info(f'Starting the {cnt}th trajectory in folder {traj_label}...')
            data = np.loadtxt(eng_out, skiprows=1)
            interval = 11
            mydata = data[::interval]
            timestep = mydata.T[0]
            time = mydata.T[1]
            num_vac = []
            num_int = []
            num_def = []

            traj_file = os.path.join(traj_dir, 'data.output')
            all_pipeline = import_file(traj_file)
            init_frame = all_pipeline.compute(0)
            reference_pipeline = Pipeline(source=StaticSource(data=init_frame))
            for frame_idx in range(0, len(data), interval):
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
                num_def.append(cnt_vacancies + cnt_interstitials)
            time_all.append(time)
            num_vac_all.append(num_vac) 
            num_int_all.append(num_int)
            num_def_all.append(num_def)

        if not time_all:
            self.logger.warning('No valid trajectories found for defect analysis.')
            return

        # for defect plot: averaging defect over trajectories (interpolating to the first trajectory's time points)
        vac_interp_all = []
        int_interp_all = []
        def_interp_all = []
        for time, num_vac_val, num_int_val, num_def_val in zip(time_all, num_vac_all, num_int_all, num_def_all):
            vac_interp = np.interp(time_all[0], time, num_vac_val)
            int_interp = np.interp(time_all[0], time, num_int_val)
            def_interp = np.interp(time_all[0], time, num_def_val)
            vac_interp_all.append(vac_interp)
            int_interp_all.append(int_interp)
            def_interp_all.append(def_interp)
        vac_avg = np.mean(vac_interp_all, axis=0)
        int_avg = np.mean(int_interp_all, axis=0)
        vac_std = np.std(vac_interp_all, axis=0)
        int_std = np.std(int_interp_all, axis=0)
        def_avg = np.mean(def_interp_all, axis=0)
        def_std = np.std(def_interp_all, axis=0)
        output_dir = single_traj_dir if single_traj_dir is not None else self.traj_folder
        with open(os.path.join(output_dir, 'defect.txt'), 'w') as f:
            f.write('Time (ps)\tnum_vac\tstd_vac\tnum_int\tstd_int\tnum_def\tstd_def\n')
            for t, v, vstd, i, istd, d, dstd in zip(time_all[0], vac_avg, vac_std, int_avg, int_std, def_avg, def_std):
                f.write(f'{t:.3f}\t{v:.2f}\t{vstd:.2f}\t{i:.2f}\t{istd:.2f}\t{d:.2f}\t{dstd:.2f}\n')
        fig, ax = plt.subplots(figsize=(6, 4))
        vac_low = vac_avg - vac_std
        vac_high = vac_avg + vac_std
        int_low = int_avg - int_std
        int_high = int_avg + int_std
        def_low = def_avg - def_std
        def_high = def_avg + def_std
        ax.plot(time_all[0], def_avg, label='Defects')
        ax.fill_between(time_all[0], def_low, def_high, alpha=0.3)
        # ax.plot(time_all[0], int_avg, label='Interstitial')
        # ax.fill_between(time_all[0], int_low, int_high, alpha=0.3)
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Number of Defects')
        ax.set_xscale('log')
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'defects.png'), dpi=300)
        self.logger.info('#------------ Defect analysis completed. ------------#')

    # cutoff from 10.1103/PhysRevB.57.7556
    def cal_cluster(self, start_traj: int, num_trajs: int, expression: str, cutoff: float=8.1):
        time_all = []
        max_cluster_size_all = []
        num_point_defect_all = []
        num_cluster_defect_all = []
        self.logger.info(f'#------------Cluster analysis, PKA_kin_eng: {self.PKA_kin_eng} eV, cutoff: {cutoff} Å------------#')
        cnt = 0
        for num_traj in range(num_trajs):
            eng_out = os.path.join(self.traj_folder, f'{start_traj+num_traj}', 'eng.out')
            if not os.path.exists(eng_out):
                continue
            cnt += 1
            if cnt > num_trajs:
                break
            self.logger.info(f'Starting the {cnt}th trajectory in folder {start_traj+num_traj}...')
            data = np.loadtxt(eng_out, skiprows=1)
            data = data[::11]   
            timestep = data.T[0]
            time = data.T[1]    
            max_cluster_size = [0]
            num_point_defect = [0]
            num_cluster_defect = [0]

            traj_file = os.path.join(self.traj_folder, f'{start_traj+num_traj}', 'data.output')
            all_pipeline = import_file(traj_file)
            init_frame = all_pipeline.compute(0)
            reference_pipeline = Pipeline(source=StaticSource(data=init_frame))
            for frame_idx in range(1, len(timestep)):
                cur_pipeline = Pipeline(source=StaticSource(data=all_pipeline.compute(frame_idx)))
                wsam = WignerSeitzAnalysisModifier(per_type_occupancies=True, output_displaced=False)
                wsam.reference = reference_pipeline.source
                cur_pipeline.modifiers.append(wsam)
                sel = ExpressionSelectionModifier(expression=expression)
                cur_pipeline.modifiers.append(sel)
                cls = ClusterAnalysisModifier(cutoff=cutoff, sort_by_size=True, only_selected=True)
                cur_pipeline.modifiers.append(cls)
                data = cur_pipeline.compute(0)
                cluster_sizes = data.tables['clusters']['Cluster Size']
                count_dict = Counter(cluster_sizes)  # {cluster size: count}
                cur_pipeline.modifiers.remove(wsam)
                cur_pipeline.modifiers.remove(sel)
                cur_pipeline.modifiers.remove(cls)
                # max cluster size 
                if count_dict:
                    max_cluster_size.append(next(iter(count_dict)))
                else:
                    max_cluster_size.append(0)
                # number of defects 
                sum = 0
                sum_pdefect = 0
                sum_cdefect = 0
                for key, count in count_dict.items():
                    if key == 1:
                        sum_pdefect += count
                    elif key >= 6:
                        sum_cdefect += count
                    sum += count
                if sum == 0:
                    num_point_defect.append(0)
                    num_cluster_defect.append(0)
                else:
                    num_point_defect.append(sum_pdefect/sum*100)    # percentage
                    num_cluster_defect.append(sum_cdefect/sum*100)  # percentage
            time_all.append(time)
            max_cluster_size_all.append(max_cluster_size)
            num_point_defect_all.append(num_point_defect)
            num_cluster_defect_all.append(num_cluster_defect)

        # for max cluster size plot: averaging over trajectories (interpolating to the first trajectory's time points)
        max_cluster_interp_all = []
        num_point_defect_interp_all = []
        num_cluster_defect_interp_all = []
        for time, max_cluster_size, num_pdefect, num_cdefect in zip(time_all, max_cluster_size_all, num_point_defect_all, num_cluster_defect_all):
            max_cluster_interp = np.interp(time_all[0], time, max_cluster_size)
            num_point_defect_interp = np.interp(time_all[0], time, num_pdefect)
            num_cluster_defect_interp = np.interp(time_all[0], time, num_cdefect)
            max_cluster_interp_all.append(max_cluster_interp)
            num_point_defect_interp_all.append(num_point_defect_interp)
            num_cluster_defect_interp_all.append(num_cluster_defect_interp)

        max_cluster_avg = np.mean(max_cluster_interp_all, axis=0)
        max_cluster_std = np.std(max_cluster_interp_all, axis=0)
        with open(os.path.join(self.traj_folder, 'max_cluster.txt'), 'w') as f:
            f.write('Time (ps)\tMax_Cluster_Size\tMax_Cluster_Size_std\n')
            for t, mcs, mcs_std in zip(time_all[0], max_cluster_avg, max_cluster_std):
                f.write(f'{t}\t{int(mcs)}\t{int(mcs_std)}\n')
        max_cluster_low = max_cluster_avg - max_cluster_std
        max_cluster_high = max_cluster_avg + max_cluster_std
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xscale('log')
        ax.plot(time_all[0], max_cluster_avg)
        ax.fill_between(time_all[0], max_cluster_low, max_cluster_high, alpha=0.3)
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Max Cluster Size')
        plt.tight_layout()
        fig.savefig(os.path.join(self.traj_folder, 'max_cluster.png'), dpi=300)

        num_pdefect_avg = np.mean(num_point_defect_interp_all, axis=0)
        num_pdefect_std = np.std(num_point_defect_interp_all, axis=0)
        num_cdefect_avg = np.mean(num_cluster_defect_interp_all, axis=0)
        num_cdefect_std = np.std(num_cluster_defect_interp_all, axis=0)
        with open(os.path.join(self.traj_folder, 'defect_cluster.txt'), 'w') as f:
            f.write('Time (ps)\tNum_Point_Defect\tNum_Point_Defect_std\tNum_Cluster_Defect\tNum_Cluster_Defect_std\n')
            for t, npd, npd_std, ncd, ncd_std in zip(time_all[0], num_pdefect_avg, num_pdefect_std, num_cdefect_avg, num_cdefect_std):
                f.write(f'{t}\t{npd}\t{npd_std}\t{ncd}\t{ncd_std}\n')
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xscale('log')
        ax.plot(time_all[0], num_pdefect_avg, label='Point Defects')
        ax.fill_between(time_all[0], num_pdefect_avg - num_pdefect_std, num_pdefect_avg + num_pdefect_std, alpha=0.3)
        ax.plot(time_all[0], num_cdefect_avg, label='Cluster Defects')
        ax.fill_between(time_all[0], num_cdefect_avg - num_cdefect_std, num_cdefect_avg + num_cdefect_std, alpha=0.3)
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('Number of Defects (%)')
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(self.traj_folder, 'defect_cluster.png'), dpi=300)
        self.logger.info('#------------ Cluster analysis completed. ------------#')
