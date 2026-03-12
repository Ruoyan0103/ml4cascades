import os, math
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
        self.logger.info(f'#------------Defect analysis, PKA_kin_eng: {self.PKA_kin_eng} eV------------#')
        cnt = 0
        traj_dirs = []
        if single_traj_dir is not None:
            traj_dirs.append((single_traj_dir, os.path.basename(single_traj_dir.rstrip(os.sep))))
        else:
            for num_traj in range(num_trajs):
                traj_id = start_traj + num_traj
                if traj_id in exclude_list:
                    self.logger.info(f'Skipping trajectory {traj_id} as it is in the exclude list.')
                    continue
                traj_dir = os.path.join(self.traj_folder, f'{traj_id}')
                traj_dirs.append((traj_dir, traj_id))

        for traj_dir, traj_id in traj_dirs:
            eng_out = os.path.join(traj_dir, 'thermo.out')
            if not os.path.exists(eng_out):
                continue
            cnt += 1
            self.logger.info(f'Starting the {cnt}th trajectory in folder {traj_id}...')
            data = np.loadtxt(eng_out, skiprows=1)
            interval = 1
            mydata = data[::interval] 
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
                with open(os.path.join(traj_dir, 'defect.txt'), 'w') as f:
                    f.write('Time (ps)\tnum_vac\tnum_int\tnum_def\n')
                    for t, v, i, d in zip(time, num_vac, num_int, num_def):
                        f.write(f'{t:.3f}\t{v:.2f}\t{i:.2f}\t{d:.2f}\n') 
        self.logger.info('#------------ Defect analysis completed. ------------#\n')

    def cal_avg_WSDefect(
        self,
        other_traj_folder: str
    ):
        self.logger.info(f'#------------Average defect analysis, PKA_kin_eng: {self.PKA_kin_eng} eV------------#')
        defect_files = []
        for traj_id in os.listdir(self.traj_folder):
            traj_dir = os.path.join(self.traj_folder, traj_id)
            if os.path.isdir(traj_dir):
                defect_file = os.path.join(traj_dir, 'defect.txt')
                if os.path.exists(defect_file):
                    defect_files.append(defect_file)
        for traj_id in os.listdir(other_traj_folder):
            traj_dir = os.path.join(other_traj_folder, traj_id)
            if os.path.isdir(traj_dir):
                defect_file = os.path.join(traj_dir, 'defect.txt')
                if os.path.exists(defect_file):
                    defect_files.append(defect_file)

        time_all = []
        num_vac_all = []
        num_int_all = []
        num_def_all = []
        for defect_file in defect_files:
            data = np.loadtxt(defect_file, skiprows=1)
            time_all.append(data.T[0])
            num_vac_all.append(data.T[1])
            num_int_all.append(data.T[2])
            num_def_all.append(data.T[3])

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

        with open(os.path.join(self.traj_folder, 'defect.txt'), 'w') as f:
            f.write('Time (ps)\tnum_vac\tstd_vac\tnum_int\tstd_int\tnum_def\tstd_def\n')
            for t, v, vstd, i, istd, d, dstd in zip(time_all[0], vac_avg, vac_std, int_avg, int_std, def_avg, def_std):
                f.write(f'{t:.3f}\t{v:.2f}\t{vstd:.2f}\t{i:.2f}\t{istd:.2f}\t{d:.2f}\t{dstd:.2f}\n')


    def cal_cluster_final(
            self, 
            start_traj: int, 
            num_trajs: int,
            exclude_list: list[int],
            expression: str, 
            cutoff: float=8.1,
            single_traj_dir: Optional[str] = None,
        ):
        self.logger.info(f'#------------Cluster analysis, PKA_kin_eng: {self.PKA_kin_eng} eV, cutoff: {cutoff} Å------------#')
        cnt = 0
        traj_dirs = []
        if single_traj_dir is not None:
            traj_dirs.append((single_traj_dir, os.path.basename(single_traj_dir.rstrip(os.sep))))
        else:
            for num_traj in range(num_trajs):
                traj_id = start_traj + num_traj
                if traj_id in exclude_list:
                    self.logger.info(f'Skipping trajectory {traj_id} as it is in the exclude list.')
                    continue
                traj_dir = os.path.join(self.traj_folder, f'{traj_id}')
                traj_dirs.append((traj_dir, traj_id))

        counters = []
        for traj_dir, traj_id in traj_dirs:
            cnt += 1
            self.logger.info(f'Starting the {cnt}th trajectory in folder {traj_id}...')
            traj_file = os.path.join(traj_dir, 'data.output')
            all_pipeline = import_file(traj_file)
            init_frame = all_pipeline.compute(0)
            reference_pipeline = Pipeline(source=StaticSource(data=init_frame))

            last_frame_idx = all_pipeline.source.num_frames - 1
            last_pipeline = Pipeline(source=StaticSource(data=all_pipeline.compute(last_frame_idx)))
            wsam = WignerSeitzAnalysisModifier(per_type_occupancies=True, output_displaced=False)
            wsam.reference = reference_pipeline.source
            last_pipeline.modifiers.append(wsam)
            sel = ExpressionSelectionModifier(expression=expression)
            last_pipeline.modifiers.append(sel)

            cls = ClusterAnalysisModifier(cutoff=cutoff, sort_by_size=True, only_selected=True)
            last_pipeline.modifiers.append(cls)
            data = last_pipeline.compute(0)
            cluster_sizes = data.tables['clusters']['Cluster Size']
            count_dict = Counter(cluster_sizes)   # {cluster size: count}
            counters.append(count_dict)
            with open (os.path.join(traj_dir, 'defect_cluster.txt'), 'w') as f:
                f.write('Cluster Size\tCount\n')
                for size, count in sorted(count_dict.items()):
                    f.write(f"{size}\t{count}\n")

        stats = {}
        all_sizes = set()
        for c in counters:
            all_sizes.update(c.keys())
        for size in all_sizes:
            values = [c.get(size, 0) for c in counters]
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(var)
            stats[size] = (mean, std)

        with open(os.path.join(self.traj_folder, 'defect_cluster.txt'), 'w') as f:
            f.write('Size\tMean\tStd\n')
            for size, (mean, std) in sorted(stats.items()):
                f.write(f"{size}\t{mean:.2f}\t{std:.2f}\n")
        self.logger.info('#------------ Cluster analysis completed. ------------#\n')


    # cutoff from 10.1103/PhysRevB.57.7556
    def cal_cluster_vs_time(
            self, 
            start_traj: int, 
            num_trajs: int,
            exclude_list: list[int],
            expression: str, 
            cutoff: float=8.1,
            single_traj_dir: Optional[str] = None,
        ):
        time_all = []
        max_cluster_size_all = []
        num_point_defect_all = []
        num_cluster_defect_all = []
        self.logger.info(f'#------------Cluster analysis, PKA_kin_eng: {self.PKA_kin_eng} eV, cutoff: {cutoff} Å------------#')
        cnt = 0
        traj_dirs = []
        if single_traj_dir is not None:
            traj_dirs.append((single_traj_dir, os.path.basename(single_traj_dir.rstrip(os.sep))))
        else:
            for num_traj in range(num_trajs):
                traj_id = start_traj + num_traj
                if traj_id in exclude_list:
                    self.logger.info(f'Skipping trajectory {traj_id} as it is in the exclude list.')
                    continue
                traj_dir = os.path.join(self.traj_folder, f'{traj_id}')
                traj_dirs.append((traj_dir, traj_id))

        for traj_dir, traj_id in traj_dirs:
            eng_out = os.path.join(traj_dir, 'thermo.out')
            if not os.path.exists(eng_out):
                continue
            cnt += 1
            self.logger.info(f'Starting the {cnt}th trajectory in folder {traj_id}...')
            data = np.loadtxt(eng_out, skiprows=1)
            interval = 1
            mydata = data[::interval]   
            time = mydata.T[1]    
            max_cluster_size = []
            num_point_defect = []
            num_cluster_defect = []

            traj_file = os.path.join(traj_dir, 'data.output')
            all_pipeline = import_file(traj_file)
            init_frame = all_pipeline.compute(0)
            reference_pipeline = Pipeline(source=StaticSource(data=init_frame))
            for frame_idx in range(0, len(data), interval): 
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
        self.logger.info('#------------ Cluster analysis completed. ------------#\n')

    def cal_liquid_atoms(
        self,
        start_traj: int,
        num_trajs: int,
        expression: str,
        exclude_list: list[int],
        single_traj_dir: Optional[str] = None,
    ):
        time_all = []
        num_liquid_all = []
        self.logger.info(f'#------------Liquid atom analysis, PKA_kin_eng: {self.PKA_kin_eng} eV------------#')
        cnt = 0
        traj_dirs = []
        if single_traj_dir is not None:
            traj_dirs.append((single_traj_dir, os.path.basename(single_traj_dir.rstrip(os.sep))))
        else:
            for num_traj in range(num_trajs):
                traj_id = start_traj + num_traj
                if traj_id in exclude_list:
                    self.logger.info(f'Skipping trajectory {traj_id} as it is in the exclude list.')
                    continue
                traj_dir = os.path.join(self.traj_folder, f'{traj_id}')
                traj_dirs.append((traj_dir, traj_id))

        for traj_dir, traj_id in traj_dirs:
            eng_out = os.path.join(traj_dir, 'thermo.out')
            if not os.path.exists(eng_out):
                continue
            cnt += 1
            self.logger.info(f'Starting the {cnt}th trajectory in folder {traj_id}...')
            
            if os.path.exists(os.path.join(traj_dir, 'liquid.txt')):
                self.logger.info(f'liquid.txt already exists in folder {traj_id}, skipping liquid atom analysis for this trajectory. Reading...')
                time, num_sec = np.loadtxt(os.path.join(traj_dir, 'liquid.txt'), skiprows=1, unpack=True)
                self.logger.warning(f'Time array length in folder {traj_id}: {len(time)} frames are read.')
                time_all.append(time)
                num_liquid_all.append(num_sec)
                continue

            data = np.loadtxt(eng_out, skiprows=1)
            interval = 1
            mydata = data[::interval] 
            time = mydata.T[1]
            num_sec = []
            traj_file = os.path.join(traj_dir, 'data.output')
            all_pipeline = import_file(traj_file)
            for frame_idx in range(0, len(data), interval):
                cur_pipeline = Pipeline(source=StaticSource(data=all_pipeline.compute(frame_idx)))
                sel = ExpressionSelectionModifier(expression=expression)
                cur_pipeline.modifiers.append(sel)
                data = cur_pipeline.compute()
                selection = data.particles['Selection']
                num_selected = np.count_nonzero(selection)
                num_sec.append(num_selected)
 
                with open(os.path.join(traj_dir, 'liquid.txt'), 'w') as f:
                    f.write('Time (ps)\tnum_selected\n')
                    for t, s in zip(time, num_sec):
                        f.write(f'{t:.3f}\t{s:.2f}\n')

            time_all.append(time)
            num_liquid_all.append(num_sec)

        if not time_all:
            self.logger.warning('No valid trajectories found for liquid atom analysis.')
            return

        # for liquid atom plot: averaging liquid atoms over trajectories (interpolating to the first trajectory's time points)
        liquid_interp_all = []
        for time, num_liquid_val in zip(time_all, num_liquid_all):
            # interpolate the number of liquid atoms to the first trajectory's time points
            liquid_interp = np.interp(time_all[0], time, num_liquid_val)
            liquid_interp_all.append(liquid_interp)
        liquid_avg = np.mean(liquid_interp_all, axis=0)
        liquid_std = np.std(liquid_interp_all, axis=0)
        output_dir = single_traj_dir if single_traj_dir is not None else self.traj_folder
        with open(os.path.join(output_dir, 'liquid.txt'), 'w') as f:
            f.write('Time (ps)\tnum_liquid\tstd_liquid\n')
            for t, l, lstd in zip(time_all[0], liquid_avg, liquid_std):
                f.write(f'{t:.3f}\t{l:.2f}\t{lstd:.2f}\n')
        self.logger.info('#------------ Liquid atom analysis completed. ------------#\n')

    def cal_R2(
        self,
        start_traj: int,
        num_trajs: int,
        exclude_list: list[int],
        n0: float, # atomic density in atoms/Å³
        ed: float, # deposited nuclear energy in eV
        single_traj_dir: Optional[str] = None,
    ):
        time_all = []
        R2_all = []
        self.logger.info(f'#------------R2 analysis, PKA_kin_eng: {self.PKA_kin_eng} eV------------#')
        cnt = 0
        traj_dirs = []
        if single_traj_dir is not None:
            traj_dirs.append((single_traj_dir, os.path.basename(single_traj_dir.rstrip(os.sep))))
        else:
            for num_traj in range(num_trajs):
                traj_id = start_traj + num_traj
                if traj_id in exclude_list:
                    self.logger.info(f'Skipping trajectory {traj_id} as it is in the exclude list.')
                    continue
                traj_dir = os.path.join(self.traj_folder, f'{traj_id}')
                traj_dirs.append((traj_dir, traj_id))

        for traj_dir, traj_id in traj_dirs:
            eng_out = os.path.join(traj_dir, 'thermo.out')
            if not os.path.exists(eng_out):
                continue
            cnt += 1
            self.logger.info(f'Starting the {cnt}th trajectory in folder {traj_id}...')
            data = np.loadtxt(eng_out, skiprows=1)
            interval = 1
            mydata = data[::interval] 
            time = mydata.T[1]
            R_frame_values = []
            Q_frame_values = []
            traj_file = os.path.join(traj_dir, 'data.output')
            all_pipeline = import_file(traj_file)
            init_frame = all_pipeline.compute(0)
            init_pos = np.asarray(init_frame.particles.positions).copy()
            init_ids = np.asarray(init_frame.particles['Particle Identifier']).copy()

            init_order = np.argsort(init_ids)
            init_ids_sorted = init_ids[init_order]
            init_pos_sorted = init_pos[init_order]

            for frame_idx in range(0, len(data), interval):
                R2_val = 0
                cur_frame = all_pipeline.compute(frame_idx)
                cur_pos = np.asarray(cur_frame.particles.positions)
                cur_ids = np.asarray(cur_frame.particles['Particle Identifier'])

                cur_order = np.argsort(cur_ids)
                cur_ids_sorted = cur_ids[cur_order]
                cur_pos_sorted = cur_pos[cur_order]
                
                dr = cur_pos_sorted - init_pos_sorted
                cell = np.asarray(cur_frame.cell.matrix)
                Lx, Ly, Lz = cell[0, 0], cell[1, 1], cell[2, 2]
                dr[:, 0] -= Lx * np.round(dr[:, 0] / Lx)
                dr[:, 1] -= Ly * np.round(dr[:, 1] / Ly)
                dr[:, 2] -= Lz * np.round(dr[:, 2] / Lz)

                dr2 = np.sum(dr**2, axis=1)
                R2_val = np.sum(dr2)
                Q_val = R2_val / (6.0 * n0 * ed)
                R_frame_values.append(R2_val)
                Q_frame_values.append(Q_val)
            
            time_all.append(time)
            R2_all.append(R_frame_values)
            with open(os.path.join(traj_dir, 'R2.txt'), 'w') as f:
                f.write('Time (ps)\tR2_val\n')
                for t, r in zip(time, R_frame_values):
                    f.write(f'{t:.3f}\t{r:.2f}\n') 
            # with open(os.path.join(traj_dir, 'Q.txt'), 'w') as f:
            #     f.write('Time (ps)\tQ_val\n')
            #     for t, q in zip(time, Q_frame_values):
            #         f.write(f'{t:.3f}\t{q:.2f}\n')

        R2_interp_all = []
        for time, R2_val in zip(time_all, R2_all):
            R2_interp = np.interp(time_all[0], time, R2_val)
            R2_interp_all.append(R2_interp)

        R2_avg = np.mean(R2_interp_all, axis=0)
        R2_std = np.std(R2_interp_all, axis=0)
        output_dir = single_traj_dir if single_traj_dir is not None else self.traj_folder
        with open(os.path.join(output_dir, 'R2.txt'), 'w') as f:
            f.write('Time (ps)\tavg_R2\tstd_R2\t\n')
            for t, r, rstd in zip(time_all[0], R2_avg, R2_std):
                f.write(f'{t:.3f}\t{r:.2f}\t{rstd:.2f}\n')
        self.logger.info('#------------ R2 analysis completed. ------------#\n')