import os, math, shutil
from pathlib import Path
from typing import Optional
import numpy as np
from ase.io.lammpsrun import read_lammps_dump_text
from matplotlib import pyplot as plt
from ml4cascades.utils import BasicCellInfo
from ovito.io import import_file, export_file
from ovito.pipeline import StaticSource, Pipeline, ReferenceConfigurationModifier
from ovito.modifiers import WignerSeitzAnalysisModifier, ExpressionSelectionModifier, \
                            ClusterAnalysisModifier, CalculateDisplacementsModifier, \
                            DeleteSelectedModifier, IdentifyDiamondModifier, ComputePropertyModifier, CoordinationAnalysisModifier
from collections import Counter, defaultdict
from ml4cascades.loggers.logger import AppLogger

module_dir = os.path.dirname(__file__)
KB = 8.617e-5

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
    lack of other methods for defect analysis
    '''
    @staticmethod
    def _tail_convergence(time, values, tail_ps=100.0):
        mask = time >= (time[-1] - tail_ps)
        t, v = time[mask], values[mask]
        if len(v) < 2:
            return dict(mean=float(v[0]) if len(v) else 0.0, std=0.0,
                        drift_pct=0.0, cv=0.0, n=len(v))
        mean  = float(v.mean())
        std   = float(v.std())
        slope = np.polyfit(t, v, 1)[0]
        drift_pct = abs(slope * (t[-1] - t[0]) / mean * 100) if mean else 0.0
        cv = std / abs(mean) * 100 if mean else 0.0  # coefficient of variation %
        return dict(mean=mean, std=std, drift_pct=drift_pct, cv=cv, n=int(len(v)))

    # Wigner-Seitz method
    def cal_WSDefect(
        self,
        start_traj: int,
        num_trajs: int,
        final: bool,
        reference_traj_file: Optional[str] = None,
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
                traj_dir = os.path.join(self.traj_folder, f'{traj_id}')
                traj_dirs.append((traj_dir, traj_id))

        for traj_dir, traj_id in traj_dirs:
            eng_out = os.path.join(traj_dir, 'thermo.out')
            if not os.path.exists(eng_out):
                self.logger.warning(f'Warning: thermo.out does not exist in folder {traj_id}.')
                continue
            else: 
                data = np.loadtxt(eng_out, skiprows=1)
                time_end = data[-1, 1]
                if time_end <= 10:
                    self.logger.warning(f'Warning: thermo.out in folder {traj_id} runs {time_end} ps.')
                    continue          
            defect_txt = os.path.join(traj_dir, 'defect.txt')
            if os.path.exists(defect_txt):
                data = np.loadtxt(defect_txt, skiprows=1)
                time_end = data[-1, 0]
                if time_end > 40:
                    if not final:
                        self.logger.info(f'defect.txt already exists in folder {traj_id}.')
                        tail_ps = 200
                        t_arr = data[:, 0]
                        for col, name in [(1, 'num_vac'), (2, 'num_int'), (3, 'num_def')]:
                            conv = self._tail_convergence(t_arr, data[:, col], tail_ps=tail_ps)
                            tag = 'CONVERGED' if (conv['drift_pct'] < 6.0 and conv['cv'] < 10.0) else 'NOT CONVERGED'
                            self.logger.info(
                                f'  [{tag}] traj {Path(traj_dir).name} {name}: '
                                f'tail mean={conv["mean"]:.1f} std={conv["std"]:.1f} '
                                f'drift={conv["drift_pct"]:.1f}% cv={conv["cv"]:.1f}% over last {tail_ps} ps '
                                f'({conv["n"]} pts)'
                            )
                    continue
            cnt += 1
            self.logger.info(f'Starting the {cnt}th trajectory in folder {traj_id}...')
            num_vac = []
            num_int = []
            num_def = []
            time = []
            if not final:
                traj_file = os.path.join(traj_dir, 'data.output')
            else:
                traj_file = os.path.join(traj_dir, 'final.output')
            all_pipeline = import_file(traj_file)
            num_frames = all_pipeline.source.num_frames

            if reference_traj_file is not None:
                reference_pipeline = import_file(reference_traj_file)
                reference_frame = reference_pipeline.compute(0)
                reference_pipeline = Pipeline(source=StaticSource(data=reference_frame))
            else:
                init_frame = all_pipeline.compute(0)
                reference_pipeline = Pipeline(source=StaticSource(data=init_frame))
            for frame_idx in range(num_frames):
                frame_data = all_pipeline.compute(frame_idx)
                t = frame_data.attributes.get('Time', frame_idx)
                cur_pipeline = Pipeline(source=StaticSource(data=frame_data))
                wsam = WignerSeitzAnalysisModifier(per_type_occupancies=True, output_displaced=False,
                                                   affine_mapping=ReferenceConfigurationModifier.AffineMapping.ToReference)
                wsam.reference = reference_pipeline.source
                cur_pipeline.modifiers.append(wsam)

                result = cur_pipeline.compute(0)
                cnt_vacancies = result.attributes['WignerSeitz.vacancy_count']
                cnt_interstitials = result.attributes['WignerSeitz.interstitial_count']
                cur_pipeline.modifiers.remove(wsam)
                time.append(t)
                num_vac.append(cnt_vacancies)
                num_int.append(cnt_interstitials)
                num_def.append(cnt_vacancies + cnt_interstitials)
                with open(os.path.join(traj_dir, 'defect.txt'), 'w') as f:
                    f.write('Time (ps)\tnum_vac\tnum_int\tnum_def\n')
                    for t_val, v, ii, d in zip(time, num_vac, num_int, num_def):
                        f.write(f'{t_val:.3f}\t{v:.2f}\t{ii:.2f}\t{d:.2f}\n')

            # --- convergence check (last 50 ps) ---
            if not final:
                tail_ps = 200
                t_arr = np.array(time)
                for arr, name in [(np.array(num_vac), 'num_vac'),
                                (np.array(num_int), 'num_int'),
                                (np.array(num_def), 'num_def')]:
                    conv = self._tail_convergence(t_arr, arr, tail_ps=tail_ps)
                    tag = 'CONVERGED' if (conv['drift_pct'] < 6.0 and conv['cv'] < 10.0) else 'NOT CONVERGED'
                    self.logger.info(
                        f'  [{tag}] traj {traj_id} {name}: '
                        f'tail mean={conv["mean"]:.1f} std={conv["std"]:.1f} '
                        f'drift={conv["drift_pct"]:.1f}% cv={conv["cv"]:.1f}% over last {tail_ps} ps '
                        f'({conv["n"]} pts)'
                )
        self.logger.info('#------------ Defect analysis completed. ------------#\n')

    def cal_avg_WSDefect(
        self,
        outlier_traj: list[int],
        other_traj_folder: str=None,
        other_outlier_traj: list[int]=None
    ):
        self.logger.info(f'#------------Average defect analysis, PKA_kin_eng: {self.PKA_kin_eng} eV------------#')
        defect_files = []
        sorted_traj_ids = sorted([traj_id for traj_id in os.listdir(self.traj_folder) if os.path.isdir(os.path.join(self.traj_folder, traj_id))], key=lambda x: int(x))
        for traj_id in sorted_traj_ids:
            if outlier_traj is not None and int(traj_id) in outlier_traj:
                self.logger.info(f'Skipping trajectory {traj_id} as it is in the outlier list.')
                continue
            traj_dir = os.path.join(self.traj_folder, traj_id)
            if os.path.isdir(traj_dir):
                defect_file = os.path.join(traj_dir, 'defect.txt')
                if os.path.exists(defect_file):
                    data = np.loadtxt(defect_file, skiprows=1)
                    time_end = data[-1, 0]
                    if time_end <= 40:
                        self.logger.warning(f'Warning: {defect_file} runs {time_end} ps.')
                        continue
                    defect_files.append(defect_file)
                    self.logger.info(f'Found defect.txt in {traj_id} running {time_end} ps.')
            else:
                self.logger.warning(f'Warning: {traj_dir} is not a directory.')
        if other_traj_folder is not None:
            sorted_other_traj_ids = sorted([traj_id for traj_id in os.listdir(other_traj_folder) if os.path.isdir(os.path.join(other_traj_folder, traj_id))], key=lambda x: int(x))
            for traj_id in sorted_other_traj_ids:
                if other_outlier_traj is not None and int(traj_id) in other_outlier_traj:
                    self.logger.info(f'Skipping trajectory {traj_id} as it is in the outlier list.')
                    continue
                traj_dir = os.path.join(other_traj_folder, traj_id)
                if os.path.isdir(traj_dir):
                    defect_file = os.path.join(traj_dir, 'defect.txt')
                    if os.path.exists(defect_file):
                        data = np.loadtxt(defect_file, skiprows=1)
                        time_end = data[-1, 0]
                        if time_end <= 40:
                            self.logger.warning(f'Warning: {defect_file} runs {time_end} ps.')
                            continue
                        defect_files.append(defect_file)
                        self.logger.info(f'Found defect.txt in {traj_id} running {time_end} ps.')
                else:
                    self.logger.warning(f'Warning: {traj_dir} is not a directory.')
        self.logger.info(f'#------------Processing {len(defect_files)} files------------#\n')
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

    # cutoff from 10.1103/PhysRevB.57.7556
    def cal_defect_cluster_final(
            self,
            start_traj: int,
            num_trajs: int,
            final: bool,
            reference_traj_file: str,
            cutoff: float = 8.1,   # r_cl; note 1.5*a0 = 8.48 A for a0 = 5.654 A
            large_cluster_min: int = 6,   # Nordlund Table IV threshold
            single_traj_dir: Optional[str] = None,
        ):
            """
            Wigner-Seitz + cluster analysis of the final frame.

            Writes per trajectory:
            vacancy_cluster.txt / interstitial_cluster.txt / defect_cluster.txt
                -- cluster-size histograms (unchanged, for the cluster figures)
            defect_stats.txt
                -- one-line summary with the table columns:
                    N_FP, F_i_isol, F_v_isol, F_i_clus, F_v_clus, Cl_max

            Fractions follow Nordlund et al. PRB 57, 7556 (1998), Table IV:
            percentages of the TOTAL number of W-S defects (vacant sites +
            multiply occupied sites); 'isolated' = mixed-type cluster of size 1,
            'clustered' = member of a mixed-type cluster with >= large_cluster_min
            defects. Clusters of size 2..(large_cluster_min-1) are in neither
            group, so F_isol + F_clus < 100 %.
            """
            self.logger.info(f'#------------Cluster analysis, PKA_kin_eng: {self.PKA_kin_eng} eV------------#')
            cnt = 0
            traj_dirs = []
            if single_traj_dir is not None:
                traj_dirs.append((single_traj_dir, os.path.basename(single_traj_dir.rstrip(os.sep))))
            else:
                for num_traj in range(num_trajs):
                    traj_id = start_traj + num_traj
                    traj_dir = os.path.join(self.traj_folder, f'{traj_id}')
                    traj_dirs.append((traj_dir, traj_id))

            for traj_dir, traj_id in traj_dirs:
                eng_out = os.path.join(traj_dir, 'thermo.out')
                if not os.path.exists(eng_out):
                    self.logger.warning(f'Warning: thermo.out does not exist in folder {traj_id}.')
                    continue
                else:
                    data = np.loadtxt(eng_out, skiprows=1)
                    time_end = data[-1, 1]
                    if time_end <= 40:
                        self.logger.warning(f'Warning: thermo.out in folder {traj_id} runs {time_end} ps.')
                        continue
                stats_txt = os.path.join(traj_dir, 'defect_stats.txt')
                if os.path.exists(stats_txt):
                    self.logger.info(f'defect_stats.txt already exists in folder {traj_id}.')
                    #continue
                cnt += 1
                self.logger.info(f'Cluster: Starting the {cnt}th trajectory in folder {traj_id}...')
                if not final:
                    traj_file = os.path.join(traj_dir, 'data.output')
                else:
                    traj_file = os.path.join(traj_dir, 'final.output')
                all_pipeline = import_file(traj_file)

                if reference_traj_file is not None:
                    reference_pipeline = import_file(reference_traj_file)
                    reference_frame = reference_pipeline.compute(0)
                    reference_pipeline = Pipeline(source=StaticSource(data=reference_frame))
                else:
                    init_frame = all_pipeline.compute(0)
                    reference_pipeline = Pipeline(source=StaticSource(data=init_frame))

                last_frame_idx = all_pipeline.source.num_frames - 1
                last_pipeline = Pipeline(source=StaticSource(data=all_pipeline.compute(last_frame_idx)))
                wsam = WignerSeitzAnalysisModifier(per_type_occupancies=True, output_displaced=False, 
                                                  affine_mapping=ReferenceConfigurationModifier.AffineMapping.ToReference)
                wsam.reference = reference_pipeline.source
                last_pipeline.modifiers.append(wsam)

                # --- Same-type cluster-size histograms (unchanged, for figures) ---
                sel_vac = ExpressionSelectionModifier(expression='Occupancy == 0')
                last_pipeline.modifiers.append(sel_vac)
                cls_vac = ClusterAnalysisModifier(cutoff=cutoff, sort_by_size=True, only_selected=True)
                last_pipeline.modifiers.append(cls_vac)
                data = last_pipeline.compute(0)
                vac_cluster_sizes = data.tables['clusters']['Cluster Size']
                last_pipeline.modifiers.remove(cls_vac)
                last_pipeline.modifiers.remove(sel_vac)

                sel_int = ExpressionSelectionModifier(expression='Occupancy > 1')
                last_pipeline.modifiers.append(sel_int)
                cls_int = ClusterAnalysisModifier(cutoff=cutoff, sort_by_size=True, only_selected=True)
                last_pipeline.modifiers.append(cls_int)
                data = last_pipeline.compute(0)
                int_cluster_sizes = data.tables['clusters']['Cluster Size']
                last_pipeline.modifiers.remove(cls_int)
                last_pipeline.modifiers.remove(sel_int)

                # --- Mixed-type clustering: basis of the table statistics ---
                sel_def = ExpressionSelectionModifier(expression='Occupancy != 1')
                last_pipeline.modifiers.append(sel_def)
                cls_def = ClusterAnalysisModifier(cutoff=cutoff, sort_by_size=True, only_selected=True)
                last_pipeline.modifiers.append(cls_def)
                data = last_pipeline.compute(0)
                def_cluster_sizes = np.array(data.tables['clusters']['Cluster Size'])

                # Per-site properties for species-resolved statistics.
                occupancy = np.array(data.particles['Occupancy']).reshape(-1)   # single species
                cluster_id = np.array(data.particles['Cluster'])                # 0 = not a defect

                is_vac = (occupancy == 0)
                is_int = (occupancy > 1)          # counted as SITES (see caption note)
                n_vac = int(np.sum(is_vac))
                n_int = int(np.sum(is_int))
                n_def = n_vac + n_int             # denominator of all fractions
                n_extra_atoms = int(np.sum(occupancy[is_int] - 1))
                n_fp = n_vac                       # = extra atoms in a closed cell

                # Size of the cluster each defect site belongs to (index by id; id>=1).
                sizes_by_id = np.concatenate(([0], def_cluster_sizes))  # id 0 -> size 0
                site_cluster_size = sizes_by_id[cluster_id]

                isolated = (site_cluster_size == 1)
                in_large = (site_cluster_size >= large_cluster_min)
                f_i_isol = 100.0 * np.sum(isolated & is_int) / n_def if n_def else 0.0
                f_v_isol = 100.0 * np.sum(isolated & is_vac) / n_def if n_def else 0.0
                f_i_clus = 100.0 * np.sum(in_large & is_int) / n_def if n_def else 0.0
                f_v_clus = 100.0 * np.sum(in_large & is_vac) / n_def if n_def else 0.0
                cl_max = int(def_cluster_sizes.max()) if def_cluster_sizes.size else 0

                last_pipeline.modifiers.remove(cls_def)
                last_pipeline.modifiers.remove(sel_def)

                with open(os.path.join(traj_dir, 'vacancy_cluster.txt'), 'w') as f:
                    f.write('Cluster size\tNumber of clusters\n')
                    for size, count in Counter(vac_cluster_sizes).items():
                        f.write(f'{size}\t{count}\n')
                with open(os.path.join(traj_dir, 'interstitial_cluster.txt'), 'w') as f:
                    f.write('Cluster size\tNumber of clusters\n')
                    for size, count in Counter(int_cluster_sizes).items():
                        f.write(f'{size}\t{count}\n')
                with open(os.path.join(traj_dir, 'defect_cluster.txt'), 'w') as f:
                    f.write('Cluster size\tNumber of clusters\n')
                    for size, count in Counter(def_cluster_sizes).items():
                        f.write(f'{size}\t{count}\n')
                with open(stats_txt, 'w') as f:
                    f.write('N_FP\tN_vac\tN_int_sites\tN_extra_atoms\t'
                            'F_i_isol(%)\tF_v_isol(%)\tF_i_clus(%)\tF_v_clus(%)\tCl_max\n')
                    f.write(f'{n_fp}\t{n_vac}\t{n_int}\t{n_extra_atoms}\t'
                            f'{f_i_isol:.2f}\t{f_v_isol:.2f}\t{f_i_clus:.2f}\t{f_v_clus:.2f}\t{cl_max}\n')
                self.logger.info(f'#------------ Defect cluster analysis completed. ------------#\n')


    def summarize_defect_stats(self, start_traj: int, num_trajs: int):
        """Aggregate defect_stats.txt over trajectories into one table row:
        means +- standard error of the mean; Cl_max = max over ALL trajectories."""
        rows = []
        for num_traj in range(num_trajs):
            traj_id = start_traj + num_traj
            stats_txt = os.path.join(self.traj_folder, f'{traj_id}', 'defect_stats.txt')
            if not os.path.exists(stats_txt):
                self.logger.warning(f'Warning: defect_stats.txt missing in folder {traj_id}.')
                continue
            rows.append(np.loadtxt(stats_txt, skiprows=1))
        if not rows:
            self.logger.warning('No defect_stats.txt files found; nothing to summarize.')
            return None
        arr = np.atleast_2d(np.array(rows))

        def mean_sem(col):
            x = arr[:, col]
            sem = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
            return x.mean(), sem

        n_fp, n_fp_e = mean_sem(0)
        fi_i, fi_i_e = mean_sem(4)
        fv_i, fv_i_e = mean_sem(5)
        fi_c, fi_c_e = mean_sem(6)
        fv_c, fv_c_e = mean_sem(7)
        cl_max_all = int(arr[:, 8].max())
        n = len(arr)
        self.logger.info(
            f'[{self.PKA_kin_eng} eV, {n} trajs]  '
            f'N_FP = {n_fp:.0f} +- {n_fp_e:.0f}   '
            f'F_i_isol = {fi_i:.1f} +- {fi_i_e:.1f} %   '
            f'F_v_isol = {fv_i:.1f} +- {fv_i_e:.1f} %   '
            f'F_i_clus = {fi_c:.0f} +- {fi_c_e:.0f} %   '
            f'F_v_clus = {fv_c:.0f} +- {fv_c_e:.0f} %   '
            f'Cl_max = {cl_max_all}')
        return n_fp, n_fp_e, fi_i, fi_i_e, fv_i, fv_i_e, fi_c, fi_c_e, fv_c, fv_c_e, cl_max_all

    def cal_defect_cluster_final_avg(
        self,
        outlier_traj: list[int],
        other_traj_folder: str,
        other_outlier_traj: list[int]
    ):
        self.logger.info(f'#------------Average cluster analysis, PKA_kin_eng: {self.PKA_kin_eng} eV------------#')
        cluster_files = []
        sorted_traj_ids = sorted([traj_id for traj_id in os.listdir(self.traj_folder) if os.path.isdir(os.path.join(self.traj_folder, traj_id))], key=lambda x: int(x))
        for traj_id in sorted_traj_ids:
            if outlier_traj is not None and int(traj_id) in outlier_traj:
                self.logger.info(f'Skipping trajectory {traj_id} as it is in the outlier list.')
                continue
            traj_dir = os.path.join(self.traj_folder, traj_id)
            if os.path.isdir(traj_dir):
                cluster_file = os.path.join(traj_dir, 'defect_cluster.txt')
                cluster_files.append(cluster_file)
        if other_traj_folder is not None:
            sorted_other_traj_ids = sorted([traj_id for traj_id in os.listdir(other_traj_folder) if os.path.isdir(os.path.join(other_traj_folder, traj_id))], key=lambda x: int(x))
            for traj_id in sorted_other_traj_ids:
                if other_outlier_traj is not None and int(traj_id) in other_outlier_traj:
                    self.logger.info(f'Skipping trajectory {traj_id} as it is in the outlier list.')
                    continue
                traj_dir = os.path.join(other_traj_folder, traj_id)
                if os.path.isdir(traj_dir):
                    cluster_file = os.path.join(traj_dir, 'defect_cluster.txt')
                    cluster_files.append(cluster_file)

        self.logger.info(f'#------------Processing {len(cluster_files)} files------------#\n')
        acc = defaultdict(float)
        nfiles = len(cluster_files)
        presence = defaultdict(int)       # NEW: for possibility (presence frequency)
        for cluster_file in cluster_files:
            data = np.loadtxt(cluster_file, skiprows=1)
            sizes = data.T[0]
            counts = data.T[1]
            local = dict(zip(sizes, counts))

            for s in local:
                acc[s] += local[s]

            for s in local.keys():
                presence[s] += 1

        all_sizes = sorted(acc.keys())
        avg_sizes = []
        avg_counts = []
        possibility = []
        for s in all_sizes:
            total = 0.0
            for cluster_file in cluster_files:
                data = np.loadtxt(cluster_file, skiprows=1)
                sizes = data[:, 0].astype(int)
                counts = data[:, 1]
                local = dict(zip(sizes, counts))
                total += local.get(s, 0.0)   

            avg_sizes.append(s)
            avg_counts.append(total / nfiles)
            possibility.append(presence[s] / nfiles)
            
        avg_sizes = np.array(avg_sizes)
        avg_counts = np.array(avg_counts)
        possibility = np.array(possibility)

        idx = np.where(possibility > 0.5)[0][-1] # find the last index where possibility > 0.2
        print('Cluster sizes with presence frequency around 0.5:', avg_sizes[idx])

        plt.figure(figsize=(6, 4))
        plt.plot(all_sizes, possibility, '-o', label='Presence Frequency', markersize=2)
        plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='0.5 threshold')
        # add text annotation for the cluster size at 0.2 threshold
        plt.text(avg_sizes[idx], 0.5, f'Cluster size: {avg_sizes[idx]:.0f}', fontsize=8, verticalalignment='bottom', horizontalalignment='right')

        plt.xscale('log')
        plt.xlabel('Cluster Size')
        plt.ylabel('Presence Frequency')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(self.traj_folder, 'defect_cluster_summary.png'), dpi=300)
        with open(os.path.join(self.traj_folder, 'defect_cluster_summary.txt'), 'w') as f:
            f.write('Cluster Size\tAverage Cluster Number\n')
            for s, c in zip(avg_sizes, avg_counts):
                f.write(f'{s}\t{c:.2f}\n')

        print('total defects:', np.sum(avg_sizes * avg_counts))
        self.logger.info(f'#------------ Average defect cluster analysis completed. ------------#\n')

    def cal_liquid_atoms(
            self,
            start_traj: int,
            num_trajs: int,
            single_traj_dir: Optional[str] = None,
            cn_cutoff: float = 3.5,   # first minimum of liquid-Ge RDF; crystal 2nd shell at 4.0 A
        ):
            self.logger.info(f'#------------Liquid atoms calculation, PKA_kin_eng: {self.PKA_kin_eng} eV------------#')
            cnt = 0
            traj_dirs = []
            if single_traj_dir is not None:
                traj_dirs.append((single_traj_dir, os.path.basename(single_traj_dir.rstrip(os.sep))))
            else:
                for num_traj in range(num_trajs):
                    traj_id = start_traj + num_traj
                    traj_dir = os.path.join(self.traj_folder, f'{traj_id}')
                    traj_dirs.append((traj_dir, traj_id))

            for traj_dir, traj_id in traj_dirs:
                eng_out = os.path.join(traj_dir, 'thermo.out')
                if not os.path.exists(eng_out):
                    self.logger.warning(f'Warning: thermo.out does not exist in folder {traj_id}.')
                    continue
                else:
                    data = np.loadtxt(eng_out, skiprows=1)
                    time_end = data[-1, 1]
                    if time_end <= 40:
                        self.logger.warning(f'Warning: thermo.out in folder {traj_id} runs {time_end} ps.')
                        continue
                amorphous_txt = os.path.join(traj_dir, 'liquid_atoms.txt')
                if os.path.exists(amorphous_txt):
                    os.rename(amorphous_txt, os.path.join(traj_dir, 'liquid_atoms_first_neighbor_cut_2.7.txt'))
                    self.logger.info(f'liquid_atoms.txt already exists in folder {traj_id}.')
                    #continue
                cnt += 1
                num_liquid_atoms = []
                num_other_atoms = []
                mean_cn_liquid_list = []
                mean_cn_crystal_list = []
                time = []
                self.logger.info(f'Starting the {cnt}th trajectory in folder {traj_id}...')
                traj_file = os.path.join(traj_dir, 'data.output')
                all_pipeline = import_file(traj_file)
                all_pipeline_data = all_pipeline.compute()
                total_num_atoms = all_pipeline_data.particles.count
                for frame_idx in range(100):
                    frame_data = all_pipeline.compute(frame_idx)
                    t = frame_data.attributes.get('Time', frame_idx)
                    time.append(t)
                    cur_pipeline = Pipeline(source=StaticSource(data=frame_data))
                    cur_pipeline.modifiers.append(IdentifyDiamondModifier())
                    # Compute neighbor-averaged KE following Nordlund & Averback PRB 56, 2421 (1997):
                    # local_ek(i) = [KE(i) + sum_j KE(j)] / (1 + NumNeighbors)
                    cur_pipeline.modifiers.append(ComputePropertyModifier(
                        output_property = 'local_ek',
                        expressions = ['c_ek / (NumNeighbors + 1)'],
                        neighbor_expressions =  ['c_ek / (NumNeighbors + 1)'],
                        cutoff_radius = 2.9
                    ))
                    # Coordination analysis on the FULL cell (no atoms deleted), so that
                    # liquid atoms at the melt boundary keep their crystalline neighbors
                    # in the neighbor count. Subset averaging is done afterwards in numpy.
                    cur_pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff = cn_cutoff))
                    data = cur_pipeline.compute(0)
                    struct_type = np.array(data.particles['Structure Type'])
                    local_ek = np.array(data.particles['local_ek'])
                    cn = np.array(data.particles['Coordination'])
                    liquid_mask = (struct_type != 1) & (local_ek > 0.15)  # 3/2*k_B*melting_point:1200 K = 0.15 eV
                    crystal_mask = (struct_type == 1)
                    diamond_atoms = int(np.sum(crystal_mask))
                    liquid_atoms = int(np.sum(liquid_mask))
                    other_atoms = total_num_atoms - diamond_atoms
                    # Mean first-shell CN of the liquid subset; NaN if no liquid atoms in frame.
                    mean_cn_liquid = float(cn[liquid_mask].mean()) if liquid_atoms > 0 else float('nan')
                    # Sanity reference: crystal population must average ~4.0-4.2 at 3.5 A.
                    mean_cn_crystal = float(cn[crystal_mask].mean()) if diamond_atoms > 0 else float('nan')
                    num_liquid_atoms.append(liquid_atoms)
                    num_other_atoms.append(other_atoms)
                    mean_cn_liquid_list.append(mean_cn_liquid)
                    mean_cn_crystal_list.append(mean_cn_crystal)
                with open(os.path.join(traj_dir, 'liquid_atoms.txt'), 'w') as f:
                    f.write('Time (ps)\tLiquid Atoms\tOther Atoms\tMean CN (liquid)\tMean CN (crystal)\n')
                    for t, n_liq, n_oth, cn_liq, cn_cry in zip(
                            time, num_liquid_atoms, num_other_atoms,
                            mean_cn_liquid_list, mean_cn_crystal_list):
                        f.write(f'{t:.3f}\t{n_liq}\t{n_oth}\t{cn_liq:.3f}\t{cn_cry:.3f}\n')
            self.logger.info(f'#------------ Liquid Atoms calculation completed. ------------#\n')

    def cal_R2(
        self,
        start_traj: int,
        num_trajs: int,
        single_traj_dir: Optional[str] = None,
    ):
        self.logger.info(f'#------------R2 calculation, PKA_kin_eng: {self.PKA_kin_eng} eV------------#')
        a = 5.76
        nearest_neighbor_dist = np.sqrt(3)/4 * a 
        threshold_dr2 = nearest_neighbor_dist ** 2
        cnt = 0
        traj_dirs = []
        if single_traj_dir is not None:
            traj_dirs.append((single_traj_dir, os.path.basename(single_traj_dir.rstrip(os.sep))))
        else:
            for num_traj in range(num_trajs):
                traj_id = start_traj + num_traj
                traj_dir = os.path.join(self.traj_folder, f'{traj_id}')
                traj_dirs.append((traj_dir, traj_id))

        for traj_dir, traj_id in traj_dirs:
            eng_out = os.path.join(traj_dir, 'thermo.out')
            if not os.path.exists(eng_out):
                self.logger.warning(f'Warning: thermo.out does not exist in folder {traj_id}.')
                continue
            else: 
                data = np.loadtxt(eng_out, skiprows=1)
                time_end = data[-1, 1]
                if time_end <= 40:
                    self.logger.warning(f'Warning: thermo.out in folder {traj_id} runs {time_end} ps.')
                    continue          
            R2_txt = os.path.join(traj_dir, 'R2.txt')
            if os.path.exists(R2_txt):
                data = np.loadtxt(R2_txt, skiprows=1)
                time_end = data[-1, 0]
                if time_end > 40:
                    self.logger.info(f'R2.txt already exists in folder {traj_id}.')
                    continue
            cnt += 1
            self.logger.info(f'Starting the {cnt}th trajectory in folder {traj_id}...')
            data = np.loadtxt(eng_out, skiprows=1)
            interval = 1
            mydata = data[::interval] 
            time = mydata.T[1]
            R2_list = []
            num_atoms_list = []
            atoms_id_list = []

            traj_file = os.path.join(traj_dir, 'data.output')
            all_pipeline = import_file(traj_file)
            init_frame = all_pipeline.compute(0)
            init_pos = np.asarray(init_frame.particles.positions).copy()
            init_ids = np.asarray(init_frame.particles['Particle Identifier']).copy()
            init_order = np.argsort(init_ids)
            init_ids_sorted = init_ids[init_order]
            init_pos_sorted = init_pos[init_order]

            init_com = np.mean(init_pos_sorted, axis=0)
            init_pos_centered = init_pos_sorted - init_com

            for frame_idx in range(0, len(data), interval):
                R2_val = 0
                cur_frame = all_pipeline.compute(frame_idx)
                cur_pos = np.asarray(cur_frame.particles.positions).copy()
                cur_ids = np.asarray(cur_frame.particles['Particle Identifier']).copy()

                cur_order = np.argsort(cur_ids)
                cur_ids_sorted = cur_ids[cur_order]
                cur_pos_sorted = cur_pos[cur_order]

                cur_com = np.mean(cur_pos_sorted, axis=0)
                cur_pos_centered = cur_pos_sorted - cur_com
                
                dr = cur_pos_sorted - init_pos_sorted
                # dr = cur_pos_centered - init_pos_centered
                cell = np.asarray(cur_frame.cell.matrix)
                Lx, Ly, Lz = cell[0, 0], cell[1, 1], cell[2, 2]
                # np.round(nearest_neighbor_dist) = 2
                dr[:, 0] = np.where(np.abs(np.round(np.abs(dr[:, 0]) - Lx)) <= 2, np.abs(dr[:, 0]) - Lx, dr[:, 0])
                dr[:, 1] = np.where(np.abs(np.round(np.abs(dr[:, 1]) - Ly)) <= 2, np.abs(dr[:, 1]) - Ly, dr[:, 1])
                dr[:, 2] = np.where(np.abs(np.round(np.abs(dr[:, 2]) - Lz)) <= 2, np.abs(dr[:, 2]) - Lz, dr[:, 2])

                dr2 = np.sum(dr**2, axis=1)
                mask = dr2 >= threshold_dr2
                frame_ids = [i for i in range(len(mask)) if mask[i] == 1]
                num_atoms = np.sum(mask)
                R2_val = np.sum(dr2[mask]) 
                R2_list.append(R2_val)
                num_atoms_list.append(num_atoms)
                atoms_id_list.append(frame_ids)

            with open(os.path.join(traj_dir, 'R2.txt'), 'w') as f:
                f.write('Time (ps)\tR2\tnum_atoms\tatoms_ids\n')
                # for t, r2, n, id_list in zip(time, R2_list, num_atoms_list, atoms_id_list):
                #     f.write(f'{t:.3f}\t{r2:.2f}\t{n:.0f}\t{id_list}\n') 
                for t, r2, n in zip(time, R2_list, num_atoms_list):
                    f.write(f'{t:.3f}\t{r2:.2f}\t{n:.0f}\n') 
        self.logger.info('#------------ R2 analysis completed. ------------#\n')

    def cal_R2_ovito(
        self,
        start_traj: int,
        num_trajs: int,
        reference_traj_file: Optional[str] = None,
        single_traj_dir: Optional[str] = None,
    ):
        a = 5.65
        nearest_neighbor_dist = np.sqrt(3)/4 * a
        expression = f'DisplacementMagnitude < {nearest_neighbor_dist:.2f}'  # atoms within nearest-neighbor distance
        self.logger.info(f'#------------R2 ovito analysis, PKA_kin_eng: {self.PKA_kin_eng} eV------------#')
        cnt = 0
        traj_dirs = []
        if single_traj_dir is not None:
            traj_dirs.append((single_traj_dir, os.path.basename(single_traj_dir.rstrip(os.sep))))
        else:
            for num_traj in range(num_trajs):
                traj_id = start_traj + num_traj
                traj_dir = os.path.join(self.traj_folder, f'{traj_id}')
                traj_dirs.append((traj_dir, traj_id))

        for traj_dir, traj_id in traj_dirs:
            eng_out = os.path.join(traj_dir, 'thermo.out')
            if not os.path.exists(eng_out):
                self.logger.warning(f'Warning: thermo.out does not exist in folder {traj_id}.')
                continue
            data = np.loadtxt(eng_out, skiprows=1)
            time_end = data[-1, 1]
            if time_end <= 40:
                self.logger.warning(f'Warning: thermo.out in folder {traj_id} runs {time_end} ps.')
                traj_path = Path(traj_dir)
                if traj_path.exists() and traj_path.is_dir():
                    shutil.rmtree(traj_path)
                self.logger.info(f'Trajectory {traj_id} has been removed due to short simulation time.')
                continue

            R2_txt = os.path.join(traj_dir, 'R2-ovito.txt')
            if os.path.exists(R2_txt):
                data = np.loadtxt(R2_txt, skiprows=1)
                if data[-1, 0] > 40:
                    self.logger.info(f'R2-ovito.txt already exists in folder {traj_id}.')
                    continue

            cnt += 1
            self.logger.info(f'Starting the {cnt}th trajectory in folder {traj_id}...')

            traj_file = os.path.join(traj_dir, 'data.output')
            all_pipeline = import_file(traj_file)
            num_frames = all_pipeline.source.num_frames

            if reference_traj_file is not None:
                reference_pipeline = import_file(reference_traj_file)
                reference_frame = reference_pipeline.compute(0)
                ref_source = Pipeline(source=StaticSource(data=reference_frame))
            else:
                init_frame = all_pipeline.compute(0)
                ref_source = Pipeline(source=StaticSource(data=init_frame))

            # Build pipeline with modifiers once per trajectory
            dispm = CalculateDisplacementsModifier(affine_mapping=ReferenceConfigurationModifier.AffineMapping.ToReference)
            dispm.reference = ref_source.source
            all_pipeline.modifiers.append(dispm)
            all_pipeline.modifiers.append(ExpressionSelectionModifier(expression=expression))

            time_list = []
            R2_ovito_list = []
            selected_atoms_list = []
            self.logger.info(f'  Traj {traj_id}: processing {num_frames} frames')
            with open(R2_txt, 'w') as f:
                f.write('Time (ps)\tR2\tSelected Atoms\n')
                for frame_idx in range(num_frames):
                    data = all_pipeline.compute(frame_idx)
                    t = data.attributes.get('Time', frame_idx)
                    selection = np.array(data.particles['Selection'], dtype=bool)
                    num_selected = int(selection.sum())
                    displacements = np.array(data.particles['Displacement Magnitude'])
                    R2 = float(np.sum(displacements[~selection] ** 2))
                    f.write(f'{t:.3f}\t{R2:.2f}\t{num_selected}\n')
                    f.flush()

        self.logger.info('#------------ R2 analysis completed. ------------#\n')

    def cal_avg_R2(
            self,
            other_traj_folder: str
        ):
            self.logger.info(f'#------------Average R2 analysis, PKA_kin_eng: {self.PKA_kin_eng} eV------------#')
            R2_files = []
            sorted_traj_ids = sorted([traj_id for traj_id in os.listdir(self.traj_folder) if os.path.isdir(os.path.join(self.traj_folder, traj_id))], key=lambda x: int(x))
            for traj_id in sorted_traj_ids:
                traj_dir = os.path.join(self.traj_folder, traj_id)
                if os.path.isdir(traj_dir):
                    R2_file = os.path.join(traj_dir, 'R2-ovito.txt')
                    if os.path.exists(R2_file):
                        data = np.loadtxt(R2_file, skiprows=1)
                        time_end = data[-1, 0]
                        if time_end <= 40:
                            self.logger.warning(f'Warning: {R2_file} runs {time_end} ps.')
                            continue
                        R2_files.append(R2_file)
                        self.logger.info(f'Found R2-ovito.txt in {traj_id} running {time_end} ps.')
            if other_traj_folder is not None:
                sorted_other_traj_ids = sorted([traj_id for traj_id in os.listdir(other_traj_folder) if os.path.isdir(os.path.join(other_traj_folder, traj_id))], key=lambda x: int(x))
                for traj_id in sorted_other_traj_ids:
                    traj_dir = os.path.join(other_traj_folder, traj_id)
                    if os.path.isdir(traj_dir):
                        defect_file = os.path.join(traj_dir, 'R2-ovito.txt')
                        if os.path.exists(defect_file):
                            data = np.loadtxt(defect_file, skiprows=1)
                            time_end = data[-1, 0]
                            if time_end <= 40:
                                self.logger.warning(f'Warning: {defect_file} runs {time_end} ps.')
                                continue
                            R2_files.append(defect_file)
                            self.logger.info(f'Found R2-ovito.txt in {traj_id} running {time_end} ps.')
            self.logger.info(f'#------------Processing {len(R2_files)} files------------#\n')
            time_all = []
            R2_all = []
            if len(R2_files) != 0:
                for R2_file in R2_files:
                    data = np.loadtxt(R2_file, skiprows=1)
                    time_all.append(data.T[0])
                    R2_all.append(data.T[1])

                R2_interp_all = []
                for time, R2_val in zip(time_all, R2_all):
                    R2_interp = np.interp(time_all[0], time, R2_val)
                    R2_interp_all.append(R2_interp)

                R2_avg = np.mean(R2_interp_all, axis=0)
                R2_std = np.std(R2_interp_all, axis=0)
                with open(os.path.join(self.traj_folder, 'R2.txt'), 'w') as f:
                    f.write('Time (ps)\tR2\tstd_R2\n')
                    for t, r2, r2_std in zip(time_all[0], R2_avg, R2_std):
                        f.write(f'{t:.3f}\t{r2:.2f}\t{r2_std:.2f}\n')
