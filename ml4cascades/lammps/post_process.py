from ml4cascades.lammps import LMPStaticCalculator
import os, subprocess, math, shutil, time
import numpy as np
from ase.build import bulk
from ase.io import read, write
from ovito.io import import_file
from ovito.modifiers import WignerSeitzAnalysisModifier, ClusterAnalysisModifier, ExpressionSelectionModifier
from ovito.modifiers import CalculateDisplacementsModifier
from ovito.pipeline import StaticSource, Pipeline
from collections import Counter
import logging
import multiprocessing as mp
import glob

AMU_TO_KG = 1.66053906660E-27  # Atomic mass unit to kg conversion factor
JOULE_TO_EV = 6.241509074E18   # Joule to eV conversion factor
ANGSTROM_TO_METER = 1E-10      # Angstroms/picosecond to meters/second conversion factor
PS_TO_S = 1E-12                # Picoseconds to seconds conversion factor
module_dir = os.path.dirname(__file__)

# Constant atomic number density for germanium
n0 = 0.0442  # atoms/Å³

# Lookup table for ionization and damage energies
energy_lookup = {
    100: (14.2, 85.8),    # (ionization, damage)
    400: (61.8, 338.2),
    1000: (168.2, 831.8),
    2000: (361.4, 1638.6),
    5000: (1016, 3984),
    10000: (2420, 7580),
    20000: (6330, 13670),
    50000: (22300, 27700)
}

def process_hkl(calc, energy, idx, hkl, eng_dir):
    """
    Process a single hkl direction for post-processing and calculate mixing ratio Q.
    Args:
        calc: CascadeCalculator instance.
        energy: Energy value in eV.
        idx: Index of the hkl direction.
        hkl: Miller indices array.
        eng_dir: Directory containing the relaxed structure (e.g., calculations/energy).
    Returns:
        List containing [cnt_vacancies, cnt_interstitials, vac_cluster_dist, inter_cluster_dist,
                        time_data, Q, total_squared_displacement_last] or None if invalid.
    """
    eng_hkl_dir = os.path.join(eng_dir, str(idx))
    trajectory_file = os.path.join(eng_hkl_dir, 'dump.PKA')
    data_file = os.path.join(eng_hkl_dir, 'data.txt')

    # Check if simulation reached 30 ps
    if os.path.exists(data_file):
        try:
            data = np.unique(np.loadtxt(data_file, skiprows=1), axis=0)
            Time = data[:, 1]  # Second column is Time
            if Time[-1] < 30:
                calc.logger.info(f"Skipping hkl {hkl} (index {idx}) for energy {energy} eV: final time {Time[-1]} ps < 30 ps.")
                return None
        except Exception as e:
            calc.logger.warning(f"Failed to read data.txt from {eng_hkl_dir}: {e}. Skipping.")
            return None
    else:
        calc.logger.info(f"data.txt not found for energy {energy} eV, hkl {hkl}. Skipping.")
        return None

    if not os.path.exists(trajectory_file):
        calc.logger.info(f"Trajectory file {trajectory_file} not found for energy {energy} eV, hkl {hkl}. Skipping.")
        return None

    all_pipeline = import_file(trajectory_file)
    # Use first frame of dump.PKA for initial positions
    initial_data = all_pipeline.compute(0)
    initial_positions = initial_data.particles.positions

    # Time-dependent analysis
    times, vacancies, interstitials, temperatures, total_squared_displacement = calc._timeDependentAnalysis(all_pipeline, all_pipeline, eng_hkl_dir)
    if not times:
        return None

    # Analysis for last frame
    last_frame = all_pipeline.compute(all_pipeline.source.num_frames - 1)
    pipeline = Pipeline(source=StaticSource(data=last_frame))
    cnt_vacancies, cnt_interstitials = calc._countVacAndInter(pipeline, all_pipeline)

    # Cluster analysis for vacancies
    vac_cluster_dist = calc._clustersAndDXA(pipeline, all_pipeline, 'Occupancy == 0')

    # Cluster analysis for interstitials
    inter_cluster_dist = calc._clustersAndDXA(pipeline, all_pipeline, 'Occupancy > 1')

    # Calculate mixing ratio Q
    # Add the modifier at the start of the pipeline
    threshold = 1.225
    modifier = CalculateDisplacementsModifier(
        reference_frame=0,
        minimum_image_convention=True  # Recommended for PBC handling
    )
    all_pipeline.modifiers.append(modifier)

    # Compute last frame (final cascade state)
    last_frame = all_pipeline.compute(all_pipeline.source.num_frames - 1)

    # Use the scalar magnitude directly
    displacement_magnitudes = last_frame.particles['Displacement Magnitude']  # array of scalars
    thresholded_displacements = np.where(displacement_magnitudes > threshold, displacement_magnitudes, 0)
    # If Q requires squared displacement, square these:
    squared_displacements = thresholded_displacements ** 2
    total_squared_displacement_last = np.sum(squared_displacements)

    ionization, damage_energy = energy_lookup.get(int(energy), (0, energy))
    Q = total_squared_displacement_last / (6 * n0 * damage_energy)

    return [cnt_vacancies, cnt_interstitials, vac_cluster_dist, inter_cluster_dist, (times, vacancies, interstitials, temperatures, total_squared_displacement), Q, total_squared_displacement_last]

def _timeDependentAnalysis(self, pipeline, reference_pipeline, eng_hkl_dir):
        """
        Analyze defects, temperature, and total squared displacements as a function of time for a given trajectory.
        Args:
            pipeline: OVITO pipeline for the trajectory file.
            reference_pipeline: OVITO pipeline for the reference structure.
            eng_hkl_dir: Directory for the energy and hkl combination.
        Returns:
            times: List of simulation times from data.txt.
            vacancies: List of vacancy counts per frame.
            interstitials: List of interstitial counts per frame.
            temperatures: List of temperatures per frame.
            total_squared_displacement: List of total squared displacements per frame.
        """
        times = []
        vacancies = []
        interstitials = []
        temperatures = []
        total_squared_displacement = []
        data_file = os.path.join(eng_hkl_dir, 'data.txt')

        # Extract time and temperature from data.txt
        if os.path.exists(data_file):
            try:
                # Remove duplicates from data.txt
                data = np.unique(np.loadtxt(data_file, skiprows=1), axis=0)
                Step, Time, Epot, Ekin, Etot, Temp, max_ek, max_ek_border = data.T
                if Time[-1] < 30:
                    self.logger.warning(f"Simulation in {eng_hkl_dir} has final time {Time[-1]} ps < 30 ps. Skipping.")
                    return [], [], [], [], []
                times = list(Time)
                temperatures = list(Temp)
            except Exception as e:
                self.logger.warning(f"Failed to read data.txt from {eng_hkl_dir}: {e}. Skipping.")
                return [], [], [], [], []
        else:
            self.logger.warning(f"data.txt not found in {eng_hkl_dir}. Skipping.")
            return [], [], [], [], []

        # Defect analysis with Wigner-Seitz analysis
        wsam = WignerSeitzAnalysisModifier(per_type_occupancies=True, output_displaced=False)
        wsam.reference = reference_pipeline.source
        pipeline.modifiers.append(wsam)
        for frame in range(pipeline.source.num_frames):
            data = pipeline.compute(frame)
            cnt_vacancies = 0
            cnt_interstitials = 0
            for occupancy in data.particles['Occupancy']:
                if occupancy == 0:
                    cnt_vacancies += 1
                if occupancy > 1:
                    cnt_interstitials += 1
            vacancies.append(cnt_vacancies)
            interstitials.append(cnt_interstitials)
        pipeline.modifiers.remove(wsam)

        # Displacement analysis
        threshold=1.225
        modifier = CalculateDisplacementsModifier(
            reference_frame=0,
            minimum_image_convention=True  # Recommended for PBC handling
        )
        pipeline.modifiers.append(modifier)
        for frame in range(pipeline.source.num_frames):
            data = pipeline.compute(frame)
            displacement_magnitudes = data.particles['Displacement Magnitude']  # array of scalars
            thresholded_displacements = np.where(displacement_magnitudes > threshold, displacement_magnitudes, 0)
            if displacement_magnitudes is not None and len(displacement_magnitudes) > 0:
                squared_displacements = thresholded_displacements ** 2
                total_squared_disp = np.sum(squared_displacements)  # Sum over all atoms
                total_squared_displacement.append(total_squared_disp)
            else:
                total_squared_displacement.append(0.0)
        pipeline.modifiers.remove(modifier)

        # Align temperatures with defect frames
        if len(temperatures) != len(vacancies):
            try:
                frame_times = np.linspace(0, times[-1], len(vacancies))  # Interpolate to match frame count
                interp_temps = np.interp(frame_times, times, temperatures)
                temperatures = list(interp_temps)
                times = list(frame_times)
            except Exception as e:
                self.logger.warning(f"Failed to interpolate temperatures in {eng_hkl_dir}: {e}. Using initial temperature {self.temp} K.")
                temperatures = [self.temp] * len(vacancies)
                times = list(np.linspace(0, times[-1], len(vacancies)))

        if len(total_squared_displacement) != len(vacancies):
            try:
                interp_disp = np.interp(frame_times, times[:len(total_squared_displacement)], total_squared_displacement)
                total_squared_displacement = list(interp_disp)
            except Exception as e:
                self.logger.warning(f"Failed to interpolate total_squared_displacement in {eng_hkl_dir}: {e}. Using zero values.")
                total_squared_displacement = [0.0] * len(vacancies)

        return times, vacancies, interstitials, temperatures, total_squared_displacement

def postProcess(self):
        """
        Post-process LAMMPS simulation outputs to count vacancies, interstitials, their clusters,
        time-dependent data, and mixing ratio Q in parallel across hkl directions.
        Excludes simulations with final time < 30 ps and reads hkl_list.dat from energy folder.
        Aligns time-dependent data to a common 0-30 ps time axis.
        """
        eng_vac = {}             # {energy: list of number of vacancies}
        eng_inter = {}           # {energy: list of number of interstitials}
        eng_vac_cluster_dist = {}  # {energy: list of Counter objects for vacancy cluster sizes}
        eng_inter_cluster_dist = {}  # {energy: list of Counter objects for interstitial cluster sizes}
        eng_vac_time = {}        # {energy: list of (time, mean_vac, std_vac)}
        eng_inter_time = {}      # {energy: list of (time, mean_inter, std_inter)}
        eng_temp_time = {}       # {energy: list of (time, mean_temp, std_temp)}
        eng_total_disp_time = {} # {energy: list of (time, mean_total_disp_squared, std_total_disp_squared)}
        eng_mixing_ratio = {}    # {energy: list of Q values}
        eng_total_squared_disp = {}  # {energy: list of total squared displacements}

        for energy in self.energies:
            eng_dir = os.path.join(self.calculation_dir, str(int(energy)))
            self.logger.info(f'---------------------------------------------------')
            self.logger.info(f'Post-processing for energy: {energy} eV')

            # Load hkl_list from eng_dir/hkl_list.dat or count directories
            hkl_file = os.path.join(eng_dir, 'hkl_list.dat')
            if os.path.exists(hkl_file):
                self.hkl_list = np.loadtxt(hkl_file, skiprows=1).tolist()
                num_runs = len(self.hkl_list)
            else:
                self.hkl_list = []
                num_runs = len(glob.glob(os.path.join(eng_dir, '[0-9]*')))
                if num_runs == 0:
                    self.logger.warning(f"No hkl_list.dat or [0-9]* found in {eng_dir}. Skipping energy {energy} eV.")
                    continue

            # Parallel processing of hkl directions
            pool = mp.Pool(processes=min(mp.cpu_count(), num_runs))
            args = [(self, energy, idx, hkl, eng_dir) for idx, hkl in enumerate(self.hkl_list)]
            results = pool.starmap(process_hkl, args)
            pool.close()
            pool.join()

            # Aggregate results
            number_of_vacancies = []
            number_of_interstitials = []
            vac_cluster_counts = []
            inter_cluster_counts = []
            time_data = []
            Q_values = []
            total_squared_disp_last = []
            for result in results:
                if result is not None:
                    cnt_vacancies, cnt_interstitials, vac_cluster_dist, inter_cluster_dist, td, Q, total_squared_disp = result
                    number_of_vacancies.append(cnt_vacancies)
                    number_of_interstitials.append(cnt_interstitials)
                    vac_cluster_counts.append(vac_cluster_dist)
                    inter_cluster_counts.append(inter_cluster_dist)
                    time_data.append(td)
                    Q_values.append(Q)
                    total_squared_disp_last.append(total_squared_disp)

            eng_vac[energy] = number_of_vacancies
            eng_inter[energy] = number_of_interstitials
            eng_vac_cluster_dist[energy] = vac_cluster_counts
            eng_inter_cluster_dist[energy] = inter_cluster_counts
            eng_mixing_ratio[energy] = Q_values
            eng_total_squared_disp[energy] = total_squared_disp_last

            # Aggregate time-dependent data
            if time_data:
                # Use the maximum number of frames from valid simulations
                max_frames = max(len(t) for t, _, _, _, _ in time_data)
                # Create a common time axis from 0 to 30 ps
                common_times = np.linspace(0, 30, max_frames)
                vac_time = []
                inter_time = []
                temp_time = []
                total_disp_time = []

                for times, vacancies, interstitials, temperatures, total_squared_displacement in time_data:
                    # Interpolate to common time axis
                    try:
                        interp_vac = np.interp(common_times, times, vacancies, left=vacancies[0], right=vacancies[-1])
                        interp_inter = np.interp(common_times, times, interstitials, left=interstitials[0], right=interstitials[-1])
                        interp_temp = np.interp(common_times, times, temperatures, left=temperatures[0], right=temperatures[-1])
                        interp_disp = np.interp(common_times, times, total_squared_displacement, left=total_squared_displacement[0], right=total_squared_displacement[-1])
                    except Exception as e:
                        self.logger.warning(f"Failed to interpolate data for energy {energy} eV, hkl {hkl}: {e}. Skipping.")
                        continue
                    vac_time.append(interp_vac)
                    inter_time.append(interp_inter)
                    temp_time.append(interp_temp)
                    total_disp_time.append(interp_disp)

                # Convert to numpy arrays for mean and std
                vac_time = np.array(vac_time)
                inter_time = np.array(inter_time)
                temp_time = np.array(temp_time)
                total_disp_time = np.array(total_disp_time)

                # Compute mean and std
                eng_vac_time[energy] = [(t, np.mean(v), np.std(v)) for t, v in zip(common_times, vac_time.T)]
                eng_inter_time[energy] = [(t, np.mean(i), np.std(i)) for t, i in zip(common_times, inter_time.T)]
                eng_temp_time[energy] = [(t, np.mean(temp), np.std(temp)) for t, temp in zip(common_times, temp_time.T)]
                eng_total_disp_time[energy] = [(t, np.mean(disp), np.std(disp)) for t, disp in zip(common_times, total_disp_time.T)]

        # Write existing outputs
        def write_txt(vacancy_dict, interstitial_dict, file_name1, file_name2):
            for energy in vacancy_dict:
                output_dir = os.path.join(self.calculation_dir, f"{int(energy)}_postprocessing")
                os.makedirs(output_dir, exist_ok=True)
                num_vac = vacancy_dict.get(energy, [])
                num_int = interstitial_dict.get(energy, [])
                if num_vac:
                    mean_vac, std_vac = np.mean(num_vac), np.std(num_vac)
                    with open(os.path.join(output_dir, file_name1), 'w') as f:
                        f.write("# Energy (eV)    Mean value    Std Dev value\n")
                        f.write("# ---------------------------------------------\n")
                        f.write(f"{energy:>10.1f} {mean_vac:>15} {std_vac:>15}\n")
                if num_int:
                    mean_int, std_int = np.mean(num_int), np.std(num_int)
                    with open(os.path.join(output_dir, file_name2), 'w') as f:
                        f.write("# Energy (eV)    Mean value    Std Dev value\n")
                        f.write("# ----------------------------------------------------\n")
                        f.write(f"{energy:>10.1f} {mean_int:>15} {std_int:>15}\n")

        write_txt(eng_vac, eng_inter, 'eng_vac.txt', 'eng_inter.txt')

        # Write time-dependent outputs
        for energy in eng_vac_time:
            output_dir = os.path.join(self.calculation_dir, f"{int(energy)}_postprocessing")
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'eng_vac_time.txt'), 'w') as f:
                f.write("# Time (ps)    Mean Vacancies    Std Dev Vacancies\n")
                f.write("# ---------------------------------------------\n")
                for t, mean_v, std_v in eng_vac_time[energy]:
                    f.write(f"{t:>10.3f} {mean_v:>15} {std_v:>15}\n")
            with open(os.path.join(output_dir, 'eng_inter_time.txt'), 'w') as f:
                f.write("# Time (ps)    Mean Interstitials    Std Dev Interstitials\n")
                f.write("# ---------------------------------------------\n")
                for t, mean_i, std_i in eng_inter_time[energy]:
                    f.write(f"{t:>10.3f} {mean_i:>15} {std_i:>15}\n")
            with open(os.path.join(output_dir, 'eng_temp_time.txt'), 'w') as f:
                f.write("# Time (ps)    Mean Temperature    Std Dev Temperature\n")
                f.write("# ---------------------------------------------\n")
                for t, mean_t, std_t in eng_temp_time[energy]:
                    f.write(f"{t:>10.3f} {mean_t:>15} {std_t:>15}\n")
            with open(os.path.join(output_dir, 'eng_total_squared_displacement_time.txt'), 'w') as f:
                f.write("# Time (ps)    Mean Total Squared Displacement    Std Dev Total Squared Displacement\n")
                f.write("# ----------------------------------------------------------------\n")
                for t, mean_d, std_d in eng_total_disp_time[energy]:
                    f.write(f"{t:>10.3f} {mean_d:>30.3f} {std_d:>30.3f}\n")

        # Write cluster size distribution with mean and std dev
        for energy in eng_vac_cluster_dist:
            output_dir = os.path.join(self.calculation_dir, f"{int(energy)}_postprocessing")
            os.makedirs(output_dir, exist_ok=True)
            cluster_sizes = set()
            for cluster_count in eng_vac_cluster_dist[energy]:
                cluster_sizes.update(cluster_count.keys())
            cluster_stats = {}
            for size in cluster_sizes:
                counts = [counter.get(size, 0) for counter in eng_vac_cluster_dist[energy]]
                mean_count = np.mean(counts)
                std_dev_count = np.std(counts) if len(counts) > 1 else 0
                cluster_stats[size] = (mean_count, std_dev_count)
            if cluster_stats:
                with open(os.path.join(output_dir, 'eng_vac_cluster_dist.txt'), 'w') as f:
                    f.write("# Cluster_Size Mean_Count Std_Dev_Count\n")
                    f.write("# ---------------------------------------------\n")
                    for size in sorted(cluster_stats.keys()):
                        mean, std = cluster_stats[size]
                        f.write(f"{size:>12} {mean:>10.2f} {std:>12.2f}\n")

        for energy in eng_inter_cluster_dist:
            output_dir = os.path.join(self.calculation_dir, f"{int(energy)}_postprocessing")
            os.makedirs(output_dir, exist_ok=True)
            cluster_sizes = set()
            for cluster_count in eng_inter_cluster_dist[energy]:
                cluster_sizes.update(cluster_count.keys())
            cluster_stats = {}
            for size in cluster_sizes:
                counts = [counter.get(size, 0) for counter in eng_inter_cluster_dist[energy]]
                mean_count = np.mean(counts)
                std_dev_count = np.std(counts) if len(counts) > 1 else 0
                cluster_stats[size] = (mean_count, std_dev_count)
            if cluster_stats:
                with open(os.path.join(output_dir, 'eng_inter_cluster_dist.txt'), 'w') as f:
                    f.write("# Cluster_Size Mean_Count Std_Dev_Count\n")
                    f.write("# ---------------------------------------------\n")
                    for size in sorted(cluster_stats.keys()):
                        mean, std = cluster_stats[size]
                        f.write(f"{size:>12} {mean:>10.2f} {std:>12.2f}\n")

        # Write mixing ratio Q output
        for energy in eng_mixing_ratio:
            output_dir = os.path.join(self.calculation_dir, f"{int(energy)}_postprocessing")
            os.makedirs(output_dir, exist_ok=True)
            Q_values = eng_mixing_ratio[energy]
            total_squared_disp_values = eng_total_squared_disp[energy]
            if Q_values and total_squared_disp_values:
                mean_Q = np.mean(Q_values)
                std_Q = np.std(Q_values) if len(Q_values) > 1 else 0
                mean_total_squared_disp = np.mean(total_squared_disp_values)
                std_total_squared_disp = np.std(total_squared_disp_values) if len(total_squared_disp_values) > 1 else 0
                ionization, damage_energy = energy_lookup.get(int(energy), (0, energy))
                with open(os.path.join(output_dir, 'eng_mixing_ratio.txt'), 'w') as f:
                    f.write("# PKA energy (eV)    Ionization energy (eV)    Damage energy (eV)    Total Squared Displacement ± dev (Å²)    Q ± deviation (Å⁵/eV)\n")
                    f.write("# --------------------------------------------------------------------------------------------------------\n")
                    f.write(f"{energy:>15} {ionization:>20.1f} {damage_energy:>20.1f} {mean_total_squared_disp:>30.1f} ± {std_total_squared_disp:>10.1f} {mean_Q:>20.1f} ± {std_Q:>10.1f}\n")

def _countVacAndInter(self, pipeline, reference_pipeline):
        """
        Count vacancies and interstitials using Wigner-Seitz analysis.
        """
        wsam = WignerSeitzAnalysisModifier(per_type_occupancies=True, output_displaced=False)
        wsam.reference = reference_pipeline.source
        pipeline.modifiers.append(wsam)
        data = pipeline.compute(0)
        cnt_vacancies = 0
        cnt_interstitials = 0
        for occupancy in data.particles['Occupancy']:
            if occupancy == 0:
                cnt_vacancies += 1
            if occupancy > 1:
                cnt_interstitials += 1
        pipeline.modifiers.remove(wsam)
        return cnt_vacancies, cnt_interstitials

def _clustersAndDXA(self, pipeline, reference_pipeline, expression):
        """
        Perform cluster analysis for vacancies or interstitials.
        """
        wsam = WignerSeitzAnalysisModifier(per_type_occupancies=True, output_displaced=False)
        wsam.reference = reference_pipeline.source
        pipeline.modifiers.append(wsam)
        sel = ExpressionSelectionModifier(expression=expression)
        pipeline.modifiers.append(sel)
        cls = ClusterAnalysisModifier(cutoff=8.1, sort_by_size=True, only_selected=True)
        pipeline.modifiers.append(cls)
        data = pipeline.compute(0)
        cluster_table = data.tables['clusters']
        cluster_sizes = cluster_table['Cluster Size']
        pipeline.modifiers.remove(wsam)
        pipeline.modifiers.remove(sel)
        pipeline.modifiers.remove(cls)
        return Counter(cluster_sizes)
