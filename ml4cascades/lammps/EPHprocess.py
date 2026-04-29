import os
import numpy as np
from matplotlib import axes, pyplot as plt
from ovito.io import import_file
from ovito.pipeline import StaticSource, Pipeline
from ovito.modifiers import ExpressionSelectionModifier
from ml4cascades.loggers.logger import AppLogger

module_dir = os.path.dirname(__file__)

class EPHprocessor: 
    def __init__(self, task_name='EPHprocessing'):
        self.log_dir = os.path.join(module_dir, 'logs', task_name)
        self.log_file = os.path.join(self.log_dir, 'EPH.log')
        self.logger = AppLogger(__name__, self.log_file, overwrite=True).get_logger()

    def analyze_coupling(self, 
                     folder_list: list[str],
                     label_list: list[str],
                     expression: str,
                     frame_idx_list: list[list[int]],  # different folder may have different frame indices
                     time_list: list[float]):
        fig, ax = plt.subplots(figsize=(6, 4))
        fig_file = f'{folder_list[0]}/coupling.png'

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2_file = f'{folder_list[0]}/density_hist.png'

        line_styles = ['-', '--', '-.', ':']    
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for frame_idx_id, frame_idx in enumerate(frame_idx_list):
            for j, folder in enumerate(folder_list):
                dump_file = f'{folder}/data.output'
                density_file = f'{folder}/density.out'
                coupling_file = f'{folder}/coupling.out'
                ek_file = f'{folder}/ek.out'

                all_pipeline = import_file(dump_file)

                density_columns = []
                coupling_columns = []
                ek_columns = []

                cur_pipeline = Pipeline(source=StaticSource(data=all_pipeline.compute(frame_idx[j])))

                if expression is not None:
                    sel = ExpressionSelectionModifier(expression=expression)
                    cur_pipeline.modifiers.append(sel)
                    cur_data = cur_pipeline.compute()

                    selection = np.asarray(cur_data.particles['Selection']) > 0
                    cur_density = np.asarray(cur_data.particles['f_friction[1]'])[selection]
                    cur_coupling = np.asarray(cur_data.particles['f_friction[2]'])[selection]
                    cur_ek = np.asarray(cur_data.particles['c_ek'])[selection]
                else:
                    cur_data = cur_pipeline.compute()
                    cur_density = np.asarray(cur_data.particles['f_friction[1]'])
                    cur_coupling = np.asarray(cur_data.particles['f_friction[2]'])
                    cur_ek = np.asarray(cur_data.particles['c_ek'])

                density_columns.append(cur_density)
                coupling_columns.append(cur_coupling)
                ek_columns.append(cur_ek)

                max_density_len = max(len(col) for col in density_columns)
                max_coupling_len = max(len(col) for col in coupling_columns)
                max_ek_len = max(len(col) for col in ek_columns)

                density_output = np.full((max_density_len, len(density_columns)), np.nan)
                coupling_output = np.full((max_coupling_len, len(coupling_columns)), np.nan)
                ek_output = np.full((max_ek_len, len(ek_columns)), np.nan)

                for i, col in enumerate(density_columns):
                    density_output[:len(col), i] = col
                for i, col in enumerate(coupling_columns):
                    coupling_output[:len(col), i] = col
                for i, col in enumerate(ek_columns):
                    ek_output[:len(col), i] = col

                density_header = "  ".join([f"f_{frame_idx[j]}"])
                coupling_header = "  ".join([f"f_{frame_idx[j]}"])
                ek_header = "  ".join([f"f_{frame_idx[j]}"])

                np.savetxt(density_file, density_output, fmt="%.10g", header=density_header, comments="")
                np.savetxt(coupling_file, coupling_output, fmt="%.10g", header=coupling_header, comments="")
                np.savetxt(ek_file, ek_output, fmt="%.10g", header=ek_header, comments="")

                c_vals = coupling_output[:, 0]
                c_vals = c_vals[~np.isnan(c_vals)]

                d_vals = density_output[:, 0]
                d_vals = d_vals[~np.isnan(d_vals)]

                ek_vals = ek_output[:, 0]
                ek_vals = ek_vals[~np.isnan(ek_vals)]

                # convert to velocity unit of Å/ps
                vel_vals = np.sqrt(2 * ek_vals * 1.60218e-19 / 72.64 * 1e20)

                # Original scatter figure
                ax.scatter(
                    ek_vals,
                    d_vals,
                    label=f"{label_list[j]} {time_list[frame_idx_id]:.0f} fs",
                    s=1,
                    alpha=0.5
                )

                # Second figure: density histogram as a curve
                d_hist_vals = d_vals[d_vals > 0]
                if len(d_hist_vals) > 1:
                    bins = np.logspace(np.log10(d_hist_vals.min()), np.log10(d_hist_vals.max()), 80)
                    hist, edges = np.histogram(d_hist_vals, bins=bins)
                    centers = 0.5 * (edges[:-1] + edges[1:])

                    ax2.plot(
                        centers,
                        hist,
                        label=f"{label_list[j]} {time_list[frame_idx_id]:.0f} fs",
                        linewidth=1,
                        linestyle = line_styles[j % len(line_styles)],
                        color = colors[frame_idx_id % len(colors)]
                    )
                    mean_density = np.mean(d_hist_vals)
                    ax2.axvline(
                        mean_density,
                        linestyle=line_styles[j % len(line_styles)],
                        linewidth=1,
                        color=colors[frame_idx_id % len(colors)],
                        alpha=0.8
                    )

        ax.legend(loc='upper right')
        ax.set_xlabel("Kinetic energy (eV)")
        ax.set_ylabel("Density (e/Å³)")
        # ax.set_ylim(0, 0.25)
        ax.set_xscale('log')
        ax.set_yscale('log')

        fig.tight_layout()
        fig.savefig(fig_file, dpi=300)
        plt.close(fig)

        ax2.legend(loc='upper right')
        ax2.set_xlabel("Density (e/Å³)")
        ax2.set_ylabel("Number of atoms")
        ax2.set_xlim(0, 0.5)
        # ax2.set_yscale('log')

        fig2.tight_layout()
        fig2.savefig(fig2_file, dpi=300)
        plt.close(fig2)





        
