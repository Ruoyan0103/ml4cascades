import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from ovito.io import import_file, export_file
from ml4cascades.loggers.logger import AppLogger

TICK_FONTSIZE = 14
LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 13

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]

module_dir = os.path.dirname(__file__)

class LammpsCascadeVisualizer:
    def __init__(self, model_name: str, task_name: str):
        self.log_dir = os.path.join(module_dir, 'logs', task_name)
        self.log_file = os.path.join(self.log_dir, f'{model_name}.log')
        self.logger = AppLogger(__name__, self.log_file, overwrite=True).get_logger()
        os.makedirs(self.log_dir, exist_ok=True)

    def exportDampfile(self, traj_folder: str, export_step: int):
        traj_file = os.path.join(traj_folder, "data.output")
        all_pipeline = import_file(traj_file)
        number_of_frames = all_pipeline.source.num_frames // export_step
        self.logger.info(f'Export {number_of_frames} frames with export step {export_step}\n')
        output_file = os.path.join(traj_folder, f"output.xyz")
        export_file(all_pipeline, output_file, "xyz", multiple_frames=True, every_nth_frame=export_step,
                    columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", "c_ep", "c_ek", "c_myflag_val"])
    
