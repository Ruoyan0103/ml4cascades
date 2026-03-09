import os
import numpy as np
from matplotlib import pyplot as plt
from ovito.io import import_file
from ovito.modifiers import ExpressionSelectionModifier
from ovito.pipeline import StaticSource, Pipeline
from ml4cascades.loggers.logger import AppLogger

module_dir = os.path.dirname(__file__)

class EPHprocessor: 
    def __init__(self, task_name='EPHprocessing'):
        self.log_dir = os.path.join(module_dir, 'logs', task_name)
        self.log_file = os.path.join(self.log_dir, 'EPH.log')
        self.logger = AppLogger(__name__, self.log_file, overwrite=True).get_logger()
    
    'Analyze the coupling for energy transfer efficiency'
    def analyze_coupling(self, 
                         dump_file: str,
                         frame_idx_list: list[int],
                         time_list: list[str],
                         figfile: str):
        figure, ax = plt.subplots(1, 1, figsize=(10, 6))
        all_pipeline = import_file(dump_file)
        for idx, frame_idx in enumerate(frame_idx_list):
            cur_pipeline = Pipeline(source=StaticSource(data=all_pipeline.compute(frame_idx)))
            expression = 'a 9'
            sel = ExpressionSelectionModifier(expression=expression)
            cur_pipeline.modifiers.append(sel)

            cur_data = cur_pipeline.compute()
            coupling_list = np.asarray(cur_data.particles['f_friction[2]'])
            print(len(coupling_list))
            x_values, y_counts = np.unique(coupling_list, return_counts=True)
            ax.plot(x_values, y_counts, linewidth=1.0, label=f'{time_list[idx]} ps')
        ax.set_xlabel('Coupling value')
        ax.set_ylabel('Occurrence count')
        ax.set_title('Coupling Value Frequency at Different Time Steps')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(figfile, dpi=300)









        
