from ase.io import read 
from matplotlib import pyplot as plt

# ion beam mixing 
class CascadeProcessor:
    def __init__(self,
                 traj_file: str):
        self.traj_file = traj_file
        
    def cal_ibm(self, 
                n0: float,
                ed: float,
                r2_fig: str,
                q_fig: str):
        traj_frames = read(self.traj_file, format='extxyz', index=":")
        Q = []
        R_2 = []
        timestep = []
        init_pos = traj_frames[0].positions
        for frame in traj_frames[1:]:
            R_2_val = 0
            curr_pos = frame.positions
            for i in range(len(curr_pos)):
                R_2_val += (curr_pos[i] - init_pos[i]) ** 2
            R_2.append(R_2_val)
            Q.append(R_2_val/(6*n0*ed))
        plt.plot(timestep, R_2)
        plt.xlabel('Timestep')
        plt.ylabel('R_2')
        plt.savefig(r2_fig)
        plt.close()

