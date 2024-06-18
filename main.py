import numpy as np
import solver
import functions
from functools import partial
import time
import data
import math
import matplotlib.pyplot as plt

parameters1 = {
    "rhoP" : [800,2000],
    "dP" : [1e-6],
    "phi" : [0],
    "V0" : [1e4],
    #"T" : np.linspace(1e-4, 10e-4, 100).tolist()
    "T" : sorted(np.logspace(-5, -3, num=127, base=10).tolist() + [0.00])
}

parameters2 = {
    "rhoP": 1000,
    "dP": 1e-5,
    "phi": 0,
    "V0": 1e4,
    "T": 1e-4
}


waveform = functions.one_plus_sin
flowfield = functions.linear
dt = 1e-6
n_steps = int(1e6)
# 40s for one particle running 1e6 steps

x0 = [0.01,0]
u0 = [0,0]

# partial function used since only parameters are iterated while the others are fixed values
simulate_single_partial = partial(solver.simulate_single,waveform = waveform, flowfield = flowfield, dt = dt, n_steps = n_steps, x0 = x0, u0 = u0)
simulate_single_phi_avg_partial = partial(solver.simulate_single_phase_averaged,waveform = waveform, flowfield = flowfield, dt = dt, n_steps = n_steps, x0 = x0, u0 = u0)

if __name__ == '__main__':

    print(solver.response_time(800,1e-7))

    # run solver with parameter, ff and wf
    start_time = time.time()
    #solver.sweep(parameters1,simulate_single_phi_avg_partial)
    duration = time.time() - start_time

    directory1 = 'raw/wf_one_plus_sin_ff_linear_dt_1e-06_nsteps_1000000_x0_[0.01, 0]_u0_[0, 0]'
    directory2 = 'raw/wf_one_plus_sin_ff_linear_dt_1e-06_nsteps_1000000_x0_[0.01, 0]_u0_[0, 0]_comp_10'

    #load_res = data.load_instance(directory,filename)
    #data.sinlge_plot(directory1,'particle_data_rhoP800_dP1e-06_V010000.0_T0.000644946677103762_phi0_wf_one_plus_sin_ff_linear.pkl')
    #data.plot_trajectory(directory1,'particle_data_rhoP800_dP1e-06_V010000.0_T0.000644946677103762_phi0_wf_one_plus_sin_ff_linear.pkl')
    #data.compare_final_x(directory1,'rhoP',[800,1200,1600,2000])
    #data.process_directory(directory1,n_interval=10)
    variable_dict = {
        'rhoP': [800],
        'dP': [1e-6]
    }
    data.compare_final_x_minus_dc(directory1,variable_dict,minus_DC=True,normalizeDC=True,comp = False)
    print('time',duration)
    #print(res)