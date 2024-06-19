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
    "dP" : [1e-6,5e-7,3e-7],
    "phi" : [0],
    "V0" : [1e3],
    #"T" : np.linspace(1e-4, 10e-4, 100).tolist()
    "T" : sorted(np.logspace(-6, -3, num=127, base=10).tolist() + [0.00])
}

parameters2 = {
    "rhoP": 1000,
    "dP": 1e-5,
    "phi": 0,
    "V0": 1e4,
    "T": 1e-4
}


waveform = functions.N_plus_sin
flowfield = functions.linear
dt = 1e-7
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


    directory3 = 'raw/wf_one_plus_sin_ff_linear_dt_1e-07_nsteps_1000000_x0_[0.01, 0]_u0_[0, 0]'
    directory4 = 'raw/wf__075_plus_sin_ff_linear_dt_1e-07_nsteps_1000000_x0_[0.01, 0]_u0_[0, 0]'
    directory5 = 'raw/wf_half_sawtooth_ff_linear_dt_1e-07_nsteps_1000000_x0_[0.01, 0]_u0_[0, 0]'
    directory6 = 'raw/wf__05_plus_sin_ff_linear_dt_1e-07_nsteps_1000000_x0_[0.01, 0]_u0_[0, 0]'
    directory7 = 'raw/wf_N_plus_sin_ff_linear_dt_1e-07_nsteps_1000000_x0_[0.01, 0]_u0_[0, 0]'
    #load_res = data.load_instance(directory,filename)
    #data.sinlge_plot(directory1,'particle_data_rhoP800_dP1e-06_V010000.0_T0.000644946677103762_phi0_wf_one_plus_sin_ff_linear.pkl')
    #data.plot_trajectory(directory1,'particle_data_rhoP800_dP1e-06_V010000.0_T0.000644946677103762_phi0_wf_one_plus_sin_ff_linear.pkl')
    #data.process_directory(directory3,n_interval=10)
    #compare_final_x_minus_dc doesn't currently work with compressed data. NEED TO BE FIXED

    variable_dict = {
        'rhoP': [800,2000],
        'dP': [1e-6],
        'V0': [1e4]
    }
    #data.separation_over_time(directory3,variable_dict,T_plot_list=[2e-4],minus_DC=True,normalizeDC=True,comp = False)
    #data.separation_rate_over_time(directory3,variable_dict,T_plot_list=[2e-4],minus_DC=True,normalizeDC=True,comp = False)
    data.energy_over_time(directory3,variable_dict,T_plot_list=[2e-4],minus_DC=True,normalizeDC=True,comp = False)
    #data.compare_final_x_minus_dc(directory3,variable_dict,minus_DC=True,normalizeDC=True,comp = False)
    #data.compare_final_energy(directory7,variable_dict,minus_DC=True,normalizeDC=True,comp = False)
    print('time',duration)
    #print(res)