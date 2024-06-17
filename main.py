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
    "T" : np.logspace(-6, -3, num=128, base=10).tolist()
}

parameters2 = {
    "rhoP": 1000,
    "dP": 1e-5,
    "phi": 0,
    "V0": 1e4,
    "T": 1e-4
}


waveform = functions.abs_sine
flowfield = functions.linear
dt = 1e-8
n_steps = int(1e7)
# 40s for one particle running 1e6 steps

x0 = [0.01,0]
u0 = [0,0]

# partial function used since only parameters are iterated while the others are fixed values
simulate_single_partial = partial(solver.simulate_single,waveform = waveform, flowfield = flowfield, dt = dt, n_steps = n_steps, x0 = x0, u0 = u0)
simulate_single_phi_avg_partial = partial(solver.simulate_single_phase_averaged,waveform = waveform, flowfield = flowfield, dt = dt, n_steps = n_steps, x0 = x0, u0 = u0)

if __name__ == '__main__':

    directory1 = 'raw/wf_one_plus_sin_ff_const_dt_1e-06_nsteps_1000000_x0_[0.01, 0]_u0_[0, 0]'
    directory2 = 'raw/wf_one_plus_sin_ff_linear_dt_1e-06_nsteps_1000000_x0_[0.01, 0]_u0_[0, 0]'
    directory3 = 'raw/wf_one_plus_sin_ff_linear_dt_1e-06_nsteps_10000000_x0_[0.01, 0]_u0_[0, 0]'
    directory4 = 'raw/wf_half_sawtooth_ff_linear_dt_1e-06_nsteps_1000000_x0_[0.01, 0]_u0_[0, 0]'
    directory5 = 'raw/wf_one_plus_sin_ff_linear_dt_1e-08_nsteps_10000000_x0_[0.01, 0]_u0_[0, 0]'
    directory6 = 'raw/wf_one_plus_sin_ff_linear_dt_1e-08_nsteps_1000000_x0_[0.01, 0]_u0_[0, 0]'
    directory7 = 'raw/wf_one_plus_sin_ff_linear_dt_1e-08_nsteps_1000000_x0_[0.01, 0]_u0_[0, 0]'
    directory8 = 'raw/wf_abs_sine_ff_linear_dt_1e-08_nsteps_1000000_x0_[0.01, 0]_u0_[0, 0]'
    directory9 = 'raw/wf_abs_sine_ff_linear_dt_1e-08_nsteps_10000000_x0_[0.01, 0]_u0_[0, 0]'
    directory10 = 'raw/wf_abs_sine_ff_linear_dt_1e-08_nsteps_10000000_x0_[0.01, 0]_u0_[0, 0]_comp_10'
    filename = 'particle_data_rhoP1000_dP1e-05_V010000.0_T0.00018420699693267163_phi0_wf_one_plus_sin_ff_linear.pkl'

    print(solver.response_time(800,1e-7))

    # run solver with parameter, ff and wf
    start_time = time.time()
    #data.process_directory(directory9,n_interval=10)
    #solver.sweep(parameters1,simulate_single_phi_avg_partial)
    duration = time.time() - start_time

       #load_res = data.load_instance(directory,filename)
    #data.sinlge_plot(directory,filename)
    #data.plot_trajectory(directory,filename)
    #data.compare_final_x(directory1,'rhoP',[800,1200,1600,2000])

    variable_dict = {
        'rhoP': [1200],
        'dP': [1e-6]
    }
    data.compare_final_x_minus_dc(directory10,variable_dict,minus_DC=False,normalizeDC=False)


    print('time',duration)
    #print(res)