import numpy as np
import solver
import functions
from functools import partial
import time
import data
import math
import matplotlib.pyplot as plt

parameters1 = {
    "rhoP" : [1000,2000],
    "dP" : [1e-6],
    "phi" : [0],
    "V0" : [100],
    "T" : np.logspace(-5, -3, 31,base = 10).tolist()+[0.0]
    #"T" : [0.0,2e-4]
}

parameters2 = {
    "rhoP": 2000,
    "dP": 1e-6,
    "phi": 0,
    "V0": 1e2,
    "T": 2e-4
}


flowfield = functions.linear
waveform = functions.combinations_of_sin
dt = 1e-6
n_steps = int(1e6)
# 5min for one particle running 1e6 steps with 8 phases

x0 = [0.015,0]
u0 = [0,0]

# partial function used since only parameters are iterated while the others are fixed values
simulate_single_partial = partial(solver.simulate_single,waveform = waveform, flowfield = flowfield, dt = dt, n_steps = n_steps, x0 = x0, u0 = u0)
simulate_single_phi_avg_partial = partial(solver.simulate_single_phase_averaged,waveform = waveform, flowfield = flowfield, dt = dt, n_steps = n_steps, x0 = x0, u0 = u0)

if __name__ == '__main__':

    print(f'response time is {solver.response_time(1000,1e-7)}')

    # run solver with parameter, ff and wf
    start_time = time.time()
    solver.sweep(parameters1, simulate_single_phi_avg_partial)
    duration = time.time() - start_time

    directory1 = 'raw/wf_N_plus_sin_ff_linear_dt_1e-06_nsteps_1000000_x0_[0.015, 0]_u0_[0, 0]_base'
    directory2 = 'raw/wf_N_plus_sin_ff_const_dt_1e-06_nsteps_1000000_x0_[0.015, 0]_u0_[0, 0]_sph'
    directory3 = 'raw/wf_N_plus_sin_ff_const_dt_1e-06_nsteps_1000000_x0_[0.015, 0]_u0_[0, 0]_twostage'
    directory4 = 'raw/wf_N_plus_sin_ff_linear_dt_1e-06_nsteps_10000000_x0_[0.015, 0]_u0_[0, 0]_base'
    directory5 = 'raw/wf_N_plus_sin_ff_const_dt_1e-06_nsteps_10000000_x0_[0.015, 0]_u0_[0, 0]_sph'
    directory6 = 'raw/wf_N_plus_sin_ff_linear_dt_1e-06_nsteps_10000000_x0_[0.015, 0]_u0_[0, 0]_twostage'

    #load_res = data.load_instance(directory,filename)
    #data.sinlge_plot(directory1,'particle_data_rhoP800_dP1e-06_V010000.0_T0.000644946677103762_phi0_wf_one_plus_sin_ff_linear.pkl')
    #data.plot_trajectory(directory1,'particle_data_rhoP800_dP1e-06_V010000.0_T0.000644946677103762_phi0_wf_one_plus_sin_ff_linear.pkl')
    #data.process_directory(directory3,n_interval=10)
    #compare_final_x_minus_dc doesn't currently work with compressed data. NEED TO BE FIXED

    directory00 = 'raw/wf_N_plus_sin_ff_linear_dt_1e-06_nsteps_1000000_x0_[0.015, 0]_u0_[0, 0]_base'
    directory01 = 'raw/wf_combinations_of_sin_ff_linear_dt_1e-06_nsteps_1000000_x0_[0.015, 0]_u0_[0, 0]'

    variable_dict1 = {
        'rhoP': [1000],
        #'dP':sorted(np.linspace(0.5e-6,1e-6,5).tolist()),
        'dP': [0.5e-6],
        'V0': [100,200]
    }
    variable_dict2 = {
        'rhoP': [1000, 2000],
        'dP': [0.5e-6,1e-6],
        'V0': [100,200]
    }
    data.compare_final_x_minus_dc(directory00, variable_dict2, minus_DC=True, normalizeDC=False, comp=False, N=int(1))
    data.compare_final_x_minus_dc(directory01, variable_dict2, minus_DC=True, normalizeDC=False, comp=False, N=int(1))
    data.compare_final_x_minus_dc(directory52, variable_dict2, minus_DC=True, normalizeDC=False, comp=False, N=int(1))
    data.compare_final_x_minus_dc(directory72, variable_dict2, minus_DC=True, normalizeDC=False, comp=False, N=int(1))

    data.separation_over_time(directory32, variable_dict2, T_plot_list=[2e-4], minus_DC=False, normalizeDC=False,
                              comp=False, fitting=False)
    data.separation_over_time(directory52, variable_dict2, T_plot_list=[2e-4], minus_DC=False, normalizeDC=False,
                              comp=False, fitting=False)
    data.separation_over_time(directory72, variable_dict2, T_plot_list=[2e-4], minus_DC=False, normalizeDC=False,
                              comp=False, fitting=False)



    #data.spherical_compare_final_x_minus_dc(directory18, variable_dict1, minus_DC=True, normalizeDC=False, comp=False, N=int(1))
    #data.plot_multiple_trajectory(directory13,variable_dict1,T_plot_list=[2e-4,0.0],ignore_final = int(100))
    #data.plot_multiple_trajectory(directory14, variable_dict1, T_plot_list=[2e-4,0.0],ignore_final = int(100))
    #data.separation_over_time(directory16,variable_dict1,T_plot_list=[2e-4],minus_DC=False,normalizeDC=False,comp = False,fitting=False)
    #data.spherical_separation_over_time(directory13, variable_dict1, T_plot_list=[2e-4,0.0], minus_DC=True, normalizeDC=False,
    #                          comp=False, fitting=False,ignore_final = int(100))
    #data.spherical_separation_over_time(directory14, variable_dict1, T_plot_list=[2e-4, 0.0], minus_DC=True,
    #                                   normalizeDC=False,
    #                                    comp=False, fitting=False, ignore_final=int(1))

    #data.separation_over_time(directory1, variable_dict1, T_plot_list=[2e-4], minus_DC=False, normalizeDC=False,comp=False, fitting=False)

    #data.separation_rate_over_time(directory8,variable_dict,T_plot_list=[2e-4],minus_DC=True,normalizeDC=True,comp = False)
    #data.energy_over_time(directory8,variable_dict,T_plot_list=[2e-4],minus_DC=True,normalizeDC=True,comp = False)
    #data.compare_final_x_minus_dc(directory2, variable_dict2, minus_DC=True, normalizeDC=False, comp=False,
    #                             N=int(1))  # N is the number of final points to average over
        #data.compare_final_x_minus_dc(directory16, variable_dict1, minus_DC=True, normalizeDC=False, comp=False, N=int(1))
    #.compare_final_x_minus_dc(directory16, variable_dict1, minus_DC=True, normalizeDC=False, comp=False, N=int(1))

    #data.compare_final_energy(directory8,variable_dict,minus_DC=True,normalizeDC=True,comp = False)
    print('time',duration)
    #print(res)