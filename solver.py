import numpy as np
import math
from itertools import product
from types import SimpleNamespace
import data
import os
from multiprocessing import Pool
import time
import functools
import functions

mu = 1.73e-5 # viscosity of air
qP = 1.6e-19 # unit charge on a particle
r1 = 5e-3
r2 = 15e-3
# for spherical design
E0 = r1*r2/(r2-r1)


def response_time(rhoP,dP):
    return rhoP*(dP**2)/(18*mu)

def derivative(rhoP,dP,V0,T,phi,waveform,flowfield,t,state):

    # waveform should take in t T and V0 to return the current instant voltage
    # flowfield should take in x = [r,z] to return the local flow velocity v = [vr,vz]

    r, z, ur, uz = state
    x = np.array([r, z])
    u = np.array([ur, uz])
    v = flowfield(x)
    V = waveform(V0,T,t,phi)

    #E = np.array([V/(np.log(r2/r1))/0.01,0])
    # this is constant E field

    E = np.array([V/(np.log(r2/r1))/x[0],0])
    # this is the 1/r relationship when r is the radial location
    # the multiplying factor for r = r1 = 0.5cm, r2 = 1.5cm is 1.8/cm ( for V0 = 10kv, E max is 18kV/cm)
    # the multiplying factor for r = r1 = 10.5cm, r2 = 11.5cm is 1.05/cm

    #E = np.array([V / (np.log(r2 / r1)) / (x[0]**2 + x[1]**2)**(0.5), 0])
    # this is the 1/r relationship when r and z are essentially x and y
    # the multiplying factor is the same

    #E = np.array([V / (np.log(r2 / r1)) *(1/0.01 - x[0] / 1e-4), 0])
    # this is the linear relationship with gradient 1/r0^2 at r = r0 and


    #E = np.array([V*E0*x[0]/((x[0]**2+x[1]**2)**(3/2)), V*E0*x[1]/((x[0]**2+x[1]**2)**(3/2))]) # E*0.33 at r = 0.015
    # this is the spherical design. see derivation
    # the multiplying factor for r = r1 is 0.94/cm

    #if x[0] <= 0.01 :
    #    E = np.array([V / (np.log(r2 / r1)) / x[0], 0])
    #else:
    #    E = np.array([V * 0.01/r1 / (np.log(r2 / r1)) / x[0], 0])
    # this is the two layer system with a mid cylinder at r = 1cm. This allows for higher electric field strength
    # the multiplying factor is the same with 1/r relationship! this is done through the 0.01/r1 term that ensure the maximum local field strength is the same at r1 and 1cm.

    acceleration = (3*math.pi*mu*dP*(v-u) + qP*E)/(math.pi*rhoP*(dP**3)/6)
    return np.array([ur,uz,acceleration[0],acceleration[1]])


def rk4_step(rhoP, dP, V0, T,phi, waveform, flowfield, t, dt, state):
    k1 = derivative(rhoP, dP, V0, T,phi, waveform, flowfield, t, state)
    k2 = derivative(rhoP, dP, V0, T,phi, waveform, flowfield, t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = derivative(rhoP, dP, V0, T,phi, waveform, flowfield, t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = derivative(rhoP, dP, V0, T,phi, waveform, flowfield, t + dt, state + dt* k3)

    # Update state
    new_state = state + dt*(k1 + 2*k2 + 2*k3 + k4) / 6
    return new_state


def simulate_single(parameters, waveform, flowfield, dt, n_steps,x0,u0):
# parameters is a dictionary with the five corresponding variables
# rhoP, dP, phi, V0 and T
    par_val = SimpleNamespace(**parameters)
    print(parameters)
    x = x0
    u = u0
    trajectory = np.zeros((n_steps, 2))
    velocity = np.zeros((n_steps, 2))
    time_points = np.zeros(n_steps)
    t = 0
    state = np.array([x0[0], x0[1], u0[0], u0[1]])
    for i in range(n_steps):
        trajectory[i] = x
        velocity[i] = u
        time_points[i] = t
        state = rk4_step(par_val.rhoP,par_val.dP, par_val.V0, par_val.T,par_val.phi, waveform, flowfield, t, dt, state)
        x = state[:2]
        u = state[2:]
        t += dt

    instance = data.particle_data(par_val.rhoP, par_val.dP, par_val.phi, par_val.V0, par_val.T, trajectory, velocity, time_points, waveform, flowfield)
    return instance


def simulate_single_phase_averaged(parameters, waveform, flowfield, dt, n_steps, x0, u0, save=True):
    # This code will loop over computing each value for phi, neglecting the assigned phi value in parameters
    # If multiple values of phi are set in the parameter dict, multiple files of the same content will be generated

    start_time = time.time()

    sum_trajectory = np.zeros((n_steps, 2))
    sum_velocity = np.zeros((n_steps, 2))
    sum_time_points = np.zeros(n_steps)
    phis = (np.array([0,0.5,1,1.5]) * math.pi).tolist()

    par_val_avg = SimpleNamespace(**parameters)

    for j in range(len(phis)):
        parameters['phi'] = phis[j]
        par_val = SimpleNamespace(**parameters)

        x = x0
        u = u0
        trajectory = np.zeros((n_steps, 2))
        velocity = np.zeros((n_steps, 2))
        time_points = np.zeros(n_steps)
        t = 0
        state = np.array([x0[0], x0[1], u0[0], u0[1]])

        save_idx = 0
        for i in range(n_steps):
            trajectory[i] = x
            velocity[i] = u
            time_points[i] = t
            state = rk4_step(par_val.rhoP, par_val.dP, par_val.V0, par_val.T, par_val.phi, waveform, flowfield, t, dt,
                             state)
            x = state[:2]
            u = state[2:]
            t += dt

        sum_trajectory += trajectory
        sum_velocity += velocity
        sum_time_points += time_points

    duration = time.time() - start_time
    if save:

        instance = data.particle_data(par_val_avg.rhoP, par_val_avg.dP, par_val_avg.phi, par_val_avg.V0, par_val_avg.T,
                                      sum_trajectory / len(phis), sum_velocity / len(phis),
                                      sum_time_points / len(phis), waveform, flowfield)

        filename = (
            f'particle_data_rhoP{par_val_avg.rhoP}_dP{par_val_avg.dP}_V0{par_val_avg.V0}_T{par_val_avg.T}_phi{par_val_avg.phi}_'
            f'wf_{waveform.__name__}_ff_{flowfield.__name__}.pkl')

        directory = (f'raw/wf_{waveform.__name__}_ff_{flowfield.__name__}_dt_{dt}_nsteps_{n_steps}_'
                     f'x0_{x0}_u0_{u0}')

        if not os.path.exists(directory):
            os.makedirs(directory)

        data.save_instance(instance, directory, filename)
        print(f'saved_file: {filename}, duration is: {duration}')
    else:
        print(f'not_saving,duration is{duration}')
    return sum_trajectory / len(phis), sum_velocity / len(phis), sum_time_points / len(phis)

def simulate_single_combination_of_sin( sin_params,parameters, flowfield, dt, n_steps, x0, u0):
    # instead of using a given waveform, this function uses parameters for a combination of sine waves.
    # the two elements of sin_params are the relative period and relative amplitudes. see functions.generate_combinations_of_sin
    combinations_of_sin = functions.InterpolatingFunction(functions.generate_combinations_of_sin, functions.x_values, T_rel=sin_params[0], A=sin_params[1])
    trajectory,velocity,time_points = simulate_single_phase_averaged(parameters, combinations_of_sin, flowfield, dt, n_steps, x0, u0,save = False)
    return trajectory[-1,1]

def generate_combinations(parameters):
    keys, values = zip(*parameters.items())
    for combination in product(*values):
        yield dict(zip(keys, combination))


def sweep(parameters,partial_func):
    # generate_combinations should be used with list() to return a list of dictionaries, each containing a combination
    par_comb = list(generate_combinations(parameters))
    with Pool(processes=128) as pool:
        results = list(pool.map(partial_func, par_comb))

    return results
# the sweep function returns the result in the format of :
# list[[result of first set of parameters],[result of second set of parameters],...]

def scan_combinations_of_sin(parameters, flowfield, dt, n_steps, x0, u0):
    partial_simulate_single_combination_of_sin = functools.partial(simulate_single_combination_of_sin,parameters=parameters, flowfield = flowfield, dt = dt, n_steps = n_steps, x0 = x0, u0 = u0)
    # each of the _comb are list of sets of A and T_rel values.
    sin_comb,base = generate_sin_combinations()
    # A_comb must start with [0,x,y....] to compute the DC component
    with Pool(processes=128) as pool:
        results = list(pool.map(partial_simulate_single_combination_of_sin, sin_comb))
    separation = np.array(results) - results[0]
    data.plot_3d(base,separation)
    for i in range(0,len(base)):
        print(f'param:{base[i]},separation:{separation[i]}')
    # final r location


def generate_sin_combinations():
    result = [[[0,0],[0,0]]]
    # Trel and Arel
    # [0,0],[0,0] refers to DC
    base = [[0,0]]
    for i in range(1,10):
        for j in range(0,10):
            result.append([[1,1/i],[1,j/30]])
            