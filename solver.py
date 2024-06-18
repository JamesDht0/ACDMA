import numpy as np
import math
from itertools import product
from types import SimpleNamespace
import data
import os
from multiprocessing import Pool

mu = 1.73e-5 # viscosity of air
qP = 1.6e-19 # unit charge on a particle
r1 = 5e-3
r2 = 15e-3


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
    # this is the 1/r relationship
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


def simulate_single_phase_averaged(parameters, waveform, flowfield, dt, n_steps, x0, u0):
    # This code will loop over computing each value for phi, neglecting the assigned phi value in parameters
    # If multiple values of phi are set in the parameter dict, multiple files of the same content will be generated

    sum_trajectory = np.zeros((n_steps, 2))
    sum_velocity = np.zeros((n_steps, 2))
    sum_time_points = np.zeros(n_steps)
    phis = (np.array([0, 1 / 4, 1 / 2, 3 / 4, 1, 5 / 4, 6 / 4, 7 / 4]) * math.pi).tolist()

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

    return sum_trajectory / len(phis), sum_velocity / len(phis), sum_time_points / len(phis)


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

