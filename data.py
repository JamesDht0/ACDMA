import pickle
import os
import matplotlib.pyplot as plt
from tkinter import Tk, StringVar, OptionMenu, Button, Label
import re
import numpy as np
import functions
import solver
from itertools import product
class particle_data:
    def __init__(self,rhoP,dP,phi,V0,T,trajectory,velocity,time_points,waveform,flowfield):
        self.rhoP = rhoP
        self.dP = dP
        self.phi = phi
        self.V0 = V0
        self.T = T
        self.trajectory = trajectory
        self.velocity = velocity
        self.time_points = time_points
        self.waveform = waveform
        self.flowfield = flowfield

    def __repr__(self):
        return (f"particle_data(rhoP={self.rhoP}, dP={self.dP}, phi={self.phi}, V0={self.V0}, T={self.T}, "
                f"trajectory={self.trajectory}, velocity={self.velocity}, time_points={self.time_points}, "
                f"waveform={self.waveform.__name__}, flowfield={self.flowfield.__name__})")
def save_instance(instance,directory,filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as file:
        pickle.dump(instance, file)
def load_instance(directory, filename):
    filepath = os.path.join(directory, filename)
    with open(filepath, 'rb') as file:
        return pickle.load(file)
def list_files(directory):
    # List all items in the directory
    items = os.listdir(directory)
    # Filter out directories, keeping only files
    files = [item for item in items if os.path.isfile(os.path.join(directory, item))]
    return files
def plot_arrays(array_x, array_y, x_label, y_label):
    plt.figure()
    plt.plot(array_x)
    plt.plot(array_y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
def sinlge_plot(directory,filename):
    # plot the trajectory, r-t,z-t plots and velocity plots.
    dat = load_instance(directory,filename)

    arrays = {

        'r_trajectory' : dat.trajectory[:,0],
        'z_trajectory' : dat.trajectory[:,1],
        'r_velocity' : dat.velocity[:0],
        'z_velocity' : dat.velocity[:1],
        'time_points' : dat.time_points
    }

    rhoP = dat.rhoP
    dP = dat.dP
    phi = dat.phi

    create_gui(arrays)
def create_gui(arrays):
    def update_plot(*args):
        selected_x = x_var.get()
        selected_y = y_var.get()
        array_x = arrays[selected_x]
        array_y = arrays[selected_y]
        plot_arrays(array_x, array_y, selected_x, selected_y)

    root = Tk()
    root.title("Select Arrays to Plot")

    x_label = Label(root, text="Select X array:")
    x_label.pack()

    x_var = StringVar(root)
    x_var.set('Array 1')  # default value
    x_menu = OptionMenu(root, x_var, *arrays.keys())
    x_menu.pack()

    y_label = Label(root, text="Select Y array:")
    y_label.pack()

    y_var = StringVar(root)
    y_var.set('Array 2')  # default value
    y_menu = OptionMenu(root, y_var, *arrays.keys())
    y_menu.pack()

    plot_button = Button(root, text="Plot", command=update_plot)
    plot_button.pack()

    root.mainloop()
def load_pkl(directory):
    pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    data = []
    for pkl_file in pkl_files:
        filepath = os.path.join(directory, pkl_file)
        with open(filepath, 'rb') as file:
            data.append(pickle.load(file))
    return data
def final_x_T(directory):
    data = load_pkl(directory)

    final_r = []
    final_z = []
    T = []

    for i in range (0,len(data)):
        final_r.append(data[i].trajectory[-1,0])
        final_z.append(data[i].trajectory[-1,1])
        T.append(data[i].T)

    order  = np.argsort(T)
    final_r_sorted = np.array(final_r)[order]
    final_z_sorted = np.array(final_z)[order]
    T_sorted = np.array(T)[order]

    print(len(T_sorted))

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(T_sorted[10:], final_r_sorted[10:], label=f'T vs final_r')
    axs[0].set_xlabel('T')
    axs[0].set_ylabel('r')
    axs[0].legend()
    axs[0].autoscale()
    axs[1].plot(T_sorted[10:], final_z_sorted[10:], label=f'T vs final_z')
    axs[1].set_xlabel('T')
    axs[1].set_ylabel('z')
    axs[1].legend()
    axs[1].autoscale()

    plt.tight_layout()
    plt.show()
def plot_trajectory(directory,filename):
    data = load_instance(directory,filename)
    print(data.time_points)
    print(len(data.trajectory[:,1]))
    plt.figure()
    plt.plot(data.trajectory[:,0],data.time_points)
    plt.show()
def extract_parameters(filename, foldername, compressed = True):
    # Define the regular expression pattern based on the filename format

    if compressed:
        file_pattern = re.compile(
            r"particle_data_rhoP(?P<rhoP>[\d\.e+-]+)_dP(?P<dP>[\d\.e+-]+)_V0(?P<V0>[\d\.e+-]+)_T(?P<T>[\d\.e+-]+)_phi(?P<phi>[\d\.e+-]+)_wf_(?P<waveform>\w+)_ff_(?P<flowfield>\w+)_comp_[\d]+\.pkl"
        )
        folder_pattern = re.compile(
            r"raw/wf_(?P<waveform>\w+)_ff_(?P<flowfield>\w+)_dt_(?P<dt>[\d\.e+-]+)_nsteps_(?P<nsteps>\d+)_x0_\[(?P<x0>[\d\., ]+)\]_u0_\[(?P<u0>[\d\., ]+)\]_comp_[\d]+"
        )
    else:
        file_pattern = re.compile(
            r"particle_data_rhoP(?P<rhoP>[\d\.e+-]+)_dP(?P<dP>[\d\.e+-]+)_V0(?P<V0>[\d\.e+-]+)_T(?P<T>[\d\.e+-]+)_phi(?P<phi>[\d\.e+-]+)_wf_(?P<waveform>\w+)_ff_(?P<flowfield>\w+)\.pkl"
        )
        # Define the regular expression pattern for the folder name
        folder_pattern = re.compile(
            r"raw/wf_(?P<waveform>\w+)_ff_(?P<flowfield>\w+)_dt_(?P<dt>[\d\.e+-]+)_nsteps_(?P<nsteps>\d+)_x0_\[(?P<x0>[\d\., ]+)\]_u0_\[(?P<u0>[\d\., ]+)\]"
        )

    # Try matching the first format
    file_match = file_pattern.match(filename)
    folder_match = folder_pattern.match(foldername)

    print('file',file_match)
    print('folder',folder_match)

    if file_match and folder_match:
        # Extract parameters from the file as a dictionary
        parameters = file_match.groupdict()

        # Convert numeric values to floats
        for key in ['rhoP', 'dP', 'V0', 'T', 'phi']:
            parameters[key] = float(parameters[key])

        # Extract waveform and flowfield from the filename
        waveform_name = parameters.pop('waveform')
        flowfield_name = parameters.pop('flowfield')

        # Convert waveform and flowfield names to function objects
        waveform = functions.waveform_functions.get(waveform_name)
        flowfield = functions.flowfield_functions.get(flowfield_name)

        if waveform is None or flowfield is None:
            raise ValueError(f"Unknown waveform or flowfield function: {waveform_name}, {flowfield_name}")

        # Extract additional parameters from the folder
        additional_params = folder_match.groupdict()
        dt = float(additional_params['dt'])
        nsteps = int(additional_params['nsteps'])
        x0 = [float(val) for val in additional_params['x0'].split(',')]
        u0 = [float(val) for val in additional_params['u0'].split(',')]

        return parameters, dt, nsteps, x0, u0, waveform, flowfield
    else:
        raise ValueError("Filename or folder name does not match the expected pattern")
def compare_final_x_minus_dc(directory, variable_dict, minus_DC = True, normalizeDC = True,compressed = True):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    keys, values = zip(*variable_dict.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    for combination in combinations:
        label = ", ".join([f"{key}={value}" for key, value in combination.items()])

        matching_files = filter_filenames_by_variables(directory, combination)
        print(f"Combination: {combination}")
        print(f"Matching files: {matching_files}")

        if not matching_files:
            print(f"No matching files for combination: {combination}")
            continue

        data = []
        for pkl_file in matching_files:
            filepath = os.path.join(directory, pkl_file)
            with open(filepath, 'rb') as file:
                data.append(pickle.load(file))

        DC_params, dt, n_step, x0, u0, waveform, flowfield = extract_parameters(matching_files[0], directory,compressed=compressed)

        DC_params['waveform'] = 'DC'


        final_r = []
        final_z = []
        T = []

        if minus_DC:
            DCdata = solver.simulate_single(DC_params, functions.DC, flowfield, dt, n_step, x0, u0)
            DC_final_r = DCdata.trajectory[-1, 0]
            DC_final_z = DCdata.trajectory[-1, 1]

            for d in data:
                final_r.append(d.trajectory[-1, 0] - DC_final_r)
                final_z.append(d.trajectory[-1, 1] - DC_final_z)
                T.append(d.T)
        else:
            for d in data:
                final_r.append(d.trajectory[-1, 0])
                final_z.append(d.trajectory[-1, 1])
                T.append(d.T)

        order = np.argsort(T)
        final_r_sorted = np.array(final_r)[order]
        final_z_sorted = np.array(final_z)[order]
        T_sorted = np.array(T)[order]
        axs[0].plot(T_sorted, final_r_sorted, label=label)
        axs[1].plot(T_sorted, final_z_sorted, label=label)

    axs[0].set_xlabel('T')
    axs[0].set_ylabel('r-r_DC')
    axs[0].autoscale()
    axs[0].set_xscale('log')

    axs[1].set_xlabel('T')
    axs[1].set_ylabel('z-z_DC')
    axs[1].autoscale()
    axs[1].set_xscale('log')

    axs[1].legend()

    plt.tight_layout()
    plt.show()
def filter_filenames_by_variables(directory, variable_dict):
    matching_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            try:
                parameters, _, _, _, _, _, _ = extract_parameters(filename, directory)
                print(f"Checking file: {filename}")
                print(f"Extracted parameters: {parameters}")
                if all(parameters.get(k) == v for k, v in variable_dict.items()):
                    matching_files.append(filename)
                    print(f"Matched file: {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    return matching_files
def process_directory(input_directory, n_interval):
    # compress the data by saving every n_interval value and puts it into a new directory.
    # _comp'n_interval' is added to the directory name and filenames

    # Create the output directory name
    base_dir_name = os.path.basename(os.path.normpath(input_directory))
    output_directory = os.path.join(os.path.dirname(input_directory), f"{base_dir_name}_comp_{n_interval}")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith('.pkl'):
            input_filepath = os.path.join(input_directory, filename)
            with open(input_filepath, 'rb') as f:
                data = pickle.load(f)

            # Select every n_interval values and include the last value
            time_points = data.time_points[::n_interval]
            trajectory = data.trajectory[::n_interval]
            velocity = data.velocity[::n_interval]

            if data.time_points[-1] not in time_points:
                time_points = np.append(time_points, data.time_points[-1])
            if not np.array_equal(data.trajectory[-1], trajectory[-1]):
                trajectory = np.vstack([trajectory, data.trajectory[-1]])
            if not np.array_equal(data.velocity[-1], velocity[-1]):
                velocity = np.vstack([velocity, data.velocity[-1]])

            # Create a new particle_data instance with the selected values
            new_data = particle_data(
                rhoP=data.rhoP,
                dP=data.dP,
                phi=data.phi,
                V0=data.V0,
                T=data.T,
                trajectory=trajectory,
                velocity=velocity,
                time_points=time_points,
                waveform=data.waveform,
                flowfield=data.flowfield
            )

            # Create the new filename and save the file
            new_filename = filename.replace('.pkl', f'_comp_{n_interval}.pkl')
            output_filepath = os.path.join(output_directory, new_filename)
            with open(output_filepath, 'wb') as f:
                pickle.dump(new_data, f)

    print(f"Processing complete. Processed files are saved in {output_directory}")