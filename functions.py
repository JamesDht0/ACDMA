import math
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from scipy.integrate import quad

# waveforms are defined here
# waveforms should be in the format of:
# base_waveform, (optionally smoothed waveform) and fast computing version using interpolation instead of calculating


class InterpolatingFunction:
    def __init__(self, array_producing_function, x_values, **params):
        """
        Initialize the InterpolatingFunction with a function to produce the array, x-values, and additional parameters.

        Parameters:
        - array_producing_function: A function that takes x_values and produces an array.
        - x_values: An array of x-values for interpolation.
        - params: Additional parameters to be passed to the array-producing function.
        """
        self.array_producing_function = array_producing_function
        self.x_values = x_values
        self.params = params
        self.array = None  # The array will be produced and stored on the first call
        self.mean = 0
        self.__name__ = array_producing_function.__name__[len("generate_"):]

    def __call__(self, V0, T, t, phi):
        """
        Interpolate the value at a specific x using the provided parameters.

        Parameters:
        - V0: Voltage amplitude.
        - T: Period.
        - t: Time.
        - phi: Phase shift.

        Returns:
        - interpolated_value: The interpolated magnitude at the specified x.
        """
        if self.array is None:
            # First call: produce and store the array
            self.array = self.array_producing_function(self.x_values, **self.params)
            self.mean = np.mean(self.array)

        if T != 0:
            phase = (t % T / T + phi / (2 * math.pi)) % 1

            # Interpolate using the stored array
            interpolated_value = np.interp(phase, self.x_values, self.array)

            interpolated_value *= -V0
            return interpolated_value
        else:
            return -V0 * self.mean


x_values = np.linspace(0,1,10000)
def generate_combinations_of_sin(x_values,T_rel,A):

    #T_rel = [1,1/2,1/3]
    #A = [1,-1,1]

    # generate combinations of sin wave plus one.
    # restricts the maximum oscillation to 2N. this is done to account for limited voltage swing to avoid plasma.
    # relative period to the base period. T_specific = T_rel*T. T_rel must be 1/int where int > 1 # relative amplitudes to the base component.
    N = 100
    waveform = np.zeros_like(x_values)
    if A[0] != 0:
        for i in range(0,len(T_rel)):
            if A[i] != 0 and T_rel[i] != 0:
                waveform += A[i]*np.sin(x_values*2*math.pi/T_rel[i])
        waveform /= (max(waveform) - min(waveform))/2/N
    waveform += 1
    return waveform
def generate_smoothed_square_wave(x_values):
    a = 0.6
    sigma = 1000
    pos_constant = (1) / a+1
    neg_constant = (1) / (a - 1)+1

    # Initialize the waveform array
    waveform = np.zeros_like(x_values)

    # Assign values based on the interval
    waveform[x_values < a] = pos_constant
    waveform[x_values >= a] = neg_constant
    extended_waveform = np.concatenate((waveform, waveform, waveform))
    smoothed_waveform = gaussian_filter1d(extended_waveform, sigma, mode='reflect')
    smoothed_waveform = np.array(smoothed_waveform)

    smoothed_waveform = smoothed_waveform[len(waveform):2 * len(waveform)]
    # Adjust to preserve the mean

    max_swing = max(abs(smoothed_waveform))
    smoothed_waveform /= max_swing
    smoothed_waveform -= np.mean(smoothed_waveform)
    smoothed_waveform *= 100
    smoothed_waveform -= np.mean(smoothed_waveform)
    smoothed_waveform += 0
    return smoothed_waveform


smoothed_square_wave = InterpolatingFunction(generate_smoothed_square_wave, x_values)
combinations_of_sin = InterpolatingFunction(generate_combinations_of_sin,x_values,T_rel = [1,1/3,1/5,1/7,1/9,1/11,1/13,1/15],A = [1,1/3,1/5,1/7,1/9,1/11,1/13,1/15])

def DC(V0,T,t,phi):
    return -V0

def one_plus_sin(V0,T,t,phi):

    if T != 0:
        phase = 2 * math.pi * (t % T) / T + phi
        return -V0 * (1 + np.sin(phase))
    else:
        return -V0

def one_plus_10sin(V0,T,t,phi):

    if T != 0:
        phase = 2 * math.pi * t / T + phi
        return -V0*(1+10*np.sin(phase))
    else:
        return -V0

def one_plus_100sin(V0,T,t,phi):

    if T != 0:
        phase = 2 * math.pi * t / T + phi
        return -V0*(1+100*np.sin(phase))
    else:
        return -V0
def N_plus_sin(V0,T,t,phi):
    N = 100
    if T != 0:
        phase = 2 * math.pi * t / T + phi
        return -V0*(1+N*np.sin(phase))
    else:
        return -V0

def abs_sine(V0,T,t,phi):

    if T != 0:
        phase = 2*math.pi*t/T + phi
        # pi/2 to compute the mean value
        return -V0*math.pi/2*abs(np.sin(phase))
    else:
        return -V0

def half_sawtooth(V0,T,t,phi):
    if T != 0:
        # increase exponentially to a certain value then decrease exponentially
        # the rate of increase and decrease are determined by the coeffs
        a = 5 # coeff of charging
        b = 1 # coeff of discharging
        c = 1/0.48933336199819516# scaling coeff, used to have time average value of 1
        phase = (2 * math.pi * (t % T) / T + phi) % (2*math.pi)/(2*math.pi)
        # note that phase is defined different here for the function to be defined between 0 and 1
        return -(min(np.exp(a*phase),np.exp(b*(1-phase)))-1)*c*V0
    else:
        return -V0

def smoothed_half_sawtooth(V0,T,t,phi):
    window_size = 0.1
    # window size measured in the fraction of time period it is averaging over.
    sample_N = 5
    # total number of samples to be taken to find the average, should be an odd number to include the central point
    sample_step = window_size/(sample_N-1)
    sum = 0
    for i in range(0,sample_N):
        sum += half_sawtooth(V0,T,t+T*(-sample_step*(sample_N-1)/2+i*sample_step),phi)
    return sum/sample_N

# smoothed sawtooth function is rather computationally expensive, since it essentially computes half sawtooth 'sample_N' times.
# A better solution might be to have a pre-computed list ( within one cycle) as a look up table.

partial_smooth_half_sawtooth = partial(smoothed_half_sawtooth,V0=1,T=1,phi=0)
def calculate_para_half_sawtooth():
    result = quad(partial_smooth_half_sawtooth,0,1)
    print(result)

#calculate_para_half_sawtooth()

# This is the code to speed up the half sawtooth calculation by calculating the list to interpolate the first time this function is called.
# The parameters are calculated in the first run as well

def square_wave(V0,T,t,phi):
    a = 0.9
    # a is the fraction of period that the function is positive. (voltage is negative)
    if T != 0 :
        phase = ((t % T) / T + phi / (2 * math.pi)) % 1
        if phase <= a:
            return -V0/(2*a-1)
        else:
            return V0/(2*a-1)
    else:
        return -V0

def plot_waveform(waveform):
    t_arr = np.linspace(0,3,30000)
    V = []
    V0 = 1
    T = 1
    phi = 0
    for t in t_arr:
        V.append(waveform(V0,T,t,phi))


    print(np.mean(V))
    plt.plot(t_arr,V)
    plt.show()

plot_waveform(combinations_of_sin)

# flowfields are defined here:

def linear(x):
    gradv = 5
    v0 = -0.05
    # V0 is the velocity at r = 0
    return np.array([0,v0+gradv*x[0]])

def neg_linear(x):
    gradv = 100
    v0 = 1
    # V0 is the velocity at r = 0
    return np.array([0, v0 + gradv * x[0]])

def linear_radial(x):
    gradrvz = -100
    gradrvr = 50
    vz0 = 2
    vr0 = -0.5
    return np.array([vr0+gradrvr*x[0],vz0+gradrvz*x[0]])

def radial_sink(x):
    gradrvz = -100
    vz0 = 2
    return np.array([2e-4/x[0], vz0 + gradrvz * x[0]])
def linear_offset(x):
    gradv = -100
    v0 = 2
    v_offset = 1
    # V0 is the velocity at r = 0
    return np.array([0,v_offset+v0+gradv*x[0]])

def linear_axial_acceleration(x): # improves axial separation a bit
    gradrvz = -100
    vz0 = 2
    gradzvz = 10
    # V0 is the velocity at r = 0
    return np.array([0, vz0 + gradrvz * x[0]+gradzvz*x[1]])
def const(x):
    return np.array([0.0,0.0])


# below are flowfields for spherical design, should be used with the spherical electric field in solver

def spherical_simple_sink(x):
    c1 = 2 # unit 1/s
    return np.array([-x[1]/x[0]*np.sqrt(x[0]**2 + x[1]**2)*c1,np.sqrt(x[0]**2 + x[1]**2)*c1])
#def sphereical_simple_vortex(x):

def spherical_simple_vortex(x):
    c1 = 0.0002
    return np.array([-x[1]/(x[0]**2+x[1]**2)*c1,x[0]/(x[0]**2+x[1]**2)*c1])

def spherical_tornado(x):
    c1 = 0.0002
    c2 = 1e-8
    return np.array([-x[1]/(x[0]**2+x[1]**2)*c1,x[0]/(x[0]**2+x[1]**2)*c1]) - np.array(([c2 * x[0] / ((x[0] ** 2 + x[1] ** 2) ** (3 / 2)), c2 * x[1] / ((x[0] ** 2 + x[1] ** 2) ** (3 / 2))]))
def spherical_E(x):
    V = 1
    E0 = 1
    return [V * E0 * x[0] / ((x[0] ** 2 + x[1] ** 2) ** (3 / 2)), V * E0 * x[1] / ((x[0] ** 2 + x[1] ** 2) ** (3 / 2))]
def plot_vector_field(vector_function, x_range=(-0.02, 0.02), y_range=(-0.02, 0.02), grid_size=20):
    """
    Plot a vector field given a vector-generating function.

    Parameters:
    - vector_function: A function that takes (x, y) and returns (vx, vy).
    - x_range: A tuple specifying the range of x values (default is (-2, 2)).
    - y_range: A tuple specifying the range of y values (default is (-2, 2)).
    - grid_size: The number of points in the grid along each axis (default is 20).
    """
    # Generate a grid of points
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x, y)

    # Compute the vectors at each point on the grid
    [U, V] = vector_function([X, Y])

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the vector field
    ax.quiver(X, Y, U, V)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Vector Field Plot')

    # Set aspect of the plot to be equal
    ax.set_aspect('equal')

    # Show the plot
    plt.show()

#plot_vector_field()

waveform_functions = {
    'DC': DC,
    'one_plus_sin': one_plus_sin,
    'half_sawtooth': half_sawtooth,
    'smoothed_half_sawtooth': smoothed_half_sawtooth,
    'one_plus_10sin': one_plus_10sin,
    'one_plus_100sin': one_plus_100sin,
    'abs_sine': abs_sine,
    'N_plus_sin': N_plus_sin,
    'combinations_of_sin':combinations_of_sin,
    'square_wave': square_wave,
    'smoothed_square_wave': smoothed_square_wave

}

flowfield_functions = {
    'linear': linear,
    'neg_linear': neg_linear,
    'const': const,
    'linear_offset' : linear_offset,
    'linear_radial': linear_radial,
    'radial_sink' : radial_sink,
    'linear_axial_acceleration':linear_axial_acceleration,
    'spherical_simple_sink' : spherical_simple_sink,
    'spherical_simple_vortex' : spherical_simple_vortex,
    'spherical_tornado' : spherical_tornado
}