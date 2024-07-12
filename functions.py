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
    def __init__(self, array_producing_function, x_values):
        """
        Initialize the InterpolatingFunction with a function to produce the array and x-values.

        Parameters:
        - array_producing_function: A function that takes x_values and produces an array.
        - x_values: An array of x-values for interpolation.
        """
        self.array_producing_function = array_producing_function
        self.x_values = x_values
        self.array = None  # The array will be produced and stored on the first call
        self.mean = 0
        self.__name__ = array_producing_function.__name__[len("generate_"):]

    def __call__(self, V0,T,t,phi):
        """
        Interpolate the value at a specific x using the provided parameters.

        Parameters:
        - x: The x-value for which to get the interpolated magnitude.
        - params: Additional parameters to be applied to the interpolated value.

        Returns:
        - interpolated_value: The interpolated magnitude at the specified x.
        """
        if self.array is None:
            # First call: produce and store the array
            self.array = self.array_producing_function(self.x_values)
            self.mean = np.mean(self.array)
        if T != 0:
            phase = (t%T/T + phi/(2*math.pi))%1

            # Interpolate using the stored array
            interpolated_value = np.interp(phase, self.x_values, self.array)

            interpolated_value *= -V0
            return interpolated_value
        else:
            return -V0*self.mean

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
    mean_smoothed = np.mean(smoothed_waveform)
    smoothed_waveform += 1 - mean_smoothed
    #print(np.mean(smoothed_waveform))
    return smoothed_waveform

x_values = np.linspace(0,1,10000)
smoothed_square_wave = InterpolatingFunction(generate_smoothed_square_wave, x_values)

def DC(V0,T,t,phi):
    return -V0

def one_plus_sin(V0,T,t,phi):

    if T != 0:
        phase = 2 * math.pi * (t % T) / T + phi
        return -V0 * (1 + np.sin(phase))
    else:
        return -V0

def _075_plus_sin(V0,T,t,phi):

    if T != 0:
        phase = 2 * math.pi * t / T + phi
        return -V0*4/3*(3/4+np.sin(phase))
    else:
        return -V0

def _05_plus_sin(V0,T,t,phi):

    if T != 0:
        phase = 2 * math.pi * t / T + phi
        return -V0*(1+2*np.sin(phase))
    else:
        return -V0
def N_plus_sin(V0,T,t,phi):
    N = 10000
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
    t_arr = np.linspace(0,3,300)
    V = []
    V0 = 1
    T = 1
    phi = 0
    for t in t_arr:
        V.append(waveform(V0,T,t,phi))

    plt.plot(t_arr,V)
    plt.show()

#plot_waveform(smoothed_square_wave)

# flowfields are defined here:

def linear(x):
    gradv = -100
    v0 = 2
    # V0 is the velocity at r = 0
    return np.array([0,v0+gradv*x[0]])

def linear_offset(x):
    gradv = -100
    v0 = 2
    v_offset = 1
    # V0 is the velocity at r = 0
    return np.array([0,v_offset+v0+gradv*x[0]])

def const(x):
    return np.array([0,1])

waveform_functions = {
    'DC': DC,
    'one_plus_sin': one_plus_sin,
    'half_sawtooth': half_sawtooth,
    'smoothed_half_sawtooth': smoothed_half_sawtooth,
    '_075_plus_sin': _075_plus_sin,
    '_05_plus_sin': _05_plus_sin,
    'abs_sine': abs_sine,
    'N_plus_sin': N_plus_sin,
    'square_wave': square_wave,
    'smoothed_square_wave': smoothed_square_wave

}

flowfield_functions = {
    'linear': linear,
    'const': const,
    'linear_offset' : linear_offset
}


