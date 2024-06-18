import math
import numpy as np
from functools import partial
from scipy.integrate import quad

# waveforms are defined here

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
        return -V0*2*(1/2+np.sin(phase))
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
        a = 1 # coeff of charging
        b = 1 # coeff of discharging
        c = 1 # scaling coeff, used to have time average value of 1
        phase = (2 * math.pi * (t % T) / T + phi) % (2*math.pi)/(2*math.pi)
        # note that phase is defined different here for the function to be defined between 0 and 1
        return (min(np.exp(a*phase),np.exp(b*(1-phase)))-1)*c*V0
    else:
        return -V0

def half_sawtooth_calc(t):
    V0 = 1
    T = 1
    phi = 0
    a = 1 # coeff of charging
    b = 1 # coeff of discharging
    c = 1/0.29744254140025633 # scaling coeff, used to have time average value of 1
    phase = (2 * math.pi * (t % T) / T + phi) % (2*math.pi)/(2*math.pi)
    # note that phase is defined different here for the function to be defined between 0 and 1
    return (min(np.exp(a*phase),np.exp(b*(1-phase)))-1)*c*V0
    return t
def calculate_para_half_sawtooth():
    result = quad(half_sawtooth_calc,0,1)
    print(result)

#calculate_para_half_sawtooth()

# flowfields are defined here:

def linear(x):
    gradv = -100
    v0 = 2
    # V0 is the velocity at r = 0
    return np.array([0,v0+gradv*x[0]])
def const(x):
    return np.array([0,1])

waveform_functions = {
    'DC': DC,
    'one_plus_sin': one_plus_sin,
    'half_sawtooth': half_sawtooth,
    '_075_plus_sin': _075_plus_sin,
    '_05_plus_sin': _05_plus_sin,
    'abs_sine' : abs_sine
}

flowfield_functions = {
    'linear': linear,
    'const': const
}


