import math
import numpy as np



# waveforms are defined here

def DC(V0,T,t,phi):
    return -V0

def one_plus_sin(V0,T,t,phi):

    if T != 0:
        phase = 2 * math.pi * (t % T) / T + phi
        return -V0 * (1 + np.sin(phase))
    else:
        return V0

def _075_plus_sin(V0,T,t,phi):

    if T != 0:
        phase = 2 * math.pi * t / T + phi
        return -V0*4/3*(3/4+np.sin(phase))
    else:
        return V0

def abs_sine(V0,T,t,phi):

    if T != 0:
        phase = 2*math.pi*t/T + phi
        # pi/2 to compute the mean value
        return -V0*math.pi/2*abs(np.sin(phase))
    else:
        return V0

def half_sawtooth(V0,T,t,phi):


    if T != 0:
        # increase linearly to 2V0 and instantly drop back to 0
        phase = (2 * math.pi * (t % T) / T + phi) % (2*math.pi)
        return V0*2*(2*math.pi-phase)/(2*math.pi)
    else:
        return V0

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
    'abs_sine' : abs_sine
}

flowfield_functions = {
    'linear': linear,
    'const': const
}

