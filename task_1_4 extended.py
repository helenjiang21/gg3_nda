import models
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

##
#  A quadratic fit with constant fixed at the initial value of PSTH spikes
#
#  inputs
#  spikes (PSTH across trials): arr 1D, full spike train from t=0 to t=1
#  T: int, number of samples in spikes. T = spikes.length
#  cutoff: int, the first 'cutoff' number of samples will be considered in the polyfit
#  window: if True, a rectangular window of width window_length will be applied to spikes before polyfit
#  window_length: width of rectangular window, used for filtering
#  plot: if True, will generate graph of the poly2D fit
def poly2Dfit(spikes, T, cutoff, window = False, window_length = 1, plot = False):
    
    if window:
        spikes_arr = np.convolve(spikes, np.ones(window_length)/window_length, mode='valid')
    spikes_arr = spikes_arr[:cutoff]
    
    time_arr = np.linspace(0, 1, num = T, endpoint = False)
    time_arr = time_arr + 1/T
    time_arr = time_arr[:cutoff]

    def f(x, a, b):
        return a * x + b * x ** 2 + spikes_arr[0]
    
    popt, _ = curve_fit(f, time_arr, spikes_arr)
    a, b = popt
    fit_arr = np.array([f(x,a,b) for x in time_arr])
    if plot:
        plt.plot(time_arr, spikes_arr)
        plt.plot(time_arr, fit_arr)
        plt.show()
    return a, b


# Sample use

# initializing models
ramp_model = models.RampModel(beta = 2, sigma = 0.5)
step_model = models.StepModel(m = 500, r = 20)

def get_fit_psth(model, iter, t):
    bin = np.zeros(t)
    for i in range(iter):
        spikes, jumps = model.simulate(T=t, get_rate = False)
        bin += spikes[0]
    bin  = bin / iter
    a, b = poly2Dfit(bin, t, 20, window = True, window_length = 5, plot = False)
    return a, b

t = 100
iter = 1000
step_a = np.array([])
step_b = np.array([])
for i in range(100):
    a, b = get_fit_psth(step_model, iter, t)
    step_a = np.append(step_a, a)
    step_b = np.append(step_b, b)

ramp_a = np.array([])
ramp_b = np.array([])
for i in range(100):
    a, b = get_fit_psth(ramp_model, iter, t)
    ramp_a = np.append(ramp_a, a)
    ramp_b = np.append(ramp_b, b)

plt.scatter(ramp_a, ramp_b, label = 'ramp $\\beta=2$ $\sigma=0.5$')
plt.scatter(step_a, step_b, label = 'step $m=500$ $r=20$')
plt.ylabel('First order term')
plt.xlabel('Second order term')
plt.title('Quadratic fit for the first 1/5 datapoints')
plt.legend()
plt.show()