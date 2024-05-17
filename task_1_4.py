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
t = 1000
ramp_model = models.RampModel(beta = 2, sigma = 0.5)
step_model = models.StepModel(m = 500, r = 20)

# creating PSTH of spikes
bin = np.zeros(t)
for i in range(5000):
    spikes, jumps = step_model.simulate(T=t, get_rate = False)
    bin += spikes[0]
bin  = bin / 5000

# applying 2D fit
a, b = poly2Dfit(bin, t, 200, window = True, window_length = 10, plot = False)
print('the best fit is y = {:.2f}x^2 + {:.2f}x + const'.format(b,a))