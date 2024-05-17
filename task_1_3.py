import models
import numpy as np
import matplotlib.pyplot as plt

t = 500

M = 500
R = 20
step_model = models.StepModel(m = M, r = R)
sig = 0.5
bet = 2
ramp_model = models.RampModel(beta = bet, sigma = sig)

def trial_fano(model, iterations, t):
    
    arr = np.empty((0,t))
    for i in range(iterations):
        spikes, jumps = model.simulate(T=t, get_rate = False)
        arr = np.append(arr, spikes, axis = 0)

    mean = np.array([])
    var = np.array([])
    for i in range(t):
        sum = 0
        sqr_sum = 0
        for j in range(iterations):
            sum += arr[j][i]
            sqr_sum += arr[j][i] ** 2
        mean = np.append(mean, sum / iterations)
        var = np.append(var, sqr_sum / iterations - (sum / iterations)**2)
    fano = var / mean
    return mean, var, fano

mean, var, fano = trial_fano(ramp_model, 500, t)
spike_times = np.linspace(0, 1, num = t, endpoint = False)

#plt.plot(spike_times, mean, label = 'mean')
#plt.plot(spike_times, var, label = 'var')
plt.plot(spike_times, fano, label = 'fano')
plt.xlabel('time (s)   ' + 't=' + str(t))
#plt.title('Fano factor of step model  '+'m ='+str(M)+'  r='+str(R))
plt.title('Fano factor of ramp model  ' + '$\\beta$=' + str(bet) + '  $\sigma$=' + str(sig))
plt.show()