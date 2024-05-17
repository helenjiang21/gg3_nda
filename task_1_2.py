import models
import numpy as np
import matplotlib.pyplot as plt

t = 1000

M = 500
R = 20
step_model = models.StepModel(m = M, r = R)
sig = 0.5
bet = 2
ramp_model = models.RampModel(beta = bet, sigma = sig)

def trial_average(model, iterations, t, N):
    
    bin = np.zeros(t)
    for i in range(iterations):
        spikes, jumps = model.simulate(T=t, get_rate = False)
        bin += spikes[0]
    bin  = bin / iterations
    
    bin = np.convolve(bin, np.ones(N)/N, mode='valid')
    return bin * t

model = step_model
bin = trial_average(model, 100, t, 50)
bin2 = trial_average(model, 500, t, 50)
bin3 = trial_average(model, 2500, t, 50)
spike_times = np.linspace(0, 1, num = bin.shape[0], endpoint = False)

plt.plot(spike_times, bin, label = 'PSTH over 100 samples')
plt.plot(spike_times, bin2, label = 'PSTH over 500 samples')
plt.plot(spike_times, bin3, label = 'PSTH over 2500 samples')
plt.title('PSTH of step model  '+'m ='+str(M)+'  r='+str(R))
#plt.title('PSTH of ramp model  ' + '$\\beta$=' + str(bet) + '  $\sigma$=' + str(sig))
plt.xlabel('time (s)   ' + 't=' + str(t))
plt.legend()
plt.show()
