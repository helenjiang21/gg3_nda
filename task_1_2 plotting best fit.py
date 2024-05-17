import models
import numpy as np
import matplotlib.pyplot as plt

t = 1000

M = 20.738
R = 2.397
step_model = models.StepModel(m = M, r = R)
sig = 0.424
bet = 2.229
ramp_model = models.RampModel(beta = bet, sigma = sig)

def trial_average(model, iterations, t, N):
    
    bin = np.zeros(t)
    for i in range(iterations):
        spikes, jumps = model.simulate(T=t, get_rate = False)
        bin += spikes[0]
    bin  = bin / iterations
    
    bin = np.convolve(bin, np.ones(N)/N, mode='valid')
    return bin * t

bin1 = trial_average(ramp_model, 5000, t, 50)
bin2 = trial_average(step_model, 5000, t, 50)
spike_times = np.linspace(0, 1, num = bin1.shape[0], endpoint = False)

plt.plot(spike_times, bin1)
plt.plot(spike_times, bin2)
#plt.title('PSTH over 2500 samples of step model  '+'m ='+str(M)+'  r='+str(R))
#plt.title('PSTH over 100 samples of ramp model  ' + '$\\beta$=' + str(bet) + '  $\sigma$=' + str(sig))
plt.xlabel('time (s)   ' + 't=' + str(t))
plt.show()

# best fit: m = 20.738, r = 2.397, beta = 2.229, sigma = 0.424