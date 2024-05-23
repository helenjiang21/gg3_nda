import models
import numpy as np
import matplotlib.pyplot as plt

t = 100

M = 30.129
R = 2.434
step_model = models.StepModel(m = M, r = R)
sig = 0.269
bet = 1.468
ramp_model = models.RampModel(beta = bet, sigma = sig)

def trial_average(model, iterations, t, N):
    
    bin = np.zeros(t)
    for i in range(iterations):
        spikes, jumps = model.simulate(T=t, get_rate = False)
        bin += spikes[0]
    bin  = bin / iterations
    
    bin = np.convolve(bin, np.ones(N)/N, mode='valid')
    return bin * t

bin1 = trial_average(ramp_model, 5000, t, 5)
bin2 = trial_average(step_model, 5000, t, 5)
spike_times = np.linspace(0, 1, num = bin1.shape[0], endpoint = False)

plt.plot(spike_times, bin1, label = 'm ='+str(M)+'  r='+str(R))
plt.plot(spike_times, bin2, label = '$\\beta$=' + str(bet) + '  $\sigma$=' + str(sig))
plt.legend()
plt.title('PSTH of ramp vs step model over 5000 samples')
plt.xlabel('time (s)   ' + 't=' + str(t))
plt.show()

# best fit: m = 20.738, r = 2.397, beta = 2.229, sigma = 0.424