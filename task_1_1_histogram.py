import models
import numpy as np
import matplotlib.pyplot as plt

def hist_jumps(model, t, iterations):
    
    spike_times = np.linspace(0, 1, num = t, endpoint = False)
    spike_times = spike_times + 1/t

    arr = np.array([])
    
    for i in range(iterations):
        spikes, jumps, rate = model.simulate(T = t, get_rate = True)
        jump_time = spike_times[get_jump(rate, 50)]
        arr = np.append(arr, jump_time)
    return arr

def get_jump(rate, jump):
    for i in range(rate[0].shape[0]):
        #print(rate[0][i])
        if rate[0][i] == jump:
            return i
        

t = 1000

#M = 500
#R = 50
#step_model = models.StepModel(m = M, r = R)
#arr = hist_jumps(step_model, t, 100)
#plt.hist(arr, bins=50, range=(0,1), density=True)
#plt.title('Histogram of jump times in step model  '+'m ='+str(M)+'  r='+str(R))

sig = 0.3
bet = 2
ramp_model = models.RampModel(beta = bet, sigma = sig)
arr = hist_jumps(ramp_model, t, 100)
plt.hist(arr, bins=50, range=(0,1), density=True)
plt.title('Histogram of jump times in ramp model  ' + '$\\beta$=' + str(bet) + '  $\sigma$=' + str(sig))

plt.xlabel('time (s)   ' + 't=' + str(t))
plt.show()