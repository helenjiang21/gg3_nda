import models
import numpy as np
import matplotlib.pyplot as plt


def plot_spike_trains(model, t, iterations):
    
    spike_times = np.linspace(0, 1, num = t, endpoint = False)
    spike_times = spike_times + 1/t
    
    for i in range(iterations):
        spikes, jumps, rate = model.simulate(T = t, get_rate = True)
        jump_time = spike_times[get_jump(rate, 50)]
        x, y = get_spike_train(spikes, spike_times, i)
        plt.scatter(x, y, color = 'blue')
        plt.plot(jump_time,i,'ro') 

def get_spike_train(spikes, spike_times, row_index):
    x = np.array([])
    y = np.array([])
    for i in range(spike_times.shape[0]):
        if spikes[0][i] != 0:
            x = np.append(x, spike_times[i])
            y = np.append(y, row_index)
    return x, y
    
def get_jump(rate, jump):
    for i in range(rate[0].shape[0]):
        #print(rate[0][i])
        if rate[0][i] == jump:
            return i



t = 1000

#M = 500
#R = 500
#step_model = models.StepModel(m = M, r = R)
#plot_spike_trains(step_model, t, 30)
#plt.title('Spike trains of step model  '+'m ='+str(M)+'  r='+str(R))

sig = 0.1
bet = 5
ramp_model = models.RampModel(beta = bet, sigma = sig)
plot_spike_trains(ramp_model, t, 30)
plt.title('Spike trains of ramp model  ' + '$\\beta$=' + str(bet) + '  $\sigma$=' + str(sig))

plt.yticks([])
plt.xlabel('time (s)   ' + 't=' + str(t))
plt.show()