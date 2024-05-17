import models
import numpy as np
import matplotlib.pyplot as plt

t = 10000
step_model = models.StepModel(m = t/2)
ramp_model = models.RampModel(beta = 3)

def func(t, row_index_step, row_index_ramp):
    step_spikes, step_jumps = step_model.simulate(T=t, get_rate=False)
    ramp_spikes, ramp_xs = ramp_model.simulate(T=t, get_rate=False)

    spike_times = np.linspace(0, 1, num = t, endpoint = False)
    spike_times = spike_times + 1/t

    step_x = np.array([])
    ramp_x = np.array([])
    step_y = np.array([])
    ramp_y = np.array([])

    for i in range(t):
        if step_spikes[0][i] != 0:
            step_x = np.append(step_x, spike_times[i])
            step_y = np.append(step_y, row_index_step)
        if ramp_spikes[0][i] != 0:
            ramp_x = np.append(ramp_x, spike_times[i])
            ramp_y = np.append(ramp_y, row_index_ramp)
        pass
    return step_x, ramp_x, step_y, ramp_y

n = 10
for i in range(n):
    step_x, ramp_x, step_y, ramp_y = func(t, i , i+n+2)
    if i == 0:
        plt.scatter(step_x, step_y, color = 'blue', label = 'step')
        plt.scatter(ramp_x, ramp_y, color = 'red', label = 'ramp')        
    else:
        plt.scatter(step_x, step_y, color = 'blue')
        plt.scatter(ramp_x, ramp_y, color = 'red')

plt.axis('off')
plt.title('spike train')
plt.legend()
plt.show()