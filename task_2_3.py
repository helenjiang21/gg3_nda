# import
from inference import *
from HMM_models import *
import matplotlib.pyplot as plt

#common parameters
x0 = 0.2
Rh = 75
T = 100

# define parameters ramp
beta = 1.5
sigma = 0.3
K = 50

# define parameters step
m = 32
r = 20

# Replace numerical arguments with variable name
rhmm = HMM_Ramp_Model(beta, sigma, K, x0, Rh)
shmm = HMM_Step_Model(m, r, x0, Rh)

spikes_step, xs_step, rates_step = shmm.simulate()
spikes_ramp, xs_ramp, rates_ramp = rhmm.simulate()

print(shmm.lamb)
print(rhmm.lamb)

ll_ramp = poisson_logpdf(spikes_ramp, rhmm.lamb)[0]
ll_step = poisson_logpdf(spikes_step, shmm.lamb)[0]


posterior_ramp, normalizer_ramp = hmm_expected_states(rhmm.get_initial_distribution(), rhmm.get_transition_matrix(), ll_ramp)
posterior_step, normalizer_step = hmm_expected_states(shmm.get_initial_distribution(), shmm.get_transition_matrix(), ll_step)
print(posterior_ramp)
print(posterior_step)

plt.plot(np.arange(T), xs_step[0])
plt.imshow(posterior_ramp.T, cmap='hot', interpolation='bilinear', origin='lower', vmin=0.0, vmax=0.3, aspect = 'auto')
plt.xlabel('Time index ($ \Delta t = 10 ms$)')
plt.ylabel('$s_t$')
plt.colorbar()
plt.show()