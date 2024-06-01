# import
from inference import *
from HMM_models_2 import *
import matplotlib.pyplot as plt

#common parameters
x0 = 0.2
Rh = 75
T = 100

# define parameters ramp
beta = 1.5
sigma = 0.5
K = 50

# define parameters step
m = 32
r = 20

# Replace numerical arguments with variable name
rhmm = HMM_Ramp(beta, sigma, K, x0, Rh, T)
shmm = HMM_Step(m, r, x0, Rh, T)

latent_ramp, rate_ramp, spike_ramp = rhmm.simulate()
latent_step, rate_step, spike_step = shmm.simulate()

t = np.arange(T)/T

ll_ramp = poisson_logpdf(spike_ramp, rhmm.lambdas)
ll_step = poisson_logpdf(spike_step, shmm.lambdas)

posterior_ramp, normalizer_ramp = hmm_expected_states(rhmm.initial_distribution, rhmm.transition_matrix, ll_ramp)
posterior_step, normalizer_step = hmm_expected_states(shmm.initial_distribution, shmm.transition_matrix, ll_step)

plt.plot(np.arange(T), latent_step)
plt.imshow(posterior_step.T, cmap='hot', interpolation='bilinear', origin='lower', vmin=0.0, vmax=0.3, aspect = 'auto')
plt.xlabel('Time index ($ \Delta t = 10 ms$)')
plt.ylabel('$s_t$')
plt.colorbar()
plt.show()

plt.plot(np.arange(T), latent_ramp)
plt.imshow(posterior_ramp.T, cmap='hot', interpolation='bilinear', origin='lower', vmin=0.0, vmax=0.3, aspect = 'auto')
plt.xlabel('Time index ($ \Delta t = 10 ms$)')
plt.ylabel('$s_t$')
plt.colorbar()
plt.show()