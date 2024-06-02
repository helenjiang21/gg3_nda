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

ll_ramp = poisson_logpdf(spike_ramp, rhmm.lambdas)
ll_step = poisson_logpdf(spike_step, shmm.lambdas)

posterior_ramp, normalizer_ramp = hmm_expected_states(rhmm.initial_distribution, rhmm.transition_matrix, ll_ramp)
posterior_step, normalizer_step = hmm_expected_states(shmm.initial_distribution, shmm.transition_matrix, ll_step)

expected_ramp = np.array([np.matmul(rhmm.states, i) for i in posterior_ramp])
expected_step = np.array([np.matmul(shmm.states, i) for i in posterior_step])

plt.plot(np.arange(T), latent_step, label = 'Actual states')
plt.plot(np.arange(T), expected_step, label = 'Expected states')
plt.imshow(posterior_step.T, cmap='hot', interpolation='bilinear', origin='lower', vmin=0.0, vmax=0.3, aspect = 'auto')
plt.xlabel('Time index ($ \Delta t = {:d} ms$)'.format(int(1000/T)))
plt.ylabel('$s_t$')
plt.colorbar()
plt.legend(loc='lower right')
plt.title('Smoothing of Step model: $m$={:d}, $r$={:d}'.format(m,r))
plt.show()

plt.plot(np.arange(T), latent_ramp, label = 'Actual states')
plt.plot(np.arange(T), expected_ramp, label = 'Expected states')
plt.imshow(posterior_ramp.T, cmap='hot', interpolation='bilinear', origin='lower', vmin=0.0, vmax=0.3, aspect = 'auto')
plt.xlabel('Time index ($ \Delta t = {:d} ms$)'.format(int(1000/T)))
plt.ylabel('$s_t$')
plt.colorbar()
plt.legend(loc='lower right')
plt.title('Smoothing of Ramp model: $\\beta$={:.1f}, $\sigma$={:.1f}'.format(beta,sigma))
plt.show()