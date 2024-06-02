# import
from inference import *
from HMM_models import *
import matplotlib.pyplot as plt

#common parameters
x0 = 0.2
Rh = 75
T = 100

# define parameters step
m = 32
r = 20

shmm = HMM_Step(m, r, x0, Rh, T)
latent_step, rate_step, spike_step = shmm.simulate()
ll_step = poisson_logpdf(spike_step, shmm.lambdas)
posterior_step, normalizer_step = hmm_expected_states(shmm.initial_distribution, shmm.transition_matrix, ll_step)

prob = np.array([i[-1] for i in posterior_step])
actual = np.array([1 if i == Rh else 0 for i in rate_step])

plt.plot(np.arange(T), prob, label = 'Computed Firing Probability')
plt.plot(np.arange(T), actual, label = 'Actual Firing State')
plt.xlabel('Time index ($ \Delta t = {:d} ms$)'.format(int(1000/T)))
plt.ylabel('$s_t$')
plt.legend(loc='lower right')
plt.title('Smoothing of Step model: $m$={:d}, $r$={:d}'.format(m,r))
plt.show()