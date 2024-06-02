# import
from inference import *
from HMM_models import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

#common parameters
x0 = 0.2
Rh = 75
T = 100
K = 100

#parameter spaces
M = 30
beta_space = np.linspace(0, 4, num=M)
sigma_space = np.exp(np.linspace(np.log(0.4), np.log(4), num=M))
m_space = np.linspace(0, T, num=M)
r_space = np.arange(10) + 1
x0_space = np.linspace(0, 0.5, num=K)

# define parameters ramp
beta = 2
sigma = 0.5

# define parameters step
m = 40
r = 5

rhmm = HMM_Ramp(beta, sigma, K, x0, Rh, T)
shmm = HMM_Step(m, r, x0, Rh, T)

N = 10
rhmm_datas = np.empty([N, T], dtype = np.int32)
shmm_datas = np.empty([N, T], dtype = np.int32)
for i in range(N):
    latent_ramp, rate_ramp, spike_ramp = rhmm.simulate()
    latent_step, rate_step, spike_step = shmm.simulate()
    rhmm_datas[i] = spike_ramp
    shmm_datas[i] = spike_step

def compute_likelihood(model, datas):
    likelihood = 0
    for d in datas: 
        ll = poisson_logpdf(d, model.lambdas, mask=None)
        likelihood += hmm_normalizer(model.initial_distribution, model.transition_matrix, ll)
    return likelihood

def sweep_ramp_models(beta_space, sigma_space, datas):
    l_matrix = np.empty([len(beta_space), len(sigma_space)])
    for i in range(len(beta_space)):
        for j in range(len(sigma_space)):
            model = HMM_Ramp(beta_space[i], sigma_space[j], K, x0, Rh, T)
            l_matrix[i][j] = compute_likelihood(model, datas)
    return l_matrix

def sweep_step_models(m_space, r_space, datas):
    l_matrix = np.empty([len(m_space), len(r_space)])
    for i in range(len(m_space)):
        for j in range(len(r_space)):
            model = HMM_Step(m_space[i], r_space[j], x0, Rh, T)
            l_matrix[i][j] = compute_likelihood(model, datas)
    return l_matrix

ramp_matrix = sweep_ramp_models(beta_space, sigma_space, rhmm_datas)
max_index_flat = np.argmax(ramp_matrix)
max_index_2d = np.unravel_index(max_index_flat, ramp_matrix.shape)
plt.scatter(max_index_2d[1], max_index_2d[0], color='red', label='Max Value')
norm1 = mcolors.PowerNorm(gamma=3)
plt.imshow(ramp_matrix, cmap='hot', norm=norm1, interpolation='bilinear', origin='lower', aspect = 'auto')
plt.xlabel('$\sigma$')
plt.ylabel('$\\beta$')
plt.colorbar()
plt.title('Likelihood heatmap for $\\beta$={:.1f}, $\sigma$={:.1f}'.format(beta,sigma))
plt.show()

step_matrix = sweep_step_models(m_space, r_space, shmm_datas)
max_index_flat = np.argmax(step_matrix)
max_index_2d = np.unravel_index(max_index_flat, step_matrix.shape)
plt.scatter(max_index_2d[1], max_index_2d[0], color='red', label='Max Value')
norm2 = mcolors.PowerNorm(gamma=3)
plt.imshow(step_matrix, cmap='hot', norm=norm2, interpolation='bilinear', origin='lower', aspect = 'auto')
plt.xlabel('$r$')
plt.ylabel('$m$')
plt.colorbar()
plt.title('Likelihood heatmap for $m$={:d}, $r$={:d}'.format(m,r))
plt.show()