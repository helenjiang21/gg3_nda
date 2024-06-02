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
ramp_priori = np.ones([len(beta_space), len(sigma_space)]) / (len(beta_space)*len(sigma_space))
step_priori = np.ones([len(m_space), len(r_space)]) / (len(m_space)*len(r_space))

# define parameters ramp
beta = 2
sigma = 2

# define parameters step
m = 60
r = 5

rhmm = HMM_Ramp(beta, sigma, K, x0, Rh, T)
shmm = HMM_Step(m, r, x0, Rh, T)

N = 30
rhmm_datas = np.empty([N, T], dtype = np.int32)
shmm_datas = np.empty([N, T], dtype = np.int32)
for i in range(N):
    latent_ramp, rate_ramp, spike_ramp = rhmm.simulate()
    latent_step, rate_step, spike_step = shmm.simulate()
    rhmm_datas[i] = spike_ramp
    shmm_datas[i] = spike_step

def compute_ll(model, datas):
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
            l_matrix[i][j] = compute_ll(model, datas)
    return l_matrix - np.min(l_matrix)

def sweep_step_models(m_space, r_space, datas):
    l_matrix = np.empty([len(m_space), len(r_space)])
    for i in range(len(m_space)):
        for j in range(len(r_space)):
            model = HMM_Step(m_space[i], r_space[j], x0, Rh, T)
            l_matrix[i][j] = compute_ll(model, datas)
    return l_matrix - np.min(l_matrix)

def compute_posteriori(likelihood, priori):
    mat = np.empty_like(likelihood)
    for i in range(likelihood.shape[0]):
        for j in range(likelihood.shape[1]):
            mat[i][j] = likelihood[i][j] * priori[i][j]
    norm = np.sum(mat)
    mat = mat / norm
    return mat

ramp_l = np.exp(sweep_ramp_models(beta_space, sigma_space, rhmm_datas))
step_l = np.exp(sweep_step_models(m_space, r_space, shmm_datas))

ramp_pos = compute_posteriori(ramp_l, ramp_priori)
step_pos = compute_posteriori(step_l, step_priori)


E_sigma = np.matmul(np.sum(ramp_pos, axis = 0), sigma_space)
E_beta = np.matmul(np.sum(ramp_pos, axis = 1), beta_space)
plt.scatter((np.log(E_sigma)-np.log(0.4)) / (np.log(4)-np.log(0.4))*M, E_beta/4*M, color = 'green', label = 'E')
max_index_flat = np.argmax(ramp_pos)
max_index_2d = np.unravel_index(max_index_flat, ramp_pos.shape)
plt.scatter(max_index_2d[1], max_index_2d[0], color='red', label='MAP')
plt.scatter((np.log(sigma)-np.log(0.4)) / (np.log(4)-np.log(0.4))*M, beta/4*M, color = 'blue', label = 'True')

plt.imshow(ramp_pos, cmap='hot', norm=None, interpolation='bilinear', origin='lower', aspect = 'auto')
plt.xlabel('$\sigma$')
plt.ylabel('$\\beta$')
plt.colorbar()
plt.legend()
plt.title('Likelihood heatmap for $\\beta$={:.1f}, $\sigma$={:.1f}'.format(beta,sigma))
plt.show()

E_r = np.matmul(np.sum(step_pos, axis = 0), r_space)
E_m = np.matmul(np.sum(step_pos, axis = 1), m_space)
plt.scatter((E_r-1)/9*10, E_m/T*M, color = 'green', label = 'E')
max_index_flat = np.argmax(step_pos)
max_index_2d = np.unravel_index(max_index_flat, step_pos.shape)
plt.scatter(max_index_2d[1], max_index_2d[0], color='red', label='MAP')
plt.scatter((r-1)/9*10, m/T*M, color = 'blue', label = 'True')

plt.imshow(step_pos, cmap='hot', norm=None, interpolation='bilinear', origin='lower', aspect = 'auto')
plt.xlabel('$r$')
plt.ylabel('$m$')
plt.colorbar()
plt.legend()
plt.title('Likelihood heatmap for $m$={:d}, $r$={:d}'.format(m,r))
plt.show()