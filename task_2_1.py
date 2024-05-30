import numpy as np
import matplotlib.pyplot as plt

def get_initial_distribution(K, x0, sig, dt):
    states = np.linspace(0,1,num = K)
    arr = (states - x0) / (sig*np.sqrt(dt))
    dist = normal_dist(arr)
    dist_norm = dist / np.sum(dist)    
    return dist_norm

def get_transition_matrix(K, bet, sig, dt):
    states = np.linspace(0,1,num = K)
    trans = np.empty([K,K])
    for i in range(K-1):
        arr = (states - states[i] - bet*dt) / (sig*np.sqrt(dt))
        dist = normal_dist(arr)
        dist_norm = dist / np.sum(dist)
        trans[i] = dist_norm
    trans[K-1] = np.zeros(K)
    trans[K-1][K-1] = 1
    return trans

def normal_dist(x, mean = 0, sd = 1):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def simulate(init_dist, trans_matrix, length, dt, K):
    t = np.arange(length) * dt
    s = np.empty(length)
    states = np.arange(K)
    s[0] = np.random.choice(states, p = init_dist)
    for i in range (1,length):
        s[i] = np.random.choice(states, p = trans_matrix[int(s[i-1])])
    s = s / K
    return t, s

#############

K = 200
dt = 1

x0 = 0.5
bet = 0
sig = 0.001

init = get_initial_distribution(K, x0, sig, dt)
trans = get_transition_matrix(K, bet, sig, dt)

#simulate 10 trials of ramp model in one plot
for i in range(10):
    t, s = simulate(init, trans, 100, dt, K)
    plt.plot(t, s, color = 'C0')
plt.ylim(bottom = 0, top = 1)
plt.xlabel('time (s)')
plt.ylabel('firing rate')
plt.title('$x_0$={:.1f}, $\\beta$={:.0f}, $\sigma$={:.3f}'.format(x0,bet,sig))
plt.show()