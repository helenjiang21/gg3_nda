import numpy as np
import matplotlib.pyplot as plt


def get_transition_matrix(m, r):
    p = r / (m+r)
    transition_matrix = np.zeros([r+1,r+1])
    for i in range(r):
        transition_matrix[i][i] = 1 - p
        transition_matrix[i][i+1] = p
    transition_matrix[r][r] = 1
    return transition_matrix

def get_initial_distribution(m, r):
    dist = np.zeros(r+1)
    dist[0] = 1
    mat = get_transition_matrix(m, r)
    for i in range(r-1):
        dist = np.matmul(dist, mat)
        print(dist)
    return dist

def simulate(init_dist, trans_matrix, length, dt):
    t = np.arange(length) * dt
    
    s = np.empty(length)
    states = np.arange(r+1)
    s[0] = np.random.choice(states, p = init_dist)
    for i in range (1,length):
        s[i] = np.random.choice(states, p = trans_matrix[int(s[i-1])])
    
    step = np.empty(length)
    for i in range(length):
        if s[i] == r:
            step[i] = 1
        else:
            step[i] = 0
    return t, step

#############

m = 20
r = 10
dt = 1

trans = get_transition_matrix(m, r)
init = get_initial_distribution(m, r)

for i in range(20):
    t, s = simulate(init, trans, 100, dt)
    plt.plot(t, s, color = 'C0')
plt.ylim(bottom = 0, top = 1)
plt.xlabel('time (s)')
plt.ylabel('firing rate')
plt.show()