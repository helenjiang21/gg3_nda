import numpy as np

class HMM_Step():

    def __init__(self, m=50, r=10, x0 = 0.2, Rh=50, T = 100):
        
        self.m = m
        self.r = r
        self.x0 = x0
        self.p = r / (m+r)
        self.Rh = Rh
        self.T = T
        self.dt = 1/T

        self.states = np.arange(self.r+1)

        self.transition_matrix = np.zeros([self.r+1,self.r+1])
        for i in range(self.r):
            self.transition_matrix[i][i] = 1 - self.p
            self.transition_matrix[i][i+1] = self.p
        self.transition_matrix[self.r][self.r] = 1

        self.initial_distribution = np.zeros(self.r+1)
        self.initial_distribution[0] = 1
        for i in range(self.r):
            self.initial_distribution = np.matmul(self.initial_distribution, self.transition_matrix)


        self.lambdas = np.ones(r+1) * self.x0 * self.Rh * self.dt
        self.lambdas[-1] = self.Rh * self.dt

    def emit(self, rate):
        y = np.random.poisson(rate * self.dt)
        return y

    def simulate(self):
        latent = np.empty(self.T)
        rate = np.empty(self.T)
        latent[0] = np.random.choice(self.states, p=self.initial_distribution)
        for i in range(1, self.T):
            latent[i] = np.random.choice(self.states, p=self.transition_matrix[int(latent[i-1])])
        for i in range(self.T):
            if latent[i] == self.r:
                rate[i] = self.Rh
            else:
                rate[i] = self.Rh * self.x0
        spikes = self.emit(rate)
        return latent, rate, spikes
    
class HMM_Ramp():

    def __init__(self, bet=0.5, sig=0.2, K=100, x0 = 0.2, Rh=50, T = 100):
        
        self.bet = bet
        self.sig = sig
        self.K = K
        self.x0 = x0
        self.Rh = Rh
        self.T = T
        self.dt = 1/T

        self.states = np.arange(self.K)

        s = np.linspace(0,1,num = K)
        self.transition_matrix = np.empty([K,K])
        for i in range(K-1):
            arr = (s - s[i] - bet*self.dt) / (sig*np.sqrt(self.dt))
            dist = normal_dist(arr)
            dist_norm = dist / np.sum(dist)
            self.transition_matrix[i] = dist_norm
        self.transition_matrix[K-1] = np.zeros(K)
        self.transition_matrix[K-1][K-1] = 1

        arr = (s - x0) / (sig*np.sqrt(self.dt))
        dist = normal_dist(arr)
        self.initial_distribution = dist / np.sum(dist) 

        self.lambdas = np.arange(self.K)/(self.K-1)*self.Rh*self.dt

    def emit(self, rate):
        y = np.random.poisson(rate * self.dt)
        return y

    def simulate(self):
        latent = np.empty(self.T)
        rate = np.empty(self.T)
        latent[0] = np.random.choice(self.states, p=self.initial_distribution)
        for i in range(1, self.T):
            latent[i] = np.random.choice(self.states, p=self.transition_matrix[int(latent[i-1])])
        for i in range(self.T):
            rate[i] = latent[i]/(self.K-1)*self.Rh
        spikes = self.emit(rate)
        return latent, rate, spikes

def normal_dist(x, mean = 0, sd = 1):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density