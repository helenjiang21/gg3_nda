import numpy as np
import numpy.random as npr

def lo_histogram(x, bins):
    """
    Left-open version of np.histogram with left-open bins covering the interval (left_edge, right_edge]
    (np.histogram does the opposite and treats bins as right-open.)
    Input & output behaviour is exactly the same as np.histogram
    """
    out = np.histogram(-x, -bins[::-1])
    return out[0][::-1], out[1:]

def gamma_isi_point_process(rate, shape):
    """
    Simulates (1 trial of) a sub-poisson point process (with underdispersed inter-spike intervals relative to Poisson)
    :param rate: time-series giving the mean spike count (firing rate * dt) in different time bins (= time steps)
    :param shape: shape parameter of the gamma distribution of ISI's
    :return: vector of spike counts with same shape as "rate".
    """
    sum_r_t = np.hstack((0, np.cumsum(rate)))
    gs = np.zeros(2)
    while gs[-1] < sum_r_t[-1]:
        gs = np.cumsum( npr.gamma(shape, 1 / shape, size=(2 + int(2 * sum_r_t[-1]),)) )
    y, _ = lo_histogram(gs, sum_r_t)

    return y

class HMM_Step_Model():

    def __init__(self, m=50, r=10, x0=0.2, Rh=50, isi_gamma_shape=None, Rl=None, dt=None):
        self.m = m
        self.r = r
        self.x0 = x0

        self.p = r / (m + r)

        self.Rh = Rh
        if Rl is not None:
            self.Rl = Rl

        self.isi_gamma_shape = isi_gamma_shape
        self.dt = dt

        self.lamb = np.ones(r+1) * self.x0 * self.Rh
        self.lamb[0] = self.Rh

    @property
    def params(self):
        return self.m, self.r, self.x0

    @property
    def fixed_params(self):
        return self.Rh, self.Rl

    def emit(self, rate):

        if self.isi_gamma_shape is None:
            # poisson spike emissions
            y = npr.poisson(rate * self.dt)
        else:
            # sub-poisson/underdispersed spike emissions
            y = gamma_isi_point_process(rate * self.dt, self.isi_gamma_shape)

        return y

    def get_transition_matrix(self):
        p = self.r / (self.m+self.r)
        transition_matrix = np.zeros([self.r+1,self.r+1])
        for i in range(self.r):
            transition_matrix[i][i] = 1 - p
            transition_matrix[i][i+1] = p
        transition_matrix[self.r][self.r] = 1
        return transition_matrix

    def get_initial_distribution(self):
        dist = np.zeros(self.r+1)
        dist[0] = 1
        mat = self.get_transition_matrix()
        for i in range(self.r-1):
            dist = np.matmul(dist, mat)
            #print(dist)
        return dist

    def simulate(self, Ntrials=1, T=100, get_rate=True):
        """
        :param Ntrials: (int) number of trials
        :param T: (int) duration of each trial in number of time-steps.
        :param get_rate: whether or not to return the rate time-series
        :return:
        spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
                an array of spike counts in each time-bin (= time step)
        jumps:  shape = (Ntrials,) ; jumps[j] is the jump time (aka step time), tau, in trial j.
        rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
        """
        # set dt (time-step duration in seconds) such that trial duration is always 1 second, regardless of T.
        dt = 1 / T
        self.dt = dt


        spikes, xs, rates = [], [], []
        for tr in range(Ntrials):

            states = np.arange(self.r+1)
            s = np.empty(T)
            s[0] = np.random.choice(states, p = self.get_initial_distribution())
            trans_matrix = self.get_transition_matrix()
            for i in range (1,T):
                s[i] = np.random.choice(states, p = trans_matrix[int(s[i-1])])

            # first set rate at all times to pre-step rate
            rate = np.ones(T) * self.x0 * self.Rh
            # then set rates after jump to self.Rh
            flag = False
            for i in range(T):
                if s[i] == self.r:
                    if not flag:
                        pass
                        jump = i
                    rate[i] = self.Rh
            
            xs.append(s)
            rates.append(rate)
            spikes.append(self.emit(rate))

        if get_rate:
            return np.array(spikes), np.array(xs), np.array(rates)
        else:
            return np.array(spikes), np.array(xs)
        
class HMM_Ramp_Model():

    def __init__(self, beta=0.5, sigma=0.2, K=100, x0=.2, Rh=50, isi_gamma_shape=None, Rl=None, dt=None):
        """
        Simulator of the Ramping Model of Latimer et al. Science 2015.
        :param beta: drift rate of the drift-diffusion process
        :param sigma: diffusion strength of the drift-diffusion process.
        :param x0: average initial value of latent variable x[0]
        :param Rh: the maximal firing rate obtained when x_t reaches 1 (corresponding to the same as the post-step
                   state in most of the project tasks)
        :param isi_gamma_shape: shape parameter of the Gamma distribution of inter-spike intervals.
                            see https://en.wikipedia.org/wiki/Gamma_distribution
        :param Rl: Not implemented. Ignore.
        :param dt: real time duration of time steps in seconds (only used for converting rates to units of inverse time-step)
        """
        self.beta = beta
        self.sigma = sigma
        self.x0 = x0

        self.Rh = Rh
        if Rl is not None:
            self.Rl = Rl

        self.isi_gamma_shape = isi_gamma_shape
        self.dt = dt
        self.K = K

        self.lamb = np.linspace(1,0,num = self.K) * self.Rh

    @property
    def params(self):
        return self.mu, self.sigma, self.x0

    @property
    def fixed_params(self):
        return self.Rh, self.Rl

    def f_io(self, xs, b=None):
        if b is None:
            return self.Rh * np.maximum(0, xs)
        else:
            return self.Rh * b * np.log(1 + np.exp(xs / b))

    def emit(self, rate):

        if self.isi_gamma_shape is None:
            # poisson spike emissions
            y = npr.poisson(rate * self.dt)
        else:
            # sub-poisson/underdispersed spike emissions
            y = gamma_isi_point_process(rate * self.dt, self.isi_gamma_shape)

        return y

    def get_initial_distribution(self, dt):
        states = np.linspace(0,1,num = self.K)
        arr = (states - self.x0) / (self.sigma*np.sqrt(dt))
        dist = self.normal_dist(arr)
        dist_norm = dist / np.sum(dist)    
        return dist_norm

    def get_transition_matrix(self, dt):
        states = np.linspace(0,1,num = self.K)
        trans = np.empty([self.K,self.K])
        for i in range(self.K-1):
            arr = (states - states[i] - self.beta*dt) / (self.sigma*np.sqrt(dt))
            dist = self.normal_dist(arr)
            dist_norm = dist / np.sum(dist)
            trans[i] = dist_norm
        trans[self.K-1] = np.zeros(self.K)
        trans[self.K-1][self.K-1] = 1
        return trans

    def normal_dist(self, x, mean = 0, sd = 1):
        prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
        return prob_density

    def simulate(self, Ntrials=1, T=100, get_rate=True):
        """
        :param Ntrials: (int) number of trials
        :param T: (int) duration of each trial in number of time-steps.
        :param get_rate: whether or not to return the rate time-series
        :return:
        spikes: shape = (Ntrial, T); spikes[j] gives the spike train, n_t, in trial j, as
                an array of spike counts in each time-bin (= time step)
        xs:     shape = (Ntrial, T); xs[j] is the latent variable time-series x_t in trial j
        rates:  shape = (Ntrial, T); rates[j] is the rate time-series, r_t, in trial j (returned only if get_rate=True)
        """
        # set dt (time-step duration in seconds) such that trial duration is always 1 second, regardless of T.
        dt = 1 / T
        self.dt = dt

        xs = np.empty([Ntrials, T])


        init_dist = self.get_initial_distribution(dt)
        trans_matrix = self.get_transition_matrix(dt)
        states = np.arange(self.K)

        for n in range(Ntrials):
            s = np.empty(T)
            s[0] = np.random.choice(states, p = init_dist)
            for i in range (1,T):
                s[i] = np.random.choice(states, p = trans_matrix[int(s[i-1])])
            xs[n] = s

        rates = self.f_io(xs / (self.K-1)) # shape = (Ntrials, T)

        spikes = np.array([self.emit(rate) for rate in rates]) # shape = (Ntrial, T)

        if get_rate:
            return spikes, xs, rates
        else:
            return spikes, xs
