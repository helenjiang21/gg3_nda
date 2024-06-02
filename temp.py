from HMM_models import *
from inference import *


def func(model, spike_data):

    #Compute expected states first with fwd-bwd algorithm
    ll = poisson_logpdf(spike_data, model.lambdas)
    init = model.initial_distribution
    trans = model.transition_matrix

    pd = np.empty(model.T)
    pd[0] = np.matmul(ll[0], init)

    for t in range(1, model.T):
        sum = 0
        for i in model.states: #state at time t-1
            for j in model.states: # state at time t
                sum += trans[i][j]*ll[t-1][i]*ll[t][j]
        pd[t] = sum
    return 