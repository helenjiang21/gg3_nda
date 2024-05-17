import models
import numpy as np
from scipy.optimize import minimize

N_realizations = 10000
T = 100

m = 29
r = 7.6
beta = 1.12
sigma = 0.56

def f_step_estimate(m, r):
    step_model = models.StepModel(m=m, r=r, x0=0.2, Rh=75)
    spikes, xs, rates = step_model.simulate(N_realizations, T=T)
    avg_rate = np.mean(spikes, axis=0)
    return avg_rate

def f_ramp_estimate(beta, sigma):
    ramp_model = models.RampModel(beta, sigma, x0=0.2, Rh=75)
    spikes, xs, rates = ramp_model.simulate(N_realizations, T=T)
    avg_rate = np.mean(spikes, axis=0)
    return avg_rate

def SE(x_0):
    m,r,beta,sigma = x_0
    print(f"m = {m}, r = {r}, beta = {beta}, sigma = {sigma}")
    se = np.sum((f_ramp_estimate(beta, sigma) - f_step_estimate(m, r))**2)
    print(f"SE = {se}")
    return se


x0 = [m, r, beta, sigma] #initial guess
result = minimize(SE, x0, method = 'Nelder-Mead', tol=1e-5)
x_optimal = result.x
print(f"Optimal values: {x_optimal}")