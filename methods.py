from preprocessing import optionData
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#simlate one Geometric Brownian Motion (GBM) path
def simulateGBM(S0, r, sigma, T, N):
    dt = T/N

    S = S0
    path = [S]
    for _ in range(N):
        Z = np.random.normal(0,1)                                                   #sample from standard normal
        S = S * np.exp((r - (0.5 * sigma**2)) * dt + (sigma * np.sqrt(dt) * Z))     #discretized solution to SDE for Geometric Brownian Motion through Ito's Lemma
        path.append(S)
    return np.array(path)

#iterate GBM over Monte Carlo sample size
def iterateGBM(S0, r, sigma, T, N, numIterations):
    allPaths = np.zeros((numIterations, N+1))  

    for i in range(numIterations):
        allPaths[i, :] = (simulateGBM(S0, r, sigma, T, N))

    return allPaths

#get call or put payoff from final prices
def getPayoffs(paths, K, call):
    Sfinal = paths[:, -1]
    if call:
        return np.maximum(Sfinal-K, 0)          #call payoff
    else:
        return np.maximum(K-Sfinal, 0)          #put payoff
    
#discount payoff based on risk-free rate and calculate avg
def avgDiscountedPayoff(payoffs, r, T):
    disc = payoffs * np.exp(-r * T)
    avgDisc = np.average(disc)

    return avgDisc

#price with black scholes for benchmarking
def blackScholes(S0, K, r, sigma, T , call):

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call:
        return S0 * norm.cdf(d1) - (K * np.exp(-r * T) * norm.cdf(d2))
    else:
        return (K * np.exp(-r * T) * norm.cdf(-d2)) - (S0 * norm.cdf(-d1))


