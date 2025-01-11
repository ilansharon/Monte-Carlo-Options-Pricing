from preprocessing import optionData
import numpy as np
import matplotlib.pyplot as plt

def simulateGBM(S0, r, sigma, T, N):
    dt = T/N

    S = S0
    path = [S]
    for _ in range(N):
        Z = np.random.normal(0,1)                                                   #sample from standard normal
        S = S * np.exp((r - (0.5 * sigma**2)) * dt + (sigma * np.sqrt(dt) * Z))     #discretized solution to SDE for Geometric Brownian Motion through Ito's Lemma
        path.append(S)
    return np.array(path)

def iterateGBM(S0, r, sigma, T, N, numIterations):
    allPaths = np.zeros((numIterations, N+1))  

    for i in range(numIterations):
        allPaths[i, :] = (simulateGBM(S0, r, sigma, T, N))

    return allPaths

def getPayoffs(paths, K, call):
    Sfinal = paths[:, -1]
    if call:
        return np.maximum(Sfinal-K, 0)
    else:
        return np.maximum(K-Sfinal, 0)
    
def avgDiscountedPayoff(payoffs, r, T):
    disc = payoffs * np.exp(-r * T)
    avgDisc = np.average(disc)

    return avgDisc


