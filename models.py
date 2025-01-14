import numpy as np

#simlate Geometric Brownian Motion paths
def simulateGBM(S0, r, sigma, T, numSteps, numPaths):

    dt = T / numSteps  
    Z = np.random.normal(0, 1, size=(numPaths, numSteps))

    paths = np.zeros((numPaths, numSteps+1))
    paths[:, 0] = S0

    for t in range(1, numSteps+1):
        drift = (r - 0.5 * sigma**2) * dt                           #drift term
        diffusion = sigma * np.sqrt(dt) * Z[:, t-1]                 #diffusion term
        paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion)     #GBM solution from Ito's Lemma

    return paths

