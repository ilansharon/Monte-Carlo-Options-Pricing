import numpy as np


class baseModel:
    def __init__(self, S0, r, sigma, T, numSteps, numPaths):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.numSteps = numSteps
        self.numPaths = numPaths

    def simulate(self):     #placeholder
        raise NotImplementedError("must override simulate")
    

class GBM(baseModel):
    def simulate(self):

        dt = self.T / self.numSteps  
        Z = np.random.normal(0, 1, size=(self.numPaths, self.numSteps))

        paths = np.zeros((self.numPaths, self.numSteps + 1))
        paths[:, 0] = self.S0

        for t in range(1, self.numSteps + 1):
            drift = (self.r - 0.5 * self.sigma**2) * dt                         #drift term
            diffusion = self.sigma * np.sqrt(dt) * Z[:, t-1]                    #diffusion term
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion)             #GBM solution from Ito's Lemma

        return paths



