import numpy as np
from options import estimate, getPayoffs, avgDiscountedPayoff
from scipy.optimize import minimize


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
    
class Heston(baseModel):
    def __init__(self, S0, r, T, numSteps, numPaths, kappa, theta, v0, rho, sigma_v):
        super().__init__(S0, r, None, T, numSteps, numPaths)
        self.kappa = kappa
        self.theta = theta
        self.v0 = v0
        self.rho = rho
        self.sigma_v = sigma_v

    def simulate(self):
        dt = self.T / self.numSteps

        S = np.zeros((self.numPaths, self.numSteps + 1))
        v = np.zeros((self.numPaths, self.numSteps + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0

        

        for t in range(1, self.numSteps + 1):
            #correlate Brownian motions
            Zs = np.random.normal(0, 1, size = self.numPaths)
            Zv = self.rho * Zs + np.sqrt(1 - self.rho**2) * np.random.normal(0, 1, size=self.numPaths)      

            #Euler-Maruyama discretization of Heston Model

            MRdrift = self.kappa * (self.theta - v[:, t-1]) * dt
            vdiffusion = self.sigma_v * np.sqrt(np.maximum(v[:, t-1], 0) * dt) * Zv
            v[:, t] = v[:, t-1] + MRdrift + vdiffusion
            v[:, t] = np.maximum(v[:, t], 0)

            drift = (self.r - 0.5 * v[:, t-1]) * dt
            diffusion = np.sqrt(np.maximum(v[:, t-1], 0) * dt) * Zs
            S[:, t] = S[:, t-1] * np.exp(drift + diffusion)

        return S
    
    def calibrate(self, marketPrices, strikes, maturities, calls):
        def objective(params):
            self.kappa, self.theta, self.v0, self.rho, self.sigma = params
            errors = []
            simulatedPaths = {}
            for K, T, mp, call in zip(strikes, maturities, marketPrices, calls):
                if T not in simulatedPaths:
                    simulatedPaths[T] = self.simulate()
                paths = simulatedPaths[T]
                finals = estimate(paths, "European")
                payoffs = getPayoffs(finals, K, call)
                hp = avgDiscountedPayoff(payoffs, self.r, T)

                errors.append((hp-mp)**2)

            return np.sum(errors)
        
        init = [2.0, 0.04, 0.04, -0.5, 0.3]
        bounds = [(0.1, 10.0), (0.01, 1.0), (0.01, 1.0), (-1.0, 1.0), (0.01, 1.0)]
        result = minimize(objective, init, bounds=bounds, method="Powell")
        self.kappa, self.theta, self.v0, self.rho, self.sigma = result.x
        if result.success:
            print("calibrated succesfully")
            print(f"kappa={self.kappa}, theta={self.theta}, v0={self.v0}, rho={self.rho}, sigma_v={self.sigma_v}")
        else:
            print("calibration failed to converge")
        
        return result

        

            



