import numpy as np
import matplotlib.pyplot as plt
from models import GBM
from options import estimate, getPayoffs, avgDiscountedPayoff

#visualization:

"""what to visualize: 
1) MC Paths & black scholes endpoint on same graph
2) 
"""

#graph a simulated GBM + MC
def graphGBM(paths, ticker):
    time = np.arange(0, len(paths[1]))
    for path in paths:
        plt.plot(time, path)
    plt.title("Asset Paths for " + ticker + " according to Geometric Brownian Motion")
    plt.show()


#demonstrate how Monte Carlo converges to Black-Scholes for European Options
def graphConvergence(S0, K, r, sigma, T, numSteps, nList, fairPrice, call):
    maxPaths = max(nList)
    modelGBM = GBM(S0, r, sigma, T, numSteps, maxPaths)
    allPaths = modelGBM.simulate()            #calculate max # of paths at first, then sample from it
    finals = estimate(allPaths, "European")

    estimates = []
    for n in nList:                                     #use first n paths
        np.random.shuffle(allPaths)
        payoffs = getPayoffs(finals[:n], K, call)
        discounted = avgDiscountedPayoff(payoffs, r, T)
        estimates.append(discounted)                                    #interested only in the average discounted payoff here
    
    plt.plot(nList, estimates, label="Monte Carlo Estimate")
    plt.axhline(y=fairPrice, color='r', linestyle='--', label="Black–Scholes Price")
    plt.xlabel("Number of Paths")
    plt.ylabel("Option Price")
    plt.title("Monte Carlo Convergence to Black–Scholes")
    plt.legend()
    plt.show()