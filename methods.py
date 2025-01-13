from preprocessing import optionData
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#simlate one Geometric Brownian Motion (GBM) path
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

def estimate(paths, type):
    if type == "European":                          #need final price for european option
        return paths[:, -1]
    elif type == "Asian":                           #need arithmetic mean for asian arithmetic option
        return np.mean(paths, axis=1)

#get call or put payoff from final prices for european options
def getPayoffs(estimations, K, call):
    if call:
        return np.maximum(estimations-K, 0)          #call payoff
    else:
        return np.maximum(K-estimations, 0)          #put payoff
    

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
    max_paths = max(nList)
    allPaths = simulateGBM(S0, r, sigma, T, numSteps, max_paths)            #calculate max # of paths at first, then sample from it
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



