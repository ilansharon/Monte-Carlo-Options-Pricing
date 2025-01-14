import numpy as np
from scipy.stats import norm


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





