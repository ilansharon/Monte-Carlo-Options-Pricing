from preprocessing import optionData, fetchMultiData
from options import getPayoffs, avgDiscountedPayoff, blackScholes, estimate
from models import GBM, Heston
from visualization import graphGBM, graphConvergence
import numpy as np
import matplotlib.pyplot as plt

TICKER = "MSFT"
EXPIRATION = "2025-01-17"
TYPE = "ATM"

S0, K, sigma, T, r = optionData(TICKER, EXPIRATION, True, TYPE)
modelGBM = GBM(S0, r, sigma, T, 3000, 5000)
paths = modelGBM.simulate()
estimationsA = estimate(paths, "Asian")
estimationsE = estimate(paths, "European")
payoffsA = getPayoffs(estimationsA, K, True)
payoffsE = getPayoffs(estimationsE, K, True)
discountedA = avgDiscountedPayoff(payoffsA, r, T)
discountedE = avgDiscountedPayoff(payoffsE, r, T)
BS = blackScholes(S0, K, r, sigma, T, True)

print("avg discounted euro payoff: ", discountedE)
print("avg discounted asian payoff: ", discountedA )
print("black scholes payout", BS)
print("S0, strike, sigma, T, r: ", S0, K, sigma, T, r)
# for path in paths:
#     plt.plot(time, path)
# plt.show()

#graphGBM(paths, TICKER)
#nList = np.arange(1, 25000, 1000)
#numSteps = 5000
#graphConvergence(S0, K, r, sigma, T, numSteps, nList, BS, True)

strikes, prices, maturities, callFlags = fetchMultiData(TICKER, 20)
# print(strikes, prices, maturities, callFlags)
# print(len(strikes), len(prices), len(maturities), len(callFlags))

heston = Heston(S0, r, T, 300, 300, 2.0, 0.04, 0.04, -0.5, 0.3)
heston.calibrate(prices, strikes, maturities, callFlags)
paths = heston.simulate()
times = np.arange(0, 300)
for path in paths:
    plt.plot(times, paths)

plt.show()