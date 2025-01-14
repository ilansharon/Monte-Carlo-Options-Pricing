from preprocessing import optionData
from options import getPayoffs, avgDiscountedPayoff, blackScholes, estimate
from models import simulateGBM
from visualization import graphGBM, graphConvergence
import numpy as np

TICKER = "MSFT"
EXPIRATION = "2025-01-17"

S0, K, sigma, T, r = optionData(TICKER, EXPIRATION, True)
paths = simulateGBM(S0, r, sigma, T, 3000, 5000)
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
# for path in paths:
#     plt.plot(time, path)
# plt.show()

graphGBM(paths, TICKER)
nList = np.arange(1, 25000, 1000)
numSteps = 5000
graphConvergence(S0, K, r, sigma, T, numSteps, nList, BS, True)