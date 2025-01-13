from preprocessing import optionData
from methods import simulateGBM, getPayoffs, avgDiscountedPayoff, blackScholes, graphGBM, graphConvergence

TICKER = "MSFT"
EXPIRATION = "2025-01-17"

S0, K, sigma, T, r = optionData(TICKER, EXPIRATION, True)
paths = simulateGBM(S0, r, sigma, T, 100, 100)
payoffs = getPayoffs(paths, K, True)
discounted = avgDiscountedPayoff(payoffs, r, T)
BS = blackScholes(S0, K, r, sigma, T, True)

print("avg discounted payoff: ", discounted)
print("black scholes payout", BS)
# for path in paths:
#     plt.plot(time, path)
# plt.show()

#graphGBM(paths, TICKER)
nList = [10, 20, 30, 40, 50, 100, 150, 200, 400, 800, 1000, 2000, 3000, 5000, 10000, 100000]
graphConvergence(S0, K, r, sigma, T, nList, BS, True)