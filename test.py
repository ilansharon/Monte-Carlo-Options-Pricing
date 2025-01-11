from preprocessing import optionData
from methods import iterateGBM, getPayoffs, avgDiscountedPayoff

S0, K, sigma, T, r = optionData("MSFT", "2025-01-17", True)
paths = iterateGBM(S0, r, sigma, T, 1000, 1000)
payoffs = getPayoffs(paths, K, True)
discounted = avgDiscountedPayoff(payoffs, r, T)

print("avg discounted payoff: ", discounted)

# for path in paths:
#     plt.plot(time, path)
# plt.show()