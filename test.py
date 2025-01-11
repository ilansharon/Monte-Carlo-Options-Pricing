from preprocessing import optionData
from methods import iterateGBM, getPayoffs, avgDiscountedPayoff, blackScholes

S0, K, sigma, T, r = optionData("MSFT", "2025-01-17", True)
paths = iterateGBM(S0, r, sigma, T, 10000, 10000)
payoffs = getPayoffs(paths, K, True)
discounted = avgDiscountedPayoff(payoffs, r, T)
BS = blackScholes(S0, K, r, sigma, T, True)

print("avg discounted payoff: ", discounted)
print("black scholes payout", BS)
# for path in paths:
#     plt.plot(time, path)
# plt.show()