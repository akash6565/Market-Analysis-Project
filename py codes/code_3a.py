import numpy as np
from scipy.stats import gumbel_r
import matplotlib.pyplot as plt

# Set parameters
N = 3
J = 2
sigma = 1

# Generate matrix of deterministic valuations
V = np.array([[1, 1],
              [1, 2],
              [1, 3]])

def choice_prob_logit(V, sigma=1):
    exp_V = np.exp(V / sigma)
    denom = np.sum(exp_V, axis=1)
    P = exp_V / denom[:, np.newaxis]
    return P

P = choice_prob_logit(V)
print("Choice probabilities:")
print(P)

# Compute expected market shares
expected_market_shares = np.mean(P, axis=0)
print("Expected market shares:")
print(expected_market_shares)

# Simulate choice using extreme value distribution
def sim_choice_logit(V, sigma=1):
    eps = np.random.gumbel(loc=0, scale=sigma, size=(N, J))
    U = V + eps
    choices = np.argmax(U, axis=1) + 1  # Add 1 to match R's indexing
    return choices

choices = sim_choice_logit(V)
print("Simulated choices:")
print(choices)

# Compare simulated market shares with expected market shares
sim_market_shares = np.bincount(choices) / N
print("Simulated market shares:")
print(sim_market_shares)

# Plot how choice probability changes with V
V1 = np.arange(-10, 11)
V2 = np.zeros_like(V1)
V = np.column_stack((V1, V2))
P = choice_prob_logit(V)

plt.plot(V[:, 0], P[:, 0])
plt.xlabel('V1')
plt.ylabel('Choice Probability')
plt.title('Effect of V1 on Choice Probability')
plt.grid(True)
plt.show()
