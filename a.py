import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson, binom, expon

# 1. Simple Probability - Rolling a 4 on a 6-sided die
prob_4 = 1 / 6
print(f"Probability of rolling a 4: {prob_4}")

# 2. Normal Distribution - Quality Control
mean, std_dev = 50, 10
samples = np.random.normal(mean, std_dev, 1000)
plt.hist(samples, bins=30, density=True, color='blue', alpha=0.6)
x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 100)
plt.plot(x, norm.pdf(x, mean, std_dev), 'r-', lw=2)
plt.title('Normal Distribution')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()

# 3. Poisson Distribution - Event Occurrences
lambda_param = 5
k = 3
prob_3_events = poisson.pmf(k, lambda_param)
print(f"Probability of 3 events: {prob_3_events}")

# 4. Binomial Distribution - Success in Trials
n, p, k_success = 10, 0.6, 7
prob_7_success = binom.pmf(k_success, n, p)
print(f"Probability of 7 successes: {prob_7_success}")

# 5. Exponential Distribution - Reliability Analysis
exp_samples = np.random.exponential(scale=2, size=1000)
plt.hist(exp_samples, bins=30, density=True, color='green', alpha=0.6)
x_exp = np.linspace(0, 10, 100)
plt.plot(x_exp, expon.pdf(x_exp, scale=2), 'r-', lw=2)
plt.title('Exponential Distribution')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()