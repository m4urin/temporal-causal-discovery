import math

import torch

var_mean = 0.0
mean_var = 0.5

X_mu = math.sqrt(var_mean) * torch.randn(10000)  # std
X_var = mean_var * 2 * torch.rand(len(X_mu))  #
X_std = torch.sqrt(X_var)
#print(X_mu)
#print(X_std)

sample_result = torch.empty(1000, len(X_mu))
for i in range(len(sample_result)):
    sample_result[i] = torch.normal(mean=X_mu, std=X_std)
sample_result = sample_result.var(dim=-1)
print("Sampling")
print("  var:", sample_result.mean())
print("  std:", sample_result.mean().sqrt())
#print("  uncertainty:", sample_result.std())

#weights = 1.0 / X_var
#weighted_mean = ((X_mu ** 2) * weights).sum() / weights.sum()

#print("Weighted mean")
#print("  mean:", weighted_mean)

#print("Weighted_variance")
#weighted_mean = (X_mu * X_var) / X_mu.sum()
#weighted_variance = (X_mu * (X_var - weighted_mean)**2).sum()  # / weights.sum()
#weighted_std = torch.sqrt(weighted_variance)
#print("  weighted_std:", weighted_std)


print("Total Variance")
total_variance = torch.mean(X_var) + torch.var(X_mu)
#total_std = torch.sqrt(total_variance)
print("  var:", total_variance)
print("  std:", total_variance.sqrt())

print("True")
print("  var:", mean_var + var_mean)
print("  std:", math.sqrt(mean_var + var_mean))

