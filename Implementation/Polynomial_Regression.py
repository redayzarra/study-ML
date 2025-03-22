import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

# Create target data
x = np.arange(0, 20, 1)
y = 1 + x**2
X = x.reshape(-1, 1)

# Plot the data points
plt.scatter(x, y, marker='x', c='r', label="Actual Value")
plt.title("no feature engineering")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Create polynomial features
X_poly = np.concatenate([X**i for i in range(3)], axis=1)

# Plot the polynomial fit

