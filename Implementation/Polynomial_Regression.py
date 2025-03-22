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

# Create target data
x = np.arange(0, 20, 1)
y = 1 + x**2

# Engineer features 
X = x**2      #<-- added engineered feature

X = X.reshape(-1, 1)  #X should be a 2-D Matrix

# Plot the data points
plt.scatter(x, y, marker='x', c='r', label="Actual Value")
plt.title("Added x**2 feature")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

