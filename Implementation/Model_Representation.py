import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

# Define x_train which is the input feature
x_train = np.array([1.0, 2.0])

# Define y_train which is the output feature (target)
y_train = np.array([300.0, 500.0])

print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# Define m which is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

# Define a specific index
index = 0

# Accessing the training example at specific index
x_i = x_train[index]
y_i = y_train[index]
print(f"(x^({index}), y^({index})) = ({x_i}, {y_i})")

# Plotting the data
plt.scatter(x_train, y_train, marker = "x", c = "r")

# Adding details to the graph
plt.title("Housing prices")
plt.ylabel("Price (in 1000's of dollars)")
plt.xlabel("Size (in 1000 sqft)")
plt.show()

# Define weight (w) and bias (b) for f_wb = (w * x_i) + b
w = 100
b = 100
print(f"w: {w}")
print(f"b: {b}")

# Create a function to calculate the model output
def calculate_model_output(x: NDArray[np.float64], w: int, b: int) -> NDArray[np.float64]:
    """
    ### Computes the prediction of a linear model `f(x) = w * x + b`

    #### Args:
        x (NDArray[np.float64]): Input data array of shape (m,).
        w (float): Weight parameter.
        b (float): Bias parameter.

    #### Returns:
        NDArray[np.float64]: The model predictions, an array of shape (m,).
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for index in range(m):
        f_wb[index] = (w * x[index]) + b
    
    return f_wb

temp_f_wb = calculate_model_output(x_train, w, b)

# Plotting the model's prediction
plt.plot(x_train, temp_f_wb, c='b', label='Our Prediction')

plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Adding details to the graph
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
