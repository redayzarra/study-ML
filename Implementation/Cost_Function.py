import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

# Define x_train which is the input feature
x_train = np.array([1.0, 2.0])

# Define y_train which is the output (target)

# Define a function to calculate the cost
def calculate_cost(x: NDArray[np.float64], y: NDArray[np.float64], w: float, b: float) -> float:
    """
    ### Computes the cost function for linear regression.
    
    #### Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    #### Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # Get the number of training examples
    m = x.shape[0]
    
    cost_sum = 0
    for index in range(m):
        # Get the model's prediction using the linear formula
        f_wb = (w * x[index]) + b
        
        # Calculate the squared error and add to cost sum 
        cost = (f_wb - y[index]) ** 2
        cost_sum += cost
    
    # Use standard cost function to calculate the total cost
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost