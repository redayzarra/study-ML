import math, copy
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

# Define x_train which is the input feature
x_train = np.array([1.0, 2.0])  

# Define y_train which is the target value
y_train = np.array([300.0, 500.0]) 

# Define a function to calculate the cost
def calculate_cost(x: NDArray[np.float64], y: NDArray[np.float64], w: float, b: float) -> float:
    # Get the number of training examples
    m = x.shape[0]
    cost = 0
    
    for index in range(m):
        # Calculate the model's prediction
        f_wb = (w * x[index]) + b
        
        # Calculate the squared error
        squared_error = (f_wb - y[index]) ** 2
        
        # Add the squared error to the total cost
        cost += squared_error
    
    # Calculate the total cost
    total_cost = (1 / (2 * m)) * cost
    return total_cost

# Define a function to compute the gradient
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]    
    
    # Setting up the derivaties of w and b
    dj_dw = 0
    dj_db = 0
    
    for index in range(m):
        # Linear regression model
        f_wb = w * x[index] + b 
        
        # Find the derivate of the cost function J(w, b) for the current index
        dj_dw_i = (f_wb - y[index]) * x[index] 
        dj_db_i = f_wb - y[index] 
        
        # Add the current derivative to the total 
        dj_db += dj_db_i
        dj_dw += dj_dw_i
        
    # Divide the total by the number of training examples 
    dj_dw /= m 
    dj_db /= m 
        
    return dj_dw, dj_db

