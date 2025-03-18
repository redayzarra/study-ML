import numpy as np
import time

# NumPy routines which allocate memory and fill arrays with zeroes based on input value or shape
a = np.zeros(4);                print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}\n")
a = np.zeros((4, 2));           print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}\n")

# NumPy routines which allocate memory and fill arrays with random values based on shape
a = np.random.random_sample(4); print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}\n")

# NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
a = np.arange(4.);              print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}\n")
a = np.random.rand(4);          print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}\n")

# NumPy routines which allocate memory and fill with user specified values
a = np.array([5,4,3,2]);  print(f"np.array([5,4,3,2]):  a = {a},    a shape = {a.shape}, a data type = {a.dtype}\n")
a = np.array([5.,4,3,2]); print(f"np.array([5.,4,3,2]): a = {a},    a shape = {a.shape}, a data type = {a.dtype}\n")

