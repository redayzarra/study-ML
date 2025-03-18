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

# Indexing operations on 1-D vectors
a = np.arange(10)
print(a)

# Accessing an element returns a scalar
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

# Accessing the last element, negative indexes count from the end
print(f"a[-1] = {a[-1]}")

# Indices must be within the range of the vector or they will produce an error
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)
    
# Slicing operations on 1-D vectors
a = np.arange(10)
print(f"a         = {a}")

# Access 5 consecutive elements (start:stop:step)
c = a[2:7:1];     print("a[2:7:1] = ", c)

# Access 3 elements separated by two 
c = a[2:7:2];     print("a[2:7:2] = ", c)

# Access all elements index 3 and above
c = a[3:];        print("a[3:]    = ", c)

# Access all elements below index 3
c = a[:3];        print("a[:3]    = ", c)

# Access all elements
c = a[:];         print("a[:]     = ", c)

