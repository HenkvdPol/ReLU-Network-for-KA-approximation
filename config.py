import numpy as np

# parameter values
d = 3
K = 3
p = 1
beta = 0.2
epsilon = 0.1

# The function to estimate
def f(x):
    return np.linalg.norm(x, ord = 1, axis = 1, keepdims= True) # f(x) = |x|_1 = sum(x)

def g(x):
    return np.linalg.norm(x , ord = 2, axis = 1, keepdims= True) # f(x) = ||x||_2

def h(x):
    return np.linalg.norm(x, ord = np.inf, axis =1, keepdims= True) # f(x) = ||x||_inf = max(x)

#Max weights for the NN
x = np.linspace(0, 1, 100).reshape((100,1)) #input to compute the sup norm of

# unless there is a large choice of
def max_weights(f, K, d, beta):
    return 2 * np.maximum(K * d, np.linalg.norm(f(x), ord=np.inf)) ** (K * np.maximum(d, p * beta))