# This is a homework problem,
# not addressed here yet:
#
# assume: y = w2 * x^2 + w1 * x 
#
# now we have two weights: w2 and w1
# we need to figure out (manually) what are the gradient
# for each d(loss)/d(w2), d(loss)/d(w1) and update w1 and w2
# sepaeately.


## REPLACE with the following, same two parameters to drive:
##
## y = x * w + b
##
## we have w and b

## assume w = 2 and b = 1, we will see how this
## the math calculation of grad can be found at:
## https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e



import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# Data Generation
np.random.seed(42)
x = np.random.rand(100, 1)


# this last term is used to add noise.
y = 1 + 2 * x + .1 * np.random.randn(100, 1) 


# Shuffles the indices
idx = np.arange(100)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:80]
# Uses the remaining indices for validation
val_idx = idx[80:]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# Initializes parameters "a" and "b" randomly
np.random.seed(42)
a = np.random.randn(1)
b = np.random.randn(1)

print(a, b)


################################
### training
################################

# Sets learning rate
lr = 1e-1
# Defines number of epochs
n_epochs = 1000

for epoch in range(n_epochs):
    # Computes our model's predicted output
    # This is the "forward" pass, in pytorch's lingo
    yhat = a + b * x_train
    
    # How wrong is our model? That's the error! 
    error = (y_train - yhat)
    # It is a regression, so it computes mean squared error (MSE)
    loss = (error ** 2).mean()
    
    # Computes gradients for both "a" and "b" parameters
    a_grad = -2 * error.mean()
    b_grad = -2 * (x_train * error).mean()
    
    # Updates parameters using gradients and the learning rate
    a = a - lr * a_grad
    b = b - lr * b_grad
    
print(a, b)

# Sanity Check: do we get the same results as our gradient descent?
from sklearn.linear_model import LinearRegression
linr = LinearRegression()
linr.fit(x_train, y_train)
print(linr.intercept_, linr.coef_[0])
