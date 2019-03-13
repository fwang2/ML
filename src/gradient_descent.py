# linear regression gradient descent
# datasets/ex1data1.txt
#
# Best cost: 4.47697137598 after 10000 iterations
# Weights: [[-3.89578082] [1.19303364]]
#

import numpy as np
 
def compute_cost(X, y, w):
    """
    X - feature matrix
    y - target value
    w - weights
    """
    n = len(X)
    return 1./(2*n) *  np.sum(( np.dot(X, w) - y[:,None])**2)

def gradient_descent(X, y, 
                     w = np.full((2,1),0.0), alpha=0.01, num_iters=1500):
    """ return (1) cost history (2) final w """
    n = len(X)
    c = np.zeros(num_iters)
    for i in range(num_iters):       
        c[i] = compute_cost(X, y, w)
        temp0 = w[0,0] - alpha /(n) * np.sum( (np.dot(X,w) - y[:,None]) * X[:,0][:,None])
        temp1 = w[1,0] - alpha /(n) * np.sum( (np.dot(X,w) - y[:,None]) * X[:,1][:,None])
        w[0,0] = temp0
        w[1,0] = temp1

    return c, w

def main():

    # ex1data1.txt 
    # column 1: population of a city
    # column 2: profit of a food truck in that city
    data = np.loadtxt("../datasets/ex1data1.txt", delimiter=',')


    # add a column to X
    # n is # of samples
    n_iter = 10000
    X = data[:,0]
    y = data[:,1]
    n,m = data.shape
    # add 1's as a first column in X
    X = np.hstack((np.ones(n)[:, None], X[:, None]))
    c, w = gradient_descent(X, y, num_iters = n_iter)
    print("Best cost: {} after {} iterations".format(c[-1], n_iter))
    print("Weights: {}".format(w))

if __name__ == '__main__':
    main()
