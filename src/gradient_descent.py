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

def gradient_descentv1(X, y, 
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

def gradient_descentv2(X, y, alpha=0.01, num_iters=1500):
    """ do gradient deswcent on any feature matrix """
    n, m  = X.shape
    c = np.zeros(num_iters)
    w = np.full((m, 1), 0.0)  # initialize w
    temp = np.full((m, 1), 0.0)
    
    for i in range(num_iters):
        c[i] = compute_cost(X, y, w)
        for j in range(m):
            temp[j,0] = w[j,0] - alpha / (n) * np.sum(
                (np.dot(X, w) - y[:, None]) * X[:, j][:, None])
        w = temp 
    return c, w

def feature_normalize(X):
    """ normalize X, for each features present
    mu, sigma's size is same as number of features
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    
    return mu, sigma, (X - mu.T) / sigma.T
    

def main():

    # ex1data1.txt 
    # column 1: population of a city
    # column 2: profit of a food truck in that city
    # data = np.loadtxt("../datasets/ex1data1.txt", delimiter=',')


    # ex1data1.txt 
    # column 1: population of a city
    # column 2: profit of a food truck in that city
    data = np.loadtxt("../datasets/ex1data2.txt", delimiter=',')



    # add a column to X
    # n is # of samples
    n_iter = 10000
    X = data[:, :2]  # first two columns
    mu, sigma, X = feature_normalize(X)
    
    y = data[:,-1]  # last columns
    n,m = data.shape
    # add 1's as a first column in X
    X = np.hstack((np.ones(n)[:, None], X))
    c, w = gradient_descentv2(X, y, num_iters = n_iter)
    print("Best cost: {} after {} iterations".format(c[-1], n_iter))
    print("Weights: {}".format(w))



if __name__ == '__main__':
    main()
