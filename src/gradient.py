import numpy as np

def compute_cost(X, y, w):
    nx, mx = X.shape
    ny, my = y.shape
    nw, mw = w.shape    
    assert nx == ny
    assert my == 1
    assert nw == mx and mw == 1
    
    err = X.dot(w) - y
    return 1.0/(2*nx) * np.sum(err**2)

def normal_equation(X, y):
    from numpy.linalg import inv
    return inv(X.T.dot(X)).dot(X.T).dot(y)


def feature_normalize(X):
    """ normalize X, for each features present
    mu, sigma's size is same as number of features
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    
    return mu, sigma, (X - mu.T) / sigma.T

def gradient_descent_w0_w1(X, y, alpha=0.01, num_iters=1500):
    """ return (1) cost history (2) final w """
    nx, mx = X.shape
    ny, my = y.shape
    assert nx == ny
    assert my == 1
    c = np.zeros(num_iters)
    w = np.full((mx, 1), 0.0)  # initialize w
    for i in range(num_iters):
        c[i] = compute_cost(X, y, w)
        err = X.dot(w) - y
        temp0 = w[0] - (alpha / nx) * np.sum( err * X[:,0].reshape(nx,1))
        temp1 = w[1] - (alpha / nx) * np.sum( err * X[:,1].reshape(nx,1))
        w[0] = temp0
        w[1] = temp1
    return c, w


def gradient_descent(X, y, alpha=0.01, num_iters=500):
    """ do gradient deswcent on any feature matrix 
    This version loop over each feature columns"""
    nx, mx = X.shape
    ny, my = y.shape
    assert nx == ny
    assert my == 1

    c = np.zeros(num_iters)
    w = np.full((mx, 1), 0.0)  # initialize w
    temp = np.full((mx, 1), 0.0)
    
    for i in range(num_iters):
        c[i] = compute_cost(X, y, w)
        for j in range(mx):
            err = X.dot(w) - y
            temp[j] = w[j] - (alpha / nx) * np.sum(
                (err * X[:, j].reshape(nx,1)))
        w = temp 
    return c, w


def gradient_descent_vf(X, y, alpha=0.01, num_iters=500):
    """ vector form
    """
    nx, mx = X.shape
    ny, my = y.shape
    assert nx == ny
    assert my == 1

    c = np.zeros(num_iters)
    w = np.full((mx, 1), 0.0)  # initialize w

    for i in range(num_iters):
        err = X.dot(w) - y
        new_X = (err.T).dot(X)
        w =  w -  ((alpha/nx) * new_X.T)
        c[i] = compute_cost(X, y, w)
    
    return c, w 