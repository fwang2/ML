from .gradient import *



# ex1data1.txt 
# column 1: population of a city
# column 2: profit of a food truck in that city
data = np.loadtxt("../datasets/ex1data1.txt", delimiter=',')

# add a column to X
# n is # of samples

n, m = data.shape
X = data[:,:-1] # all columns except last 
y = data[:,-1] # last columns

# add 1's as a first column in X
y = y.reshape(n,1)
X = np.hstack((np.ones((n, 1)), X.reshape(n, m - 1)))

c, w = gradient_descent(X, y)
print("Gradient Descent:")
print("Cost: {}".format(c[-1]))
print("w0: {:.2f}, w1: {:.2f}".format(w[0, 0], w[1, 0]))


w = normal_equation(X, y)
c = compute_cost(X, y, w)
print("\nNormal equation:")
print("Cost: {}".format(c))
print("w0: {:.2f}, w1: {:.2f}".format(w[0, 0], w[1, 0]))


