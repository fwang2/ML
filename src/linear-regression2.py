# linear regression gradient descent
# datasets/ex1data1.txt
#
# Best cost: 4.47697137598 after 10000 iterations
# Weights: [[-3.89578082] [1.19303364]]
#

from gradient import * 




# ex1data1.txt 
# column 1: population of a city
# column 2: profit of a food truck in that city
data = np.loadtxt("../datasets/ex1data2.txt", delimiter=',')
n, m = data.shape

n_iter = 10000

# add a column to X
X = data[:,:-1] # all columns except last 
y = data[:, -1]  # last columns
mu, sigma, X = feature_normalize(X)

y = y.reshape(n,1)
X = np.hstack((np.ones((n, 1)), X.reshape(n, m - 1)))


c, w = gradient_descent(X, y, num_iters=500)
print("Gradient Descent:")
print("Cost: {:,.2f} after {} iterations".format(c[-1], n_iter))
print("Weights: {}".format(w.T))

# Estimate the price of a 1650 sq-ft, 3 br house
temp = np.array([1.0, 1650.0, 3.0])
temp[1] = (temp[1] - mu[0])/sigma[0]
temp[2] = (temp[2] - mu[1])/sigma[1];
price = temp.reshape(1,3).dot(w)
print("Predicted price for 1650 sq ft, 3 bed rooms: {}".format(price))

X = data[:,:-1] # all columns except last 
y = data[:, -1]  # last columns

y = y.reshape(n,1)
X = np.hstack((np.ones((n, 1)), X.reshape(n, m - 1)))

w = normal_equation(X, y)
print("\nNormal equation:")
c = compute_cost(X, y, w)
print("Cost: {:,.2f}".format(c))
print("Weights: {}".format(w.T))


