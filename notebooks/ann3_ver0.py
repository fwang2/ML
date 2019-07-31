#%%
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length
    return ((y_true - y_pred)** 2).mean()

class ANN3:
    '''
    2 inputs, a hidden layer, a output layer, see above graph for details
    '''
    def __init__(self):
        # self.w1, self.w2, self.w3,self.w4,self.w5,self.w6 = np.random.randn(6)
        # self.b1, self.b2, self.b3 = np.random.randn(3)
        self.w1, self.w2, self.w3,self.w4,self.w5,self.w6 = np.repeat(0,6)
        self.b1, self.b2, self.b3 = np.repeat(0, 3)
        self.losses = []
        self.epochs = []

    def feedforward(self,x):
        # x is 2-element array
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1
    
    def train(self, data, y_trues):
        '''
        data is (nx2) array, n is number of samples, two element is weight and height
        y_trues is the corresponding ground truth for each sample
        '''
        lr = 0.1 # learning rate
        epoches = 1000 # number of times to loop through the entire dataset
        for epoch in range(epoches):
            for x, y_true in zip(data, y_trues):
                
                # -----
                # do feedforward
                # -----
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)
                
                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)
                
                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                
                y_pred = o1
                
                # -----
                # do paritial derivative
                # naming: d_L_d_w1 represents "partial L / partial w1"
                # -----
                
                d_L_d_ypred = -2 * (y_true - y_pred)
                
                # Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)
                
                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)
                
                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)
                
                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)
                
                # ------
                # update weights and bias
                # ------
            
                # Neuron h1
                self.w1 -= lr * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= lr * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= lr * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
                
                # Neuron h2
                self.w3 -= lr * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= lr * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= lr * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
                
                # Neuron o1
                self.w5 -= lr * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= lr * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= lr * d_L_d_ypred * d_ypred_d_b3
                
            # --- calculate total loss at the end of each epoch
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(y_trues, y_preds)
                # print("Epoch %d loss: %.3f" % (epoch, loss))
                self.losses.append(loss)
                self.epochs.append(epoch)

# define dataset

data = np.array([
    [-2, -1], # Alice
    [25,6], # Bob
    [17, 4], # Charlie
    [-15, -6], # Diana
])
y_trues = np.array([
    1, # alice
    0, # bob
    0, # charlie
    1, # diana
])

# Training our NN
network = ANN3()
network.train(data, y_trues)

# Make some predictions
emily = np.array([-7, -3])  # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches

print("Emily: %.3f" % network.feedforward(emily))
print("Frank: %.3f" % network.feedforward(frank))

plt.plot(network.epochs, network.losses)
