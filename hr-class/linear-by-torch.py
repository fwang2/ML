#
# this code is tested on python 3.7 and pytorch 1.1
#
# input is 1, 2, 3, output is 2, 4, 6
# so single weight to learn is 2.
# we can see how close we get.


import torch
import torch.nn as nn
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # we have one x as input
        # one output as y, therefore 1, 1
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        """ take in a x, output a y
        note that we call self.linear(), internally linear
        has forward() defined and called """

        y_pred = self.linear(x)
        return y_pred


model = Model()

# if you change reduction = 'mean', you will observe results
# become unstable: meaning sometimes it is good, sometime it is bad

criterion = nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training loop

for epoch in range(500):

    # forward pass. Noted here we had written for loop before
    # to iterate on each data point. With pytorch, we don't have to
    # do that anymore.

    y_pred = model(x_data)


    # compute and print loss
    loss = criterion(y_pred, y_data)
    print("epoch {}, loss = {:.2f}".format(epoch, loss.data))

    # zero grad, backward pass and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# let's see how well it predicts

input = Variable(torch.tensor([[4.0]]))
print("Predict (input 4):", model.forward(input).data[0][0])
