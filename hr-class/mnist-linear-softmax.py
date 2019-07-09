# https://github.com/pytorch/examples/blob/master/mnist/main.py
#
# test environment:
#   python 3.7.x and pytorch 1.1
#
# ChangeLogs:
# (1) use linear
# (2) use with torch.no_grad() instead of "violatile = True"
#
#
# Accuracy: 96%


# see Sung Kim's PyTorch Lecture 09: Softmax classifier

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
batch_size = 64

# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)



## Input layer: 784, output is 10
## all the hidden layer is up to you to design

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))

        # no need for activation, as we will use logits for entrophy loss

        return self.l5(x)  


model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):

    # put our model into training mode
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        # we feed the data into the model
        # and 'output' is our prediction

        output = model(data)

        # using 'output' prediction, we calculate the loss

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)

            # make sure we are clear on the input and output:
            # 'data' comes in as batch, therefore, its shape is (64, 1, 28, 28)
            # 'output' is prediction, for each data point, there is a outcome of
            # 10-tuple. Therefore, 'output' shape is (64, 10)

            output = model(data)

            # sum up batch loss
            test_loss += criterion(output, target).item()
  
            #
            # get the index of the max
            #
            # the max() function return two things: first is the max value
            # itself, the second thing is the index of the max value.
            # here, we really just want the index, that is what [1] is for.
            #
            # pred = output.data.max(1, keepdim=True)[1]
            #
            # the following is a bit easier to read.

            # I think, here 'pred' shape is 64 x 1.
            # 
            pred = output.argmax(dim=1, keepdim=True)

            # tensor.view_as(other): view this tensor the as the same size as
            # the other.

            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test()
