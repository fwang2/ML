##
## example from:
## https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e


import numpy as np 
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot

from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Our data was in Numpy arrays, but we need to transform them into PyTorch's Tensors
# and then we send them to the chosen device
x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

# Here we can see the difference - notice that .type() is more useful
# since it also tells us WHERE the tensor is (device)
print(type(x_train), type(x_train_tensor), x_train_tensor.type())

lr = 1e-1
n_epochs = 1000

torch.manual_seed(42)
a = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)
b = torch.randn(1, requires_grad=True, dtype=torch.float, device=device)

def try1():

    for epoch in range(n_epochs):
        yhat = a + b * x_train_tensor
        error = y_train_tensor - yhat
        loss = (error ** 2).mean()

        # No more manual computation of gradients! 
        # a_grad = -2 * error.mean()
        # b_grad = -2 * (x_tensor * error).mean()
        
        # We just tell PyTorch to work its way BACKWARDS from the specified loss!
        loss.backward()
        # Let's check the computed gradients...
        print(a.grad)
        print(b.grad)
        
        # What about UPDATING the parameters? Not so fast...
        
        # FIRST ATTEMPT
        # AttributeError: 'NoneType' object has no attribute 'zero_'
        # a = a - lr * a.grad
        # b = b - lr * b.grad
        # print(a)

        # SECOND ATTEMPT
        # RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.
        # a -= lr * a.grad
        # b -= lr * b.grad        
        
        # THIRD ATTEMPT
        # We need to use NO_GRAD to keep the update out of the gradient computation
        # Why is that? It boils down to the DYNAMIC GRAPH that PyTorch uses...
        with torch.no_grad():
            a -= lr * a.grad
            b -= lr * b.grad
        
        # PyTorch is "clingy" to its computed gradients, we need to tell it to let it go...
        a.grad.zero_()
        b.grad.zero_()
        
    print(a, b)


def try2():
    """
    We use a optimizer to update parameters (using gradient as in try1)
    and in this case, we don't have to call zero_ to reset the gradient 
    optimizer provide .zero_grad() will do the job.
    """

    # Defines a SGD optimizer to update the parameters
    optimizer = optim.SGD([a, b], lr=lr)

    for epoch in range(n_epochs):
        yhat = a + b * x_train_tensor
        error = y_train_tensor - yhat
        loss = (error ** 2).mean()

        loss.backward()    
        
        # No more manual update!
        # with torch.no_grad():
        #     a -= lr * a.grad
        #     b -= lr * b.grad
        optimizer.step()
        
        # No more telling PyTorch to let gradients go!
        # a.grad.zero_()
        # b.grad.zero_()
        optimizer.zero_grad()
        
    print(a, b)

def try3():
    """ now we make use of loss function that pytorch provided.
    """

    # Defines a MSE loss function
    loss_fn = nn.MSELoss(reduction='mean')

    # note here we have to tell optimizer
    # which 'parameter' we want it to update.
    # in the follow on class definition, we can further
    # simplify this step by convert a and b to nn.Parameters
    # and then feeding optimizer can be as simple as
    #     nn.Parameters()
    #

    optimizer = optim.SGD([a, b], lr=lr)

    for epoch in range(n_epochs):
        yhat = a + b * x_train_tensor
        
        # No more manual loss!
        # error = y_tensor - yhat
        # loss = (error ** 2).mean()
        loss = loss_fn(y_train_tensor, yhat)

        loss.backward()    
        optimizer.step()
        optimizer.zero_grad()
        
    print(a, b)

#try1()
#try2()
#try3()

class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        
    def forward(self, x):
        # Computes the outputs / predictions
        return self.a + self.b * x


def try4():
    # Now we can create a model and send it at once to the device
    model = ManualLinearRegression().to(device)
    # We can also inspect its parameters using its state_dict
    print(model.state_dict())

    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        # What is this?!?
        model.train()

        # No more manual prediction!
        # yhat = a + b * x_tensor
        yhat = model(x_train_tensor)
        
        loss = loss_fn(y_train_tensor, yhat)
        loss.backward()    
        optimizer.step()
        optimizer.zero_grad()
        
    print(model.state_dict())

#try4()

##
## Here, we wrap torch's own Linear model instead of writing our own
## 
class LayerLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # Instead of our custom parameters, we use a Linear layer with single input and single output
        self.linear = nn.Linear(1, 1)
                
    def forward(self, x):
        # Now it only takes a call to the layer to make predictions
        return self.linear(x)

### Alternatively
### we can simplify the model building by:
###
### model = nn.Sequential(nn.Linear(1, 1)).to(device)
###



### THE following code
### is just a refactoring to make the flow more compact.
###


def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x)
        # Computes loss
        loss = loss_fn(y, yhat)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()
    
    # Returns the function that will be called inside the train loop
    return train_step


###
### does it make it better? or more obscure?
###
def try5(model, loss_fn, optimizer):
    # Creates the train_step function for our model, loss function and optimizer
    train_step = make_train_step(model, loss_fn, optimizer)
    losses = []

    # For each epoch...
    for epoch in range(n_epochs):
        # Performs one train step and returns the corresponding loss
        loss = train_step(x_train_tensor, y_train_tensor)
        losses.append(loss)
        
    # Checks model's parameters
    print(model.state_dict())



from torch.utils.data import Dataset, TensorDataset

class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

# Wait, is this a CPU tensor now? Why? Where is .to(device)?
x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

train_data = CustomDataset(x_train_tensor, y_train_tensor)
print(train_data[0])

train_data = TensorDataset(x_train_tensor, y_train_tensor)
print(train_data[0])

### Once we have the dataset, we can:

def try6(model, loss_fn, optimizer):

    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)

    losses = []
    train_step = make_train_step(model, loss_fn, optimizer)

    for epoch in range(n_epochs):
        for x_batch, y_batch in train_loader:
            # the dataset "lives" in the CPU, so do our mini-batches
            # therefore, we need to send those mini-batches to the
            # device where the model "lives"
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            loss = train_step(x_batch, y_batch)
            losses.append(loss)
            
    print(model.state_dict())


### final, with evaluation

def try7():


    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    # Builds dataset with ALL data
    dataset = TensorDataset(x_tensor, y_tensor)
    # Splits randomly into train and validation datasets
    train_dataset, val_dataset = random_split(dataset, [80, 20])
    # Builds a loader for each dataset to perform mini-batch gradient descent
    train_loader = DataLoader(dataset=train_dataset, batch_size=16)
    val_loader = DataLoader(dataset=val_dataset, batch_size=20)

    # Builds a simple sequential model
    model = nn.Sequential(nn.Linear(1, 1)).to(device)
    print(model.state_dict())

    # Sets hyper-parameters
    lr = 1e-1
    n_epochs = 150

    # Defines loss function and optimizer
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=lr)

    losses = []
    val_losses = []
    # Creates function to perform train step from model, loss and optimizer
    train_step = make_train_step(model, loss_fn, optimizer)

    # Training loop
    for epoch in range(n_epochs):
        # Uses loader to fetch one mini-batch for training
        for x_batch, y_batch in train_loader:
            # NOW, sends the mini-batch data to the device
            # so it matches location of the MODEL
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # One stpe of training
            loss = train_step(x_batch, y_batch)
            losses.append(loss)
            
        # After finishing training steps for all mini-batches,
        # it is time for evaluation!
            
        # We tell PyTorch to NOT use autograd...
        # Do you remember why?
        with torch.no_grad():
            # Uses loader to fetch one mini-batch for validation
            for x_val, y_val in val_loader:
                # Again, sends data to same device as model
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                
                # What is that?!
                model.eval()
                # Makes predictions
                yhat = model(x_val)
                # Computes validation loss
                val_loss = loss_fn(y_val, yhat)
                val_losses.append(val_loss.item())

    print(model.state_dict())
    print(np.mean(losses))
    print(np.mean(val_losses))

try7()

