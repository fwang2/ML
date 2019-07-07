import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc =  nn.Linear(in_features=28*28, out_features=10)
        
    def forward(self, t):
        t = t.reshape(-1, 28*28)
        t = F.softmax(self.fc(t))
        
        return t
    
    

# download training set
# alternative: FashionMNIST

train_set = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)


network = Network()

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
#optimizer = optim.Adam(network.parameters(), lr=0.01)
optimizer = optim.SGD(network.parameters(), lr=0.5)

total_loss = 0
total_correct = 0
total = len(train_set)

# one epoch only
for batch in train_loader:

    images , labels = batch

    preds = network(images)
    loss = F.cross_entropy(preds, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step() # update weights

    total_loss += loss.item()
    total_correct += get_num_correct(preds, labels)

print("total correct: {:.2f}, total {}, accuracy: {:.4f}".format(
    total_correct, total, total_correct/total))