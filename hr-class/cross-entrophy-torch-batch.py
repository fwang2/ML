
# this small code snippet demos
# pytorch entrophy supports batch loss calculation as well.
#

import torch
import torch.nn as nn
from torch.autograd import Variable

loss = nn.CrossEntropyLoss()

Y = Variable(torch.LongTensor([2, 0, 1]), requires_grad=False)

Y_pred1 = Variable(torch.Tensor(
    [[0.1, 0.2, 0.9], # predict2 2
    [1.1, 0.1, 0.2],  # predicts 0
    [0.2, 2.1, 0.1]]  # predicts 1
))

Y_pred2 = Variable(torch.Tensor(
    [[0.8, 0.2, 0.3], # predicts 0
    [0.2, 0.3, 0.5],  # predicts 2
    [0.2, 0.2, 0.5]]  # predicts 2
))

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print("Batch Loss1 = ", l1.data)
print("Batch loss2 = ", l2.data)
