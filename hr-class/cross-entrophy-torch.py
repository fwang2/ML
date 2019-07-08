import torch
import torch.nn as nn
from torch.autograd import Variable



# There are two differences compare to entrophy numpy version:
#
# ONE:
# Y_pred is logits (not softmax)
# nn.CrossEntrophyLoss() internally apply softmax
#
# TWO:
# Input Y is class itself, not one-hot
#
# here we have 0 - that means class 0 is the label.
# 
#
# the [2.0, 1.0, 0.1] -> predicted correct in the sense that first element (0)
# has the greatest value, thus the desired choice.
#

loss = nn.CrossEntropyLoss()

Y = Variable(torch.LongTensor([0]), requires_grad=False)


Y_pred1 = Variable(torch.Tensor([[2.0, 1.0, 0.1]])) # 1 by 3, predicted correct idx = 0
Y_pred2 = Variable(torch.Tensor([[0.5, 2.0, 0.3]]))  # 1 by 3, predicted wrong idx = 1

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print("pytorch loss1 = ", l1.data)
print("pytorch loss2 = ", l2.data)


