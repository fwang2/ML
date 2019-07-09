import torch
import torch.nn as nn
from torch.autograd import Variable

# One cell RNN, input 4, output 2, sequence 5
# https://www.youtube.com/watch?v=ogZi5oIo4fI&list=PLlMkM4tgfjnJ3I-dbhO9JTw7gNty6o_2m&index=12


cell = nn.RNN(input_size=4, hidden_size=2, batch_first=True)

# One hot encoding
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

inputs = Variable(torch.Tensor([[h, e, l, l, o]])) # shape (1, 5, 4)
hidden = Variable(torch.randn(1, 1, 2)) # shape (1, 5, 2)
out, hidden = cell(inputs, hidden)
print(out.data)


## batch input

##
inputs = Variable(
    torch.Tensor([
        [h, e, l, l, o],
        [e, o, l, l, l],
        [l, l, e, e, l]
]))
print("Input size:", inputs.size())
# 3 batch size, 5 sequence_length, 4, one-hot size

hidden = Variable(torch.randn(1, 3, 2))  # shape (3, 5, 2)

out, hidden = cell(inputs, hidden)
print("out size:", out.size())


