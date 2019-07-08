#
# cross entrophy example
#
# cross entrophy - a way to calculate the difference between two
# distributions
#
# see my notes on "cross entrophy"


import numpy as np

# Note: the label is in one-hot format
Y = np.array([1, 0, 0])


# Note: prediction value is output of softmax
#
# You will see in pytorch CrossEntropyLoss(), this is not the case
# that function does softmax internally.


Y_pred1 = np.array([0.7, 0.2, 0.1])
Y_pred2 = np.array([0.1, 0.3, 0.6])

loss1 = -Y * np.log(Y_pred1)
loss2 = -Y * np.log(Y_pred2)

print("loss 1 on each input = ", loss1)
print("loss 2 on each input = ", loss2)

print("loss 1 (sum) = ", np.sum(loss1))
print("loss 2 (sum) = ", np.sum(loss2))




