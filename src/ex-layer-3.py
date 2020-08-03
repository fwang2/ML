import numpy as np
np.random.seed(1)


def relu(x):
    ''' this function sets all negative number to 0 '''
    return (x > 0) * x


def relu2deriv(x):
    ''' Return 1 for x > 0; return 0 otherwise '''
    return x > 0


alpha = 0.2
hidden_size = 4
streetlights = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]])

walk_vs_stop = np.array([[1, 1, 0, 0]]).T

# randomly initialize weight matrix: 0 to 1
weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1

for it in range(60):
    layer_2_error = 0
    for i in range(len(streetlights)):
        # go through each input
        # do forward propergation, which is weighted sum
        layer_0 = streetlights[i:i + 1]

        # REFER TO Step #3
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)

        # REFER TO Step #4
        layer_2_error += np.sum((layer_2 - walk_vs_stop[i:i + 1])**2)

        # REFER TO Step #5
        layer_2_delta = (layer_2 - walk_vs_stop[i:i + 1])

        # NEW, not covered in previous steps
        # this line computes the delta at layer_1 given the delta at layer_2
        # by taking the layer_2_delta and multiplying it by its connecting
        # weights (weights_1_2)
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)

        # REFER TO Step #6, but calculated different, need some revisit
        weight_delta_1_2 = layer_1.T.dot(layer_2_delta)
        weight_delta_0_1 = layer_0.T.dot(layer_1_delta)

        # update weights
        weights_1_2 -= alpha * weight_delta_1_2
        weights_0_1 -= alpha * weight_delta_0_1

    #
    if (it % 10 == 9):
        print(f"Error: {layer_2_error}")