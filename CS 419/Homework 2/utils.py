import numpy as np


def square_hinge_loss(targets, outputs):
    # Write thee square hinge loss here
    return np.sum(np.square(np.max(0, 1 - targets * outputs)))


def logistic_loss(targets, outputs):
    # Write thee logistic loss loss here
    g = 1 / (1 + np.exp(-outputs))
    return np.sum(np.log(1 + np.exp(-targets * g)))


def perceptron_loss(targets, outputs):
    # Write thee perceptron loss here
    return np.sum(np.max(0, - targets * outputs))


def L2_regulariser(weights):
    # Write the L2 loss here
    weights[-1] = 0
    return np.sum(np.square(weights))


def L4_regulariser(weights):
    # Write the L4 loss here
    weights[-1] = 0
    return np.sum(np.power(weights, 4))


def square_hinge_grad(weights, inputs, targets, outputs):
    # Write thee square hinge loss gradient here
    return 2 * np.max(0, 1 - targets * outputs) * np.matmul(inputs.T, outputs)


def logistic_grad(weights, inputs, targets, outputs):
    # Write thee logistic loss loss gradient here
    # print(outputs)
    loss = np.sum(inputs, 1) * (np.exp(-targets * outputs) /
                                (1 + np.exp(-targets * outputs)))
    # print(loss.shape)
    return np.sum(loss)


def perceptron_grad(weights, inputs, targets, outputs):
    # Write thee perceptron loss gradient here
    d = (outputs * targets < 0)
    return np.matmul(inputs.T, outputs)


def L2_grad(weights):
    # Write the L2 loss gradient here
    weights[0] = 0
    return 2 * weights


def L4_grad(weights):
    # Write the L4 loss gradient here
    weights[0] = 0
    return 4 * np.power(weights, 3)


loss_functions = {"square_hinge_loss": square_hinge_loss,
                  "logistic_loss": logistic_loss,
                  "perceptron_loss": perceptron_loss}

loss_grad_functions = {"square_hinge_loss": square_hinge_grad,
                       "logistic_loss": logistic_grad,
                       "perceptron_loss": perceptron_grad}

regularizer_functions = {"L2": L2_regulariser,
                         "L4": L4_regulariser}

regularizer_grad_functions = {"L2": L2_grad,
                              "L4": L4_grad}
