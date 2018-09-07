import argparse
from scipy.optimize import minimize

import utils
import numpy as np


class DataLoader(object):
    # this class has a standard iterator declared
    # __len__ returns the number of batches (size of the object)
    # __get_item__ handles integer based indexing of the object
    def __init__(self, data_file, batch_size):
        with open(data_file, 'r') as df:
            data = df.readlines()

        data = data[1:]
        data = data[:(len(data) // batch_size) * batch_size]
        np.random.shuffle(data)
        data = np.array([[float(col) for col in row.split(',')]
                         for row in data])
        input_data, targets = data[:, :-1], data[:, -1]
        input_data = np.hstack([input_data, np.ones(
            (len(input_data), 1), dtype=np.float32)])

        self.num_features = input_data.shape[1]
        self.current_batch_index = 0
        self.input_batches = np.split(
            input_data, len(input_data) // batch_size)
        self.target_batches = np.split(targets, len(targets) // batch_size)
        print('TTTTTTTTTTTTTTTTTTTTTTT',targets)
    def __len__(self):
        return len(self.input_batches)

    def __getitem__(self, i):

        batch_input_data = self.input_batches[i]
        batch_targets = self.target_batches[i]
        return batch_input_data, batch_targets

def classify(inputs, weights):
    # this functions returns w^Tx . The output  is batch_size*1
    return np.dot(inputs, np.reshape(weights, (np.size(weights), 1)).reshape((-1,)))


def get_objective_function(trainx, trainy, loss_type, regularizer_type, loss_weight):
    # this function calculates the loss for a current batch
    loss_function = utils.loss_functions[loss_type]
    if regularizer_type != None:

        regularizer_function = utils.regularizer_functions[regularizer_type]

    def objective_function(weights):
        loss = 0

        inputs, targets = trainx, trainy
        outputs = classify(inputs, weights)
        loss += loss_weight * loss_function(targets, outputs)
        if regularizer_type != None:
            # regulariser function is called from utils.py
            loss += regularizer_function(weights)
        return loss
    return objective_function


def get_gradient_function(trainx, trainy, loss_type, regularizer_type, loss_weight):
    # This is a way to declare function inside a function
    # The get_gradient_function receives the train data from the current batch
    # and all other parameters on which the loss function and gradient depend
    # like C,regulariser_type and loss function
    loss_grad_function = utils.loss_grad_functions[loss_type]
    if regularizer_type != None:
        regularizer_grad_function = utils.regularizer_grad_functions[regularizer_type]
    # gradient function is called from scipy.optimise.minimise()
    # the only paramter its can send is weights
    # hence there was a need to pass the current batch through get_objective_function

    def gradient_function(weights):

        gradient = np.zeros(len(weights), dtype=np.float32)
        X = trainx
        Y = trainy
        outputs = classify(X, weights)
        # loss_grad_function is called from utils.py
        gradient = loss_weight * \
            loss_grad_function(weights, X, Y, outputs) / len(trainx)
        if regularizer_type != None:
            # regulariser grad function is called from utils.py
            gradient += regularizer_grad_function(weights)
        return gradient
    return gradient_function


def train(data_loader, loss_type, regularizer_type, loss_weight):
    initial_model_parameters = np.random.random((data_loader.num_features))

    num_epochs = 1000
    for i in range(num_epochs):
        loss = 0
        if(i == 0):
            start_parameters = initial_model_parameters
        for j in range(len(data_loader)):
            trainx, trainy = data_loader[j]
            objective_function = get_objective_function(trainx, trainy, loss_type,
                                                        regularizer_type, loss_weight)
            gradient_function = get_gradient_function(trainx, trainy, loss_type,
                                                      regularizer_type, loss_weight)
            # to know about this function please read about scipy.optimise.minimise
            trained_model_parameters = minimize(objective_function,
                                                start_parameters,
                                                method="CG",
                                                jac=gradient_function,
                                                options={'disp': False,
                                                         'maxiter': 100})
            loss += objective_function(trained_model_parameters.x)
            start_parameters = trained_model_parameters.x
        # prints the batch loss
        print("loss is  ", loss)

    print("Optimizer information:")
    print(trained_model_parameters)
    return trained_model_parameters.x


def test(inputs, weights):
    outputs = classify(inputs, weights)
    probs = 1 / (1 + np.exp(-outputs))
    # this is done to get all terms in 0 or 1 You can change for -1 and 1
    return np.round(probs)


def write_csv_file(outputs, output_file):
    # dumps the output file
    with open(output_file, "w") as out_file:
        out_file.write("ID, Output\n")
        for i in range(len(outputs)):
            out_file.write("{}, {}".format(i + 1, str(outputs[i])) + "\n")


def get_data(data_file):
    with open(data_file, 'r') as df:
        data = df.readlines()

    data = data[1:]
    data = np.array([[float(col) for col in row.split(',')] for row in data])
    input_data = np.hstack([data, np.ones((len(data), 1), dtype=np.float32)])

    return input_data


def main(args):
    train_data_loader = DataLoader(args.train_data_file, args.batch_size)

    trained_model_parameters = train(
        train_data_loader, args.loss_type, args.regularizer_type, args.loss_weight)
    test_data = get_data(args.test_data_file)
    test_data_output = test(test_data, trained_model_parameters)

    write_csv_file(test_data_output, "output.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--loss", action="store", dest="loss_type",
                        type=str, help="Loss function to be used", default="logistic_loss")
    parser.add_argument("--regularizer", action="store", dest="regularizer_type",
                        type=str, help="Regularizer to be used", default=None)
    parser.add_argument("--batch-size", action="store", dest="batch_size",
                        type=int, help="Batch size for training", default=20)
    parser.add_argument("--train-data", action="store", dest="train_data_file",
                        type=str, help="Train data file", default="train.csv")
    parser.add_argument("--test-data", action="store", dest="test_data_file",
                        type=str, help="Test data file", default="test.csv")
    parser.add_argument("--loss-weight", action="store", dest="loss_weight",
                        type=float, help="Relative weight", default=1.0)
    args = parser.parse_args()

    main(args)
