import numpy as np
import matplotlib.pyplot as plt
import time

import constants as cst
import neural_network
import loss_function
import activation_function
import dataset_building


def forwardStep(neural_network:list, x:np.array)->float:
    """Compute y with x as entry"""
    layers = neural_network.list_layers
    for i in range(0, cst.LAYERS_SCHEME_LENGHT):
        lyr = layers[i]
        if i == 0:
            lyr.inputs = x
            lyr.computeOutputsFirstLayer()
            lyr.computeActivationFunctionFirstLayer()
        elif i == cst.LAYERS_SCHEME_LENGHT - 1:
            lyr.inputs = layers[i - 1].act_func
            lyr.computeOutputsLastLayer()
            lyr.computeActivationFunctionLastLayer()
        else:
            lyr.inputs = layers[i - 1].act_func
            lyr.computeOutputs()
            lyr.computeActivationFunction()
    return layers[cst.LAYERS_SCHEME_LENGHT - 1].act_func

def lossFunctionStep(neural_network:list, x:np.array, y_theorical:float)->float:
    """Compute the loss function with y_predicted and y_theorical"""
    neural_network.entries = x
    y_predicted = forwardStep(neural_network, x)
    y_predicted = y_predicted[0]
    prediction_error = loss_function.MSE(y_predicted, y_theorical)
    prediction_error = prediction_error[0]
    return y_predicted, prediction_error

def errorLevelBatch(neural_network:list, list_errors_per_iteration:float)->list:
    """Compute the backpropagation of the gradient using batches"""
    layers = neural_network.list_layers
    array_error_level = np.full(cst.LAYERS_SCHEME_LENGHT, 0.0, dtype = object)
    array_error_level[0] = np.array([0.0])
    array_error_level[-1] = np.array([0.0])
    for i in range(cst.LAYERS_SCHEME_LENGHT - 1, 0, -1):
        lyr = layers[i]
        if i == cst.LAYERS_SCHEME_LENGHT - 1:
            error_level = np.array([sum(list_errors_per_iteration)])
            #error_level_1 = lyr.act_func - np.array([list_errors_per_iteration])
            #error_level_2 = activation_function.derivative_tanh(lyr.outputs)
            #error_level = error_level_1 * error_level_2
        else:
            error_level_weights_errors = np.dot(np.transpose(layers[i + 1].weights), array_error_level[i + 1])
            error_level_act_func = activation_function.derivative_tanh(lyr.outputs)
            error_level = error_level_weights_errors * error_level_act_func
        array_error_level[i] = error_level
    return array_error_level

def weightGradientBatch(neural_network:list, list_errors_per_iteration:float)->np.array:
    """Compute the weight gradient for each layer using batches"""
    layers = neural_network.list_layers
    array_error_level = errorLevelBatch(neural_network, list_errors_per_iteration)
    array_weight_gradient = np.full(cst.LAYERS_SCHEME_LENGHT, 1.0, dtype = object)
    array_weight_gradient[0] = np.array([1.0])
    array_weight_gradient[-1] = np.array([1.0])
    for i in range(cst.LAYERS_SCHEME_LENGHT - 2, 0, -1):
        array_weight_gradient[i] = array_error_level[i] * np.transpose(layers[i - 1].act_func)
    return array_weight_gradient

def biasGradientBatch(neural_network:list, sum_error_per_iteration:float)->np.array:
    """Compute the bias gradient for each layer using batches"""
    array_error_level = errorLevelBatch(neural_network, sum_error_per_iteration)
    array_bias_gradient = np.full(cst.LAYERS_SCHEME_LENGHT, 0.0, dtype = object)
    array_bias_gradient[0] = np.array([0.0])
    array_bias_gradient[-1] = np.array([0.0])
    for i in range(cst.LAYERS_SCHEME_LENGHT - 2, 0, -1):
        array_bias_gradient[i] = np.transpose(array_error_level[i])
    return array_bias_gradient

def backwardStepBatch(neural_network:list, sum_error_per_iteration:float)->None:
    """Update all the weights and bias using batches"""
    layers = neural_network.list_layers
    array_weight_gradient = weightGradientBatch(neural_network, sum_error_per_iteration)     # Change function if you don't use batch
    array_bias_gradient = biasGradientBatch(neural_network, sum_error_per_iteration)
    for i in range(1, cst.LAYERS_SCHEME_LENGHT - 1):
        lyr = layers[i]
        for j in range(0, lyr.height):
            lyr.weights[j] -= cst.LEARNING_RATE * array_weight_gradient[i][j]
            lyr.bias[j] -= cst.LEARNING_RATE * array_bias_gradient[i][j]
    return None

def getParameters(neural_network:list)->list:
    """Get all weights and bias of neural network"""
    layers = neural_network.list_layers
    array_weights = []
    array_bias = []
    for i in range(0, cst.LAYERS_SCHEME_LENGHT):
        array_weights.append(layers[i].weights)
        array_bias.append(layers[i].bias)
    return array_weights, array_bias

def displayNeuralNetwork(i:int, y_predicted:float = None, prediction_error:float = None)->None:
    """Write in a txt file weights and bioas of layers"""
    array_weights, array_bias = getParameters(nn_sinus)
    with open(f"parameters/parameters_{i}.txt", "w", encoding = "utf-8") as parameters_file:
        cst.LAYERS_SCHEME_LENGHT = len(array_weights)
        parameters_file.write(f"Exit Predicted: {y_predicted}\nPrediction Error: {prediction_error}\n\n\n")
        for p in range(0, cst.LAYERS_SCHEME_LENGHT):
            if p == 0:
                parameters_file.write(f"Weights Layer {p} (Entry):\n{array_weights[p]}\nBias Layer {p}:\n{array_bias[p]}\n\n")
            elif p == cst.LAYERS_SCHEME_LENGHT - 1:
                parameters_file.write(f"Weights Layer {p} (Exit):\n{array_weights[p]}\nBias Layer {p}:\n{array_bias[p]}\n\n")
            else:
                parameters_file.write(f"Weights Layer {p}:\n{array_weights[p]}\nBias Layer {p}:\n{array_bias[p]}\n\n")
    return None


if __name__ == "__main__":

    list_x, list_y, list_x_training, list_y_training, list_x_testing, list_y_testing, list_x_nn, list_y_nn = dataset_building.dataset_sinus()
    list_x_predicted = []
    list_y_predicted = []
    list_prediction_error = []
    list_epochs = [i for i in range(0, cst.EPOCHS_NB)]

    neural_network_name = "Sinus Neural Network"
    nn_sinus = neural_network.NeuralNetwork(neural_network_name, cst.LAYERS_SCHEME_LENGHT, cst.LAYERS_SCHEME)
    displayNeuralNetwork(0)


    t1 = time.time()
    for k in range(0, cst.EPOCHS_NB):
        list_temp_prediction_error = []
        for p in range(0, cst.BATCHES_NB):
            list_errors_per_iteration = []
            for q in range(0, cst.BATCH_SIZE):
                x_training = list_x_training[p * cst.BATCH_SIZE + q]
                y_training = list_y_training[p * cst.BATCH_SIZE + q]
                y_predicted, prediction_error = lossFunctionStep(nn_sinus, np.array([x_training]), y_training)
                list_x_predicted.append(x_training)
                list_y_predicted.append(y_predicted)
                list_temp_prediction_error.append(prediction_error)
                list_errors_per_iteration.append(prediction_error)
            backwardStepBatch(nn_sinus, list_errors_per_iteration)
            displayNeuralNetwork(p, y_predicted, prediction_error)
        list_prediction_error.append(sum(list_temp_prediction_error))
        print(f"Compute: {round(k / cst.EPOCHS_NB * 100, 2)}%")
    t2 = time.time()

    for i in range(0, cst.TESTING_DATASET_SIZE):
        list_y_nn[i] = lossFunctionStep(nn_sinus, np.array([list_x_nn[i]]), list_y_nn[i])[0]

    print(f"Time Compute: {round((t2 - t1) / 60, 2)}m")


    displayNeuralNetwork(cst.BATCHES_NB)


    plt.figure(figsize = (10, 5))
    #plt.subplot(2, 1, 1)
    plt.plot(list_epochs, list_prediction_error, "c.--")
    plt.title("Neural Network Analyze")
    plt.xlabel("epochs")
    plt.ylabel("Error")
    plt.grid()
    plt.show()

    plt.figure(figsize = (10, 5))
    #plt.subplot(2, 1, 2)
    plt.plot(list_x, list_y, "gs", label = "Dataset")
    plt.plot(list_x_training, list_y_training, "b+", label = "Training Dataset")
    plt.plot(list_x_testing, list_y_testing, "r+", label = "Testing Dataset")
    plt.plot(list_x_nn, list_y_nn, "k.", label = "Predicted Dataset")
    plt.title("Neural Network - Sinus")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

    #plt.tight_layout()
    #plt.show()