import numpy as np

import neuron
import activation_function


class Layer:

    def __init__(self, number, height_layer, height_previous_layer):
        self.number = number
        self.height = height_layer
        self.height_previous_layer = height_previous_layer
        self.list_neurons = [neuron.Neuron(self.number, i, self.height_previous_layer) for i in range(0, self.height)]
        self.weights = np.zeros((self.height, self.height_previous_layer))
        self.createWeightsMatrix()
        self.bias = np.array([self.list_neurons[i].bias for i in range(0, self.height)])
        self.inputs = np.array([self.list_neurons[i].inputs for i in range(0, self.height)])
        self.outputs = np.array([self.list_neurons[i].output for i in range(0, self.height)])
        self.act_func = np.array([self.list_neurons[i].act_func for i in range(0, self.height)])
    
    def __repr__(self):
        return f"<class Layer | Number: {self.number} | Height: {self.height} | Number of inputs: {self.height_previous_layer}>"
    

    def createWeightsMatrix(self):
        for i in range(0, self.height):
            for j in range(0, self.height_previous_layer):
                self.weights[i][j] = self.list_neurons[i].weights[j]
        return None

    def computeOutputs(self):
        for i in range(0, self.height):
            self.list_neurons[i].computeOutputs()
        return None
    
    def computeActivationFunction(self):
        for i in range(0, self.height):
            self.list_neurons[i].computeActivationFunction()
        return None

    """
    def computeOutputs(self):
        self.outputs = np.dot(self.weights, self.inputs) + self.bias
        return None
    
    def computeActivationFunction(self):
        self.act_func = activation_function.tanh(self.outputs)
        return None
    """


class FirstLayer:

    def __init__(self, height_layer, height_inputs):
        self.number = 0
        self.height = height_layer
        self.height_inputs = height_inputs
        self.list_neurons = [neuron.NeuronFirstLayer(self.number, i, self.height_inputs) for i in range(0, self.height)]
        self.weights = np.zeros((self.height, self.height_inputs))
        self.createWeightsMatrixFirstLayer()
        self.bias = np.array([self.list_neurons[i].bias for i in range(0, self.height)])
        self.inputs = np.array([self.list_neurons[i].inputs for i in range(0, self.height)])
        self.outputs = np.array([self.list_neurons[i].output for i in range(0, self.height)])
        self.act_func = np.array([self.list_neurons[i].act_func for i in range(0, self.height)])
    
    def __repr__(self):
        return f"<class FirstLayer | Number: {self.number} | Height: {self.height} | Number of inputs: {self.height_inputs}>"
    

    def createWeightsMatrixFirstLayer(self):
        for i in range(0, self.height):
            for j in range(0, self.height_inputs):
                self.weights[i][j] = self.list_neurons[i].weights[j]
        return None

    def computeOutputsFirstLayer(self):
        for i in range(0, self.height):
            self.list_neurons[i].computeOutputsFirstLayer()
        return None
    
    def computeActivationFunctionFirstLayer(self):
        for i in range(0, self.height):
            self.list_neurons[i].computeActivationFunctionFirstLayer()
        return None

    """
    def computeOutputsFirstLayer(self):
        self.outputs = np.dot(self.weights, self.inputs) + self.bias
        return None

    def computeActivationFunctionFirstLayer(self):
        self.act_func = activation_function.identity(self.outputs)
        return None
    """


class LastLayer:

    def __init__(self, number, height_layer, height_previous_layer):
        self.number = number
        self.height = height_layer
        self.height_previous_layer = height_previous_layer
        self.list_neurons = [neuron.NeuronLastLayer(self.number, i, self.height_previous_layer) for i in range(0, self.height)]
        self.weights = np.zeros((self.height, self.height_previous_layer))
        self.createWeightsMatrixLastLayer()
        self.bias = np.array([self.list_neurons[i].bias for i in range(0, self.height)])
        self.inputs = np.array([self.list_neurons[i].inputs for i in range(0, self.height)])
        self.outputs = np.array([self.list_neurons[i].output for i in range(0, self.height)])
        self.act_func = np.array([self.list_neurons[i].act_func for i in range(0, self.height)])
    
    def __repr__(self):
        return f"<class LastLayer | Number: {self.number} | Height: {self.height} | Number of inputs: {self.height_previous_layer}>"
    

    def createWeightsMatrixLastLayer(self):
        for i in range(0, self.height):
            for j in range(0, self.height_previous_layer):
                self.weights[i][j] = self.list_neurons[i].weights[j]
        return None

    def computeOutputsLastLayer(self):
        for i in range(0, self.height):
            self.list_neurons[i].computeOutputsLastLayer()
        return None
    
    def computeActivationFunctionLastLayer(self):
        for i in range(0, self.height):
            self.list_neurons[i].computeActivationFunctionLastLayer()
        return None
    
    """
    def computeOutputsLastLayer(self):
        self.outputs = np.dot(self.weights, self.inputs) + self.bias
        return None

    def computeActivationFunctionLastLayer(self):
        self.act_func = activation_function.identity(self.outputs)
        return None
    """