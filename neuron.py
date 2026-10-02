import numpy as np
import random as rd

import activation_function
import constants as cst


class Neuron:

    def __init__(self, layer, number, height_previous_layer):
        self.layer = layer
        self.number = number
        self.height_previous_layer = height_previous_layer
        self.bias = np.array(rd.random() * cst.BIAS_RANGE_FACTOR)
        self.inputs = np.zeros(self.height_previous_layer)
        self.weights = np.zeros(self.height_previous_layer)
        self.createWeights()
        self.output = np.zeros(1)
        self.act_func = np.zeros(1)
        #self.output = np.array([np.dot(self.weights, self.inputs) + self.bias])
        #self.act_func = activation_function.tanh(self.output)
    
    def __repr__(self):
        return f"<class Neuron | Layer: {self.layer} | Number: {self.number} | Number of inputs: {self.height_previous_layer} | Bias: {self.bias}>"
    

    def createWeights(self):
        for j in range(0, self.height_previous_layer):
            #self.weights[j] = np.random.normal(loc = cst.WEIGHTS_MEAN, scale = cst.WEIGHTS_VARIANCE, size = 1)     #  Normal distribution
            self.weights[j] = rd.uniform(-cst.WEIGHTS_SEMI_RANGE, cst.WEIGHTS_SEMI_RANGE)   #  Uniform distribution
        return None

    def computeOutputs(self):
        self.output = np.dot(self.weights, self.inputs) + self.bias
        return None
    
    def computeActivationFunction(self):
        self.act_func = activation_function.tanh(self.output)
        return None


class NeuronFirstLayer:

    def __init__(self, layer, number, height_previous_layer):
        self.layer = layer
        self.number = number
        self.height_previous_layer = height_previous_layer
        self.bias = np.array(0.0)
        self.inputs = np.zeros(self.height_previous_layer)
        self.weights = np.zeros(self.height_previous_layer)
        self.createWeightsFirstLayer()
        self.output = np.zeros(self.height_previous_layer)
        self.act_func = np.array(np.dot(self.weights, self.inputs) + self.bias)
    
    def __repr__(self):
        return f"<class NeuronFirstLayer | Layer: {self.layer} | Number: {self.number} | Number of inputs: {self.height_previous_layer} | Bias: {self.bias}>"
    

    def createWeightsFirstLayer(self):
        for j in range(0, self.height_previous_layer):
            self.weights[j] = 1.0
        return None
    
    def computeOutputsFirstLayer(self):
        self.output = np.dot(self.weights, self.inputs) + self.bias
        return None

    def computeActivationFunctionFirstLayer(self):
        self.act_func = activation_function.identity(self.output)
        return None


class NeuronLastLayer:

    def __init__(self, layer, number, height_previous_layer):
        self.layer = layer
        self.number = number
        self.height_previous_layer = height_previous_layer
        self.bias = np.array(0.0)
        self.inputs = np.zeros(self.height_previous_layer)
        self.weights = np.zeros(self.height_previous_layer)
        self.createWeightsLastLayer()
        self.output = np.zeros(self.height_previous_layer)
        self.act_func = activation_function.identity(self.output)
    
    def __repr__(self):
        return f"<class NeuronLastLayer | Layer: {self.layer} | Number: {self.number} | Number of inputs: {self.height_previous_layer} | Bias: {self.bias}>"
    

    def createWeightsLastLayer(self):
        for j in range(0, self.height_previous_layer):
            self.weights[j] = 1.0
        return None
    
    def computeOutputsLastLayer(self):
        self.output = np.dot(self.weights, self.inputs) + self.bias
        return None

    def computeActivationFunctionLastLayer(self):
        self.act_func = activation_function.identity(self.output)
        return None