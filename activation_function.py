import numpy as np
import sympy as sp


def identity(output):
    """Compute the identity function"""
    return output

def derivative_identity(output:np.array)->np.array:
    """Compute the identity function derivative"""
    z = sp.symbols("z")
    f = sp.Identity(z)
    df_dz = sp.diff(f, z)
    nb_values = len(output)
    values = np.zeros(nb_values)
    for i in range(0, nb_values):
        values[i] = df_dz.subs({z: output[i]})
    return values


def tanh(output:np.array)->np.array:
    """Compute the tanh"""
    return np.tanh(output)

def derivative_tanh(output:np.array)->np.array:
    """Compute the tanh derivative"""
    nb_outputs = len(output)
    function_tanh = tanh(output)
    derivative = np.zeros(nb_outputs)
    for i in range(0, nb_outputs):
        derivative[i] = 1 - function_tanh[i] * function_tanh[i]
    return derivative


"""
def tanh(output:np.array)->np.array:
    Compute the tanh
    return np.tanh(output)

def derivative_tanh(output:np.array)->np.array:
    Compute the tanh derivative
    z = sp.symbols("z")
    f = sp.tanh(z)
    df_dz = sp.diff(f, z)
    nb_values = len(output)
    values = np.zeros(nb_values)
    for i in range(0, nb_values):
        values[i] = df_dz.subs({z: output[i]})
    return values
"""


def sigmoid(output:np.array)->np.array:
    """Compute the sigmoid"""
    return 1 / (1 + np.exp(-output))

def derivative_sigmoid(output:np.array)->np.array:
    """Compute the sigmoid derivative"""
    nb_outputs = len(output)
    function_sigmoid = sigmoid(output)
    derivative = np.zeros(nb_outputs)
    for i in range(0, nb_outputs):
        derivative[i] = function_sigmoid[i] * (1 - function_sigmoid[i])
    return derivative


def reLu(output:np.array)->np.array:
    """Compute the ReLu"""
    nb_outputs = len(output)
    values = np.zeros(nb_outputs)
    for i in range(0, nb_outputs):
        if output[i] >= 0.0:
            values[i] = output[i]
    return values

def derivative_reLu(output:np.array)->np.array:
    """Compute the ReLu derivative"""
    nb_outputs = len(output)
    derivative = np.zeros(nb_outputs)
    for i in range(0, nb_outputs):
        if output[i] >= 0.0:
            derivative[i] = 1.0
    return derivative