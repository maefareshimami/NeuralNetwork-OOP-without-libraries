import numpy as np


def MSE0(y_predicted:float, y_theorical:float)->float:
    """Compute the Mean Square Error between y_predicted and y"""
    values = np.array([y_predicted, y_theorical])
    return np.linalg.norm(values, ord = 2)

def MSE(y_predicted:float, y_theorical:float)->float:
    """Compute the Mean Square Error between y_predicted and y"""
    return 1.0 / 2.0 * (y_predicted * y_predicted - y_theorical * y_theorical)     # Multiplication cost less than power