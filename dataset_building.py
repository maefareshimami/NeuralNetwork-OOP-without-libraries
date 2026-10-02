import numpy as np
import random as rd

import constants as cst


def dataset_sinus()->(list, list, list, list, list, list, list, list):
    """Create properly the training dataset and the testing dataset"""
    list_x = []
    list_y = []
    for _ in range(0, cst.DATASET_SIZE):
        x = rd.uniform(0, 2 * np.pi)
        y = float(np.sin(x))
        list_x.append(x)
        list_y.append(y)
    list_x_training = list_x[:cst.TRAINING_DATASET_SIZE]
    list_y_training = list_y[:cst.TRAINING_DATASET_SIZE]
    list_x_testing = list_x[cst.TRAINING_DATASET_SIZE:]
    list_y_testing = list_y[cst.TRAINING_DATASET_SIZE:]
    list_x_nn = list_x[cst.TRAINING_DATASET_SIZE:]
    list_y_nn = list_y[cst.TRAINING_DATASET_SIZE:]
    return list_x, list_y, list_x_training, list_y_training, list_x_testing, list_y_testing, list_x_nn, list_y_nn