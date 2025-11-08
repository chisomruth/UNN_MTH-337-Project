import os
import random
import tensorflow as tf
import numpy as np


class Config:
    DATA_PATH = './Datasets/Salary Data.csv'
    BATCH_SIZE = 32
    EPOCHS = 100
    SEED = 42
    TEST_SIZE = 0.2
    VALIDATION_SPLIT = 0.2
    LEARNING_RATES = [1e-4, 1e-3, 1e-2, 1e-1]
    OPTIMIZERS = ['adam', 'adagrad', 'rmsprop', 'sgd']
    RESULTS_DIR = 'mlp_results'

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()