import os
import random
import tensorflow as tf
import numpy as np


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    tf.config.experimental.enable_op_determinism()


class Config:
    TRAIN_DIR = './Datasets/mechanical-tools-dataset/train_data_V2/train_data_V2'
    VAL_DIR = './Datasets/mechanical-tools-dataset/validation_data_V2/validation_data_V2'

    BATCH_SIZE = 64
    IMAGE_SIZE = (300, 300)
    EPOCHS = 50
    SEED = 42
    LEARNING_RATES = [1e-4, 1e-3, 1e-2, 1e-1]
    OPTIMIZERS = ['adam', 'adagrad', 'rmsprop', 'sgd']
    RESULTS_DIR = 'cnn_results'

