import numpy as np
from easydict import EasyDict

cfg = EasyDict()

# Linear Regression Configurations


# Dataset path
cfg.dataframe_path = 'linear_regression_dataset.csv'
cfg.learning_rate = 0.8
cfg.n_iterations = 5000
cfg.regularization_coeff = 0
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1

cfg.base_functions = [  # TODO list of basis functions
    lambda x: 1,
    lambda x: x,
    lambda x: x ** 2,
]
