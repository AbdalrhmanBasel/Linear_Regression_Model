import numpy as np

from configs.linear_regression_cfg import cfg
from datasets.linear_regression_dataset import LinearRegressionDataset
from models.linear_regression_model import LinearRegression
from utils1.metrics import MSE
from utils1.visualisation import Visualisation

if __name__ == '__main__':
    # Dataset
    dataset = LinearRegressionDataset(cfg=cfg)
    data = dataset()

    # Getting data Split
    X_train = data['inputs']['train']
    y_train = data['targets']['train']
    X_test = data['inputs']['test']
    y_test = data['targets']['test']

    # Predicted With Model
    LinearRegression = LinearRegression(cfg.base_functions, cfg.regularization_coeff, cfg.learning_rate, cfg.n_iterations)
    LinearRegression.fit(X_train, y_train)
    prediction = LinearRegression.predict(X_test)
    print(prediction)

    mse = MSE(prediction, y_test)
    print(f"Mean Squared Error: {mse}")

    # Visualize The Model
    Graph = Visualisation()
    Graph.visualise_predicted_trace(prediction, X_test, y_test, plot_title='Predicted Trace and Targets')





