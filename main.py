import data_utils
from LinearRegressionModel import LRModel
import numpy as np

PLOT = False
ALPHA = .03
np.random.seed(878)
MAX_M = 10  # max number of features to explore
REG = 0


def main():
    # generate training and validation data
    training_x_all, training_t = data_utils.generate_data(bins=10)
    validation_x_all, validation_t = data_utils.generate_data(bins=100)

    # run regression for different values of `m`
    models = []
    for m in range(1, MAX_M+1):
        # slice training-set based on desired num-features
        training_x = training_x_all[:, 0:m+1]
        validation_x = validation_x_all[:, 0:m+1]
        model = LRModel(
            num_features=m,
            alpha=ALPHA,
            training_data=(training_x, training_t),
            validation_data=(validation_x, validation_t),
            reg=REG,
        )
        for i in range(100):
            model.update_weights()
        validation_error = model.compute_rss(validation_x, validation_t)
        print(f"{m} | {validation_error}")
        models.append(model)
    # plot error vs m
    if PLOT:
        data_utils.plot_error_vs_m(
            models, validation_x, validation_t, title="validation error vs m")
        data_utils.plot_error_vs_m(
            models, training_x, training_t, title="training error vs m")


if __name__ == "__main__":
    main()
