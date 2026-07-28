from pathlib import Path

import h5py
import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "datasets"
LAYER_SIZES = (20, 7, 5, 1)


def load_data():
    with h5py.File(DATA_DIR / "train_catvnoncat.h5", "r") as file:
        train_x = np.asarray(file["train_set_x"], dtype=np.float64)
        train_y = np.asarray(file["train_set_y"], dtype=np.float64).reshape(1, -1)
    with h5py.File(DATA_DIR / "test_catvnoncat.h5", "r") as file:
        test_x = np.asarray(file["test_set_x"], dtype=np.float64)
        test_y = np.asarray(file["test_set_y"], dtype=np.float64).reshape(1, -1)

    # Each image becomes one column: (height * width * channels, examples).
    train_x = train_x.reshape(train_x.shape[0], -1).T / 255.0
    test_x = test_x.reshape(test_x.shape[0], -1).T / 255.0
    return train_x, train_y, test_x, test_y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def initialize_parameters(layer_sizes, seed=1):
    rng = np.random.default_rng(seed)
    parameters = []
    for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        # He initialization keeps ReLU activations from shrinking or exploding.
        weights = rng.standard_normal((output_size, input_size))
        weights *= np.sqrt(2.0 / input_size)
        bias = np.zeros((output_size, 1))
        parameters.append([weights, bias])
    return parameters


def forward(x, parameters):
    activation = x
    cache = []

    for weights, bias in parameters[:-1]:
        previous_activation = activation
        linear_output = weights @ previous_activation + bias
        activation = np.maximum(0.0, linear_output)
        cache.append((previous_activation, linear_output))

    weights, bias = parameters[-1]
    previous_activation = activation
    linear_output = weights @ previous_activation + bias
    probabilities = sigmoid(linear_output)
    cache.append((previous_activation, linear_output))
    return probabilities, cache


def binary_cross_entropy(probabilities, labels):
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    losses = labels * np.log(probabilities)
    losses += (1.0 - labels) * np.log(1.0 - probabilities)
    return float(-np.mean(losses))


def backward(probabilities, labels, parameters, cache):
    sample_count = labels.shape[1]
    gradients = [None] * len(parameters)

    # For sigmoid plus binary cross-entropy, dZ simplifies to A - Y.
    linear_gradient = probabilities - labels

    for layer in range(len(parameters) - 1, -1, -1):
        weights = parameters[layer][0]
        previous_activation, linear_output = cache[layer]
        weight_gradient = linear_gradient @ previous_activation.T / sample_count
        bias_gradient = np.sum(linear_gradient, axis=1, keepdims=True) / sample_count
        gradients[layer] = (weight_gradient, bias_gradient)

        if layer > 0:
            activation_gradient = weights.T @ linear_gradient
            previous_linear_output = cache[layer - 1][1]
            linear_gradient = activation_gradient * (previous_linear_output > 0.0)

    return gradients


def update_parameters(parameters, gradients, learning_rate):
    for (weights, bias), (weight_gradient, bias_gradient) in zip(
        parameters, gradients
    ):
        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient


def train(x, y, hidden_sizes=LAYER_SIZES[:-1], iterations=2500, learning_rate=0.0075):
    layer_sizes = (x.shape[0], *hidden_sizes, y.shape[0])
    parameters = initialize_parameters(layer_sizes)

    for iteration in range(iterations):
        probabilities, cache = forward(x, parameters)
        gradients = backward(probabilities, y, parameters, cache)
        update_parameters(parameters, gradients, learning_rate)
        if iteration % 500 == 0 or iteration == iterations - 1:
            cost = binary_cross_entropy(probabilities, y)
            print(f"Iteration {iteration:4d} | cost {cost:.6f}")

    return parameters


def predict(x, parameters):
    probabilities, _ = forward(x, parameters)
    return (probabilities >= 0.5).astype(np.int64)


def accuracy(predictions, labels):
    return 100.0 * float(np.mean(predictions == labels))


def main():
    train_x, train_y, test_x, test_y = load_data()
    parameters = train(train_x, train_y)

    train_predictions = predict(train_x, parameters)
    test_predictions = predict(test_x, parameters)
    print(f"Training accuracy: {accuracy(train_predictions, train_y):.2f}%")
    print(f"Test accuracy: {accuracy(test_predictions, test_y):.2f}%")


if __name__ == "__main__":
    main()
