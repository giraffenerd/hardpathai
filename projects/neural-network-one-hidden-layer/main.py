from pathlib import Path

import numpy as np


DATA_PATH = Path(__file__).resolve().parent / "datasets" / "planar_dataset.npz"


def load_data():
    with np.load(DATA_PATH) as data:
        return data["X"], data["Y"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def initialize_parameters(input_size, hidden_size, output_size):
    rng = np.random.RandomState(3)
    return {
        "W1": rng.randn(hidden_size, input_size) * 0.01,
        "b1": np.zeros((hidden_size, 1)),
        "W2": rng.randn(output_size, hidden_size) * 0.01,
        "b2": np.zeros((output_size, 1)),
    }


def forward(parameters, x):
    z1 = parameters["W1"] @ x + parameters["b1"]
    a1 = np.tanh(z1)
    z2 = parameters["W2"] @ a1 + parameters["b2"]
    a2 = sigmoid(z2)
    return a1, a2


def train(x, y, hidden_size=4, iterations=10000, learning_rate=1.2):
    parameters = initialize_parameters(x.shape[0], hidden_size, y.shape[0])
    sample_count = x.shape[1]
    for _ in range(iterations):
        a1, a2 = forward(parameters, x)
        dz2 = a2 - y
        dw2 = dz2 @ a1.T / sample_count
        db2 = np.sum(dz2, axis=1, keepdims=True) / sample_count
        dz1 = parameters["W2"].T @ dz2 * (1.0 - a1**2)
        dw1 = dz1 @ x.T / sample_count
        db1 = np.sum(dz1, axis=1, keepdims=True) / sample_count
        parameters["W1"] -= learning_rate * dw1
        parameters["b1"] -= learning_rate * db1
        parameters["W2"] -= learning_rate * dw2
        parameters["b2"] -= learning_rate * db2
    return parameters


def predict(parameters, x):
    return (forward(parameters, x)[1] >= 0.5).astype(int)


def main():
    x, y = load_data()
    parameters = train(x, y)
    predictions = predict(parameters, x)
    accuracy = 100.0 * np.mean(predictions == y)
    print(f"Training examples: {x.shape[1]}")
    print(f"Training accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
