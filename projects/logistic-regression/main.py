from pathlib import Path

import h5py
import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "datasets"


def load_data():
    #preproccessing
    with h5py.File(DATA_DIR / "train_catvnoncat.h5", "r") as train_file:
        train_x = np.asarray(train_file["train_set_x"], dtype=np.float64)
        train_y = np.asarray(train_file["train_set_y"], dtype=np.float64).reshape(1, -1) 
        ### one row, auto cols
        ##   That is different from a two-dimensional row vector:
        #   (1, 209)
        #   Although both contain 209 values, NumPy treats their dimensions differently:
        #   a.shape == (209,)     # 1D array
        #   b.shape == (1, 209)   # 2D row vector
    with h5py.File(DATA_DIR / "test_catvnoncat.h5", "r") as test_file:
        test_x = np.asarray(test_file["test_set_x"], dtype=np.float64)
        test_y = np.asarray(test_file["test_set_y"], dtype=np.float64).reshape(1, -1)
    train_x = train_x.reshape(train_x.shape[0], -1).T / 255.0
    test_x = test_x.reshape(test_x.shape[0], -1).T / 255.0
    return train_x, train_y, test_x, test_y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def train(x, y, iterations=2000, learning_rate=0.005):
    weights = np.zeros((x.shape[0], 1))
    bias = 0.0
    sample_count = x.shape[1]
    for _ in range(iterations):
        probabilities = sigmoid(weights.T @ x + bias)
        difference = probabilities - y
        weights -= learning_rate * (x @ difference.T / sample_count)
        bias -= learning_rate * float(np.sum(difference) / sample_count)
    return weights, bias


def predict(weights, bias, x):
    return (sigmoid(weights.T @ x + bias) >= 0.5).astype(int)


def main():
    train_x, train_y, test_x, test_y = load_data()
    weights, bias = train(train_x, train_y)
    train_predictions = predict(weights, bias, train_x)
    test_predictions = predict(weights, bias, test_x)
    train_accuracy = 100.0 * np.mean(train_predictions == train_y)
    test_accuracy = 100.0 * np.mean(test_predictions == test_y)
    print(f"Training accuracy: {train_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    for index, prediction in enumerate(test_predictions.ravel()):
        label = "cat" if prediction == 1 else "not cat"
        print(f"Test image {index}: {label}")


if __name__ == "__main__":
    main()
