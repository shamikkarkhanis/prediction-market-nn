from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class TrainHistory:
    losses: List[float]


class NeuralNetwork:
    def __init__(
        self,
        layer_sizes: List[int],
        learning_rate: float = 0.01,
        random_seed: int | None = None,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError(
                "layer_sizes must include at least input and output layers"
            )

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(random_seed)

        self.parameters: Dict[str, np.ndarray] = {}
        self.caches: Dict[str, np.ndarray] = {}
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        self.parameters.clear()
        for layer in range(1, len(self.layer_sizes)):
            fan_in = self.layer_sizes[layer - 1]
            fan_out = self.layer_sizes[layer]
            self.parameters[f"W{layer}"] = self.rng.standard_normal(
                (fan_out, fan_in)
            ) * np.sqrt(2.0 / fan_in)
            self.parameters[f"b{layer}"] = np.zeros((fan_out, 1))

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_derivative(z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def forward_propagation(self, x: np.ndarray) -> np.ndarray:
        a = x
        self.caches = {"A0": x}
        last_layer = len(self.layer_sizes) - 1

        for layer in range(1, last_layer):
            w = self.parameters[f"W{layer}"]
            b = self.parameters[f"b{layer}"]
            z = w @ a + b
            a = self._relu(z)
            self.caches[f"Z{layer}"] = z
            self.caches[f"A{layer}"] = a

        z_last = (
            self.parameters[f"W{last_layer}"] @ a + self.parameters[f"b{last_layer}"]
        )
        a_last = self._sigmoid(z_last)
        self.caches[f"Z{last_layer}"] = z_last
        self.caches[f"A{last_layer}"] = a_last
        return a_last

    def cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        m = y_true.shape[1]
        loss = (
            -np.sum(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)) / m
        )
        return float(loss)

    def backward_propagation(
        self, x: np.ndarray, y_true: np.ndarray
    ) -> Dict[str, np.ndarray]:
        m = x.shape[1]
        last_layer = len(self.layer_sizes) - 1
        grads: Dict[str, np.ndarray] = {}

        a_last = self.caches[f"A{last_layer}"]
        d_z = a_last - y_true

        for layer in range(last_layer, 0, -1):
            a_prev = self.caches[f"A{layer - 1}"]
            w = self.parameters[f"W{layer}"]

            grads[f"dW{layer}"] = (d_z @ a_prev.T) / m
            grads[f"db{layer}"] = np.sum(d_z, axis=1, keepdims=True) / m

            if layer > 1:
                z_prev = self.caches[f"Z{layer - 1}"]
                d_a_prev = w.T @ d_z
                d_z = d_a_prev * self._relu_derivative(z_prev)

        return grads

    def _gradient_descent_step(self, grads: Dict[str, np.ndarray]) -> None:
        for layer in range(1, len(self.layer_sizes)):
            self.parameters[f"W{layer}"] -= self.learning_rate * grads[f"dW{layer}"]
            self.parameters[f"b{layer}"] -= self.learning_rate * grads[f"db{layer}"]

    def _iterate_batches(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ):
        sample_count = x.shape[1]
        if shuffle:
            indices = self.rng.permutation(sample_count)
            x_epoch = x[:, indices]
            y_epoch = y_true[:, indices]
        else:
            x_epoch = x
            y_epoch = y_true

        for start in range(0, sample_count, batch_size):
            end = min(start + batch_size, sample_count)
            yield x_epoch[:, start:end], y_epoch[:, start:end]

    def train(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        epochs: int = 1000,
        print_every: int = 100,
        gradient_descent: str = "batch",
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> TrainHistory:
        mode = gradient_descent.lower()
        sample_count = x.shape[1]
        if mode == "batch":
            effective_batch_size = sample_count
        elif mode == "mini-batch":
            if batch_size <= 0:
                raise ValueError(
                    "batch_size must be > 0 for mini-batch gradient descent"
                )
            effective_batch_size = min(batch_size, sample_count)
        elif mode == "stochastic":
            effective_batch_size = 1
        else:
            raise ValueError(
                "gradient_descent must be one of: batch, mini-batch, stochastic"
            )

        losses: List[float] = []

        for epoch in range(1, epochs + 1):
            for x_batch, y_batch in self._iterate_batches(
                x=x,
                y_true=y_true,
                batch_size=effective_batch_size,
                shuffle=shuffle,
            ):
                self.forward_propagation(x_batch)
                grads = self.backward_propagation(x_batch, y_batch)
                self._gradient_descent_step(grads)

            y_pred = self.forward_propagation(x)
            loss = self.cost(y_true, y_pred)
            losses.append(loss)

            if print_every > 0 and (
                epoch == 1 or epoch % print_every == 0 or epoch == epochs
            ):
                print(f"Epoch {epoch}/{epochs} - loss: {loss:.6f}")

        return TrainHistory(losses=losses)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward_propagation(x)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict_proba(x)
        return (probabilities >= threshold).astype(int)
