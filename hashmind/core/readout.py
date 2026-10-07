"""Host-side linear readouts trained in closed form (ridge regression)."""

from __future__ import annotations

import numpy as np


class RidgeReadout:
    """y = F W + b, fitted by ridge regression. Works for regression and, via
    one-hot targets, multi-class classification (argmax of the outputs).

    Uses the primal (D x D) or dual (N x N) solve, whichever is smaller.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.classes: np.ndarray | None = None

    def fit(self, F: np.ndarray, Y: np.ndarray) -> "RidgeReadout":
        F = np.asarray(F, np.float64)
        Y = np.asarray(Y, np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        self.mu = F.mean(0)
        ym = Y.mean(0)
        Fc, Yc = F - self.mu, Y - ym
        n, d = Fc.shape
        if d <= n:
            self.W = np.linalg.solve(Fc.T @ Fc + self.alpha * np.eye(d), Fc.T @ Yc)
        else:
            self.W = Fc.T @ np.linalg.solve(Fc @ Fc.T + self.alpha * np.eye(n), Yc)
        self.b = ym
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("readout not fitted")
        return (np.asarray(F, np.float64) - self.mu) @ self.W + self.b

    # -- classification convenience --------------------------------------------

    def fit_classes(self, F: np.ndarray, labels: np.ndarray) -> "RidgeReadout":
        self.classes, idx = np.unique(labels, return_inverse=True)
        Y = -np.ones((len(labels), len(self.classes)))
        Y[np.arange(len(labels)), idx] = 1.0
        return self.fit(F, Y)

    def predict_classes(self, F: np.ndarray) -> np.ndarray:
        if self.classes is None:
            raise RuntimeError("call fit_classes first")
        return self.classes[self.predict(F).argmax(1)]

    @property
    def n_parameters(self) -> int:
        return 0 if self.W is None else self.W.size + self.W.shape[1]
