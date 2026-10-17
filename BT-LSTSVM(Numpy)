"""Offline, label-free-at-prediction port of the legacy BT-LSTSVM equations.

Only the linear and RBF kernels used by the old classifier are supported.
The historical test-label remapping and random wrong-class assignment are not
part of this implementation. No model trained here is registered with Agent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _features(value: np.ndarray, *, width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("features must be a nonempty two-dimensional matrix")
    if width is not None and array.shape[1] != width:
        raise ValueError("feature dimension does not match training data")
    if not np.isfinite(array).all():
        raise ValueError("nonfinite features are not accepted or imputed")
    return array


def _kernel(
    left: np.ndarray,
    right: np.ndarray,
    kind: str,
    bandwidth: float,
) -> np.ndarray:
    if kind == "linear":
        return left @ right.T
    distances = np.maximum(
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T,
        0.0,
    )
    # MATLAB kernelfun.m calls this quantity pars(1), not gamma or sigma.
    return np.exp(-distances / (2.0 * bandwidth))


@dataclass(frozen=True)
class TwinScores:
    positive_distance: np.ndarray
    negative_distance: np.ndarray


class BinaryLSTSVM:
    """Least-squares twin SVM for labels +1 and -1.

    Uses the old LSTWSVM_DAG normal equations and epsilon=1e-4 default.
    A and B do not receive the spurious zero-vector rows in the MATLAB code.
    """

    def __init__(
        self,
        *,
        c1: float = 1.0,
        c2: float = 1.0,
        kernel: str = "rbf",
        bandwidth: float = 1.0,
        epsilon: float = 1e-4,
    ) -> None:
        if kernel not in {"linear", "rbf"}:
            raise ValueError("kernel must be 'linear' or 'rbf'")
        if not all(
            np.isfinite(value) and value > 0
            for value in (c1, c2, bandwidth, epsilon)
        ):
            raise ValueError("c1, c2, bandwidth and epsilon must be finite and positive")
        self.c1, self.c2 = float(c1), float(c2)
        self.kernel, self.bandwidth = kernel, float(bandwidth)
        self.epsilon = float(epsilon)

    def fit(self, x: np.ndarray, y: np.ndarray) -> BinaryLSTSVM:
        """Fit the two regularized least-squares separating hyperplanes."""
        feature_matrix = _features(x)
        labels = np.asarray(y)
        if labels.shape != (feature_matrix.shape[0],) or not np.isin(
            labels, (-1, 1)
        ).all():
            raise ValueError("binary labels must be a vector containing only +1 and -1")

        positive, negative = feature_matrix[labels == 1], feature_matrix[labels == -1]
        if not len(positive) or not len(negative):
            raise ValueError("both binary classes require training rows")

        self._width = feature_matrix.shape[1]
        self._anchors = np.vstack((positive, negative)) if self.kernel == "rbf" else None
        positive_design = self._design(positive)
        negative_design = self._design(negative)
        positive_gram = positive_design.T @ positive_design
        negative_gram = negative_design.T @ negative_design
        identity = np.eye(positive_design.shape[1])

        # BT-LSTSVM fits one least-squares hyperplane near each class. The two
        # regularized systems below are the paired normal equations.
        # For unequal class sizes the MATLAB Woodbury branches are equivalent
        # to these direct systems, including their asymmetric ridge factors.
        ridge1 = (
            self.epsilon / self.c1
            if self.kernel == "rbf" and len(positive) > len(negative)
            else self.epsilon
        )
        ridge2 = (
            self.epsilon / self.c2
            if self.kernel == "rbf" and len(positive) < len(negative)
            else self.epsilon
        )
        self._plane1 = np.linalg.solve(
            positive_gram / self.c1 + negative_gram + ridge1 * identity,
            -(negative_design.T @ np.ones(len(negative))),
        )
        self._plane2 = np.linalg.solve(
            negative_gram / self.c2 + positive_gram + ridge2 * identity,
            positive_design.T @ np.ones(len(positive)),
        )
        return self

    def _design(self, x: np.ndarray) -> np.ndarray:
        base = (
            x
            if self._anchors is None
            else _kernel(x, self._anchors, self.kernel, self.bandwidth)
        )
        return np.column_stack((base, np.ones(len(x))))

    def scores(self, x: np.ndarray) -> TwinScores:
        if not hasattr(self, "_plane1"):
            raise ValueError("model has not been fitted")
        data = _features(x, width=self._width)
        design = self._design(data)
        return TwinScores(
            positive_distance=np.abs(design @ self._plane1),
            negative_distance=np.abs(design @ self._plane2),
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Assign each row to the nearer of the two fitted hyperplanes."""
        scores = self.scores(x)
        # Same tie break as MATLAB: plane 1 wins.
        return np.where(scores.positive_distance <= scores.negative_distance, 1, -1)


class BTLSTSVM:
    """Binary tree: peel off each class in fixed order; final class is fallback.

    The class order is explicit, not optimized using the test set. Predictions
    require only X; no test labels are accepted or consulted.
    """

    def __init__(
        self,
        *,
        c1: float = 1.0,
        c2: float = 1.0,
        kernel: str = "rbf",
        bandwidth: float = 1.0,
        epsilon: float = 1e-4,
    ) -> None:
        self.parameters = dict(
            c1=c1,
            c2=c2,
            kernel=kernel,
            bandwidth=bandwidth,
            epsilon=epsilon,
        )
        # Fail before fit so an invalid tree cannot be carried into training.
        BinaryLSTSVM(**self.parameters)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        class_order: tuple[int, ...] | None = None,
    ) -> BTLSTSVM:
        """Fit one-vs-remaining binary nodes in the declared class order."""
        feature_matrix = _features(x)
        labels = np.asarray(y)
        if labels.shape != (len(feature_matrix),) or not np.issubdtype(
            labels.dtype, np.integer
        ):
            raise ValueError("multiclass labels must be a one-dimensional integer vector")
        classes = tuple(int(item) for item in np.unique(labels))
        order = classes if class_order is None else tuple(class_order)
        if (
            len(classes) < 2
            or len(order) != len(classes)
            or set(order) != set(classes)
        ):
            raise ValueError("class_order must contain every training class exactly once")

        self._width = feature_matrix.shape[1]
        self.classes_ = order
        self.models_ = []

        # Each node peels off one class. The last class needs no model because
        # samples rejected by every earlier node belong to that residual class.
        for position, positive_class in enumerate(order[:-1]):
            active_rows = np.isin(labels, order[position:])
            binary_labels = np.where(
                labels[active_rows] == positive_class, 1, -1
            )
            self.models_.append(
                BinaryLSTSVM(**self.parameters).fit(
                    feature_matrix[active_rows], binary_labels
                )
            )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Route rows through the fitted tree, using the final class as fallback."""
        if not hasattr(self, "models_"):
            raise ValueError("model has not been fitted")
        feature_matrix = _features(x, width=self._width)
        predicted_labels = np.full(
            len(feature_matrix), self.classes_[-1], dtype=np.int64
        )
        unresolved_indices = np.arange(len(feature_matrix))
        for label, model in zip(self.classes_[:-1], self.models_):
            if not len(unresolved_indices):
                break
            decisions = model.predict(feature_matrix[unresolved_indices]) == 1
            predicted_labels[unresolved_indices[decisions]] = label
            unresolved_indices = unresolved_indices[~decisions]
        return predicted_labels


# Compatibility aliases for the historical W-spelling used by earlier releases.
BinaryLSTWSVM = BinaryLSTSVM
SequentialBTLSTWSVM = BTLSTSVM
