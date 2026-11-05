"""Inductive TCD distances: only training samples may be intermediate vertices."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances


def minimax_closure(distances):
    """Compute the min-max closure using Floyd-Warshall updates."""
    result = np.array(distances, dtype=float, copy=True)
    for k in range(len(result)):
        result = np.minimum(result, np.maximum(result[:, k, None], result[None, k, :]))
    return result


def extend_query(edges, closure):
    """Return min_p max(d(query,p), closure[p,j]) for training endpoints j."""
    return np.maximum(np.asarray(edges)[:, None], closure).min(axis=0)


class TCDKNNClassifier:
    """Uniform-vote KNN using a frozen training closure and independent test queries."""
    def __init__(self, n_neighbors=3, attribute_level=False):
        self.n_neighbors = n_neighbors
        self.attribute_level = attribute_level

    def fit(self, X, y):
        self.X_train_ = np.array(X, dtype=float, copy=True)
        if self.attribute_level:
            self.closures_, self.scales_ = [], []
            squared = np.zeros((len(X), len(X)))
            for column in self.X_train_.T:
                closure = minimax_closure(np.abs(column[:, None] - column))
                # A constant training attribute uses scale 1, avoiding division by zero.
                scale = float(closure.max()) or 1.0
                self.closures_.append(closure)
                self.scales_.append(scale)
                squared += (closure / scale) ** 2
            distances = np.sqrt(squared)
        else:
            self.closure_ = minimax_closure(pairwise_distances(self.X_train_))
            distances = self.closure_
        self.knn_ = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric='precomputed')
        self.knn_.fit(distances, y)
        return self

    def query_distances(self, X):
        """No query can change training distances or another query's prediction."""
        result = []
        for query in np.asarray(X, dtype=float):
            if self.attribute_level:
                squared = np.zeros(len(self.X_train_))
                for index, (closure, scale) in enumerate(zip(self.closures_, self.scales_)):
                    edges = np.abs(self.X_train_[:, index] - query[index])
                    squared += (extend_query(edges, closure) / scale) ** 2
                result.append(np.sqrt(squared))
            else:
                edges = np.linalg.norm(self.X_train_ - query, axis=1)
                result.append(extend_query(edges, self.closure_))
        return np.asarray(result)

    def predict(self, X):
        return self.knn_.predict(self.query_distances(X))


class FoldTCDCrossValidator:
    """Classifier-only compatibility helper; use run.py for fold-local feature selection."""
    attribute_level = False

    def __init__(self, args, attributions, labels, n_neighbors=5):
        self.args = args
        self.X = np.asarray(attributions)
        self.labels = np.asarray(labels)
        self.n_neighbors = n_neighbors

    def cross_validate(self, n_splits=10, random_state=42):
        from sklearn.model_selection import StratifiedKFold
        from classification import impute_fold, preprocess_features
        categorical = self.args.dataconf[self.args.dataname]['label_attributes_index'] or []
        scores = []
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_idx, test_idx in cv.split(self.X, self.labels):
            train, test = impute_fold(self.X[train_idx], self.X[test_idx], categorical)
            train, test = preprocess_features(train, test, range(self.X.shape[1]), categorical)
            model = TCDKNNClassifier(self.n_neighbors, self.attribute_level)
            model.fit(train, self.labels[train_idx])
            scores.append(np.mean(model.predict(test) == self.labels[test_idx]))
        return float(np.mean(scores)), float(np.std(scores))
