"""Shared folds with preprocessing and feature selection fitted on training data."""
from copy import deepcopy
from time import perf_counter
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from models.TCDFS import feature_selelction_base_on_tcdfs
from models.ATCDFS import feature_selelction_base_on_atcdfs
from tcd_knn import TCDKNNClassifier


def impute_fold(train, test, categorical):
    """Use training means only, preserving even entirely missing columns."""
    train, test = np.array(train, dtype=object, copy=True), np.array(test, dtype=object, copy=True)
    for column in range(train.shape[1]):
        if column in categorical:
            continue
        a, b = train[:, column].astype(float), test[:, column].astype(float)
        valid = np.isfinite(a)
        mean = float(a[valid].mean()) if valid.any() else 0.0
        train[:, column] = np.where(valid, a, mean)
        test[:, column] = np.where(np.isfinite(b), b, mean)
    return train, test


def preprocess_features(train, test, selected, categorical):
    """Fit encoding/scaling on selected training columns; transform the test fold."""
    selected = sorted(selected)
    numeric = [i for i in selected if i not in categorical]
    categories = [i for i in selected if i in categorical]
    train_parts, test_parts = [], []
    if numeric:
        scaler = MinMaxScaler()
        train_parts.append(scaler.fit_transform(train[:, numeric].astype(float)))
        test_parts.append(scaler.transform(test[:, numeric].astype(float)))
    if categories:
        encoder = OneHotEncoder(handle_unknown='ignore')
        train_parts.append(encoder.fit_transform(train[:, categories]).toarray())
        test_parts.append(encoder.transform(test[:, categories]).toarray())
    return np.column_stack(train_parts), np.column_stack(test_parts)


def classify(args, train, test, y_train, y_test):
    """Fit five classifiers on one training fold and score its held-out fold."""
    classifiers = {
        'SVM': SVC(), 'DT': DecisionTreeClassifier(random_state=0),
        'KNN3': KNeighborsClassifier(3, weights='distance'),
        'TCKNN3': TCDKNNClassifier(n_neighbors=3),
        'ATCKNN3': TCDKNNClassifier(n_neighbors=3, attribute_level=True),
    }
    scores = {}
    for name, model in classifiers.items():
        model.fit(train, y_train)
        scores[name] = float(np.mean(model.predict(test) == y_test))
    return scores


def evaluate_parameter(args, X, y, selector=None, evaluator=None):
    """Evaluate exactly one fixed k and summarize all ten folds."""
    X, y = np.asarray(X, dtype=object), np.asarray(y)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError('Expected samples-by-features X and one label per sample.')
    values, counts = np.unique(y, return_counts=True)
    if len(values) < 2 or len(y) < 10:
        raise ValueError('Ten-fold classification requires >=10 samples and >=2 classes.')
    if counts.min() < 10:
        args.logger.warning('Some classes have fewer than ten samples; some test folds will lack them.')
    folds = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed).split(X, y))
    k = args.tcdfs_k
    conditional_attribute_count = X.shape[1]
    if conditional_attribute_count == 0:
        raise ValueError('At least one conditional attribute is required.')
    min_train = min(len(train) for train, _ in folds)
    if min_train < 3 or (k < 1 or k > min_train):
        raise ValueError('k must be in [1, smallest training-fold size]; KNN needs >=3 training samples.')
    if any(len(np.unique(y[train])) < 2 for train, _ in folds):
        raise ValueError('Every training fold must contain at least two classes.')
    if selector is None:
        selector = {'tcdfs_k': feature_selelction_base_on_tcdfs,
                    'atcdfs_k': feature_selelction_base_on_atcdfs}[args.model]
    evaluator = evaluator or classify
    categorical = args.dataconf[args.dataname]['label_attributes_index'] or []
    results = {'protocol': 'fixed_parameter_stratified_10_fold_cv', 'seed': args.seed,
               'dataset': args.dataname, 'model': args.model, 'std_ddof': 0,
               'accuracy_unit': 'fraction', 'selection_rate_unit': 'fraction', 'parameters': []}
    args.logger.info('Protocol: fixed-parameter 10-fold CV; shared folds; std ddof=0; accuracy in [0,1].')
    args.logger.warning('Selecting best k using these CV scores still introduces selection bias. '
                        'Use nested CV or an independent test set for tuned performance.')
    fold_args = deepcopy(args)
    fold_args.tcdfs_k = k
    records = []
    args.logger.info('Parameter start: model=%s k=%d', args.model, k)
    for number, (train_idx, test_idx) in enumerate(folds, 1):
        train, test = impute_fold(X[train_idx], X[test_idx], categorical)
        started = perf_counter()
        selected = sorted(set(selector(fold_args, train.T.copy(), y[train_idx].tolist())))
        elapsed = perf_counter() - started
        if not selected:
            raise ValueError('Empty feature subset for k={}, fold={}; no folds omitted.'.format(k, number))
        if any(i < 0 or i >= X.shape[1] for i in selected):
            raise ValueError('Feature selector returned an invalid column index.')
        train_features, test_features = preprocess_features(train, test, selected, categorical)
        scores = evaluator(fold_args, train_features, test_features, y[train_idx], y[test_idx])
        records.append({'fold': number, 'train_indices': train_idx.tolist(),
                        'test_indices': test_idx.tolist(), 'selected_features': [int(i) for i in selected],
                        'selected_count': len(selected),
                        'selection_rate': len(selected) / conditional_attribute_count,
                        'selection_seconds': elapsed, 'accuracy': scores})
        args.logger.info('k=%d fold=%d/10 train=%d test=%d selected=%s selected_count=%d/%d selection_rate=%.8f selection_seconds=%.6f accuracy=%s',
                         k, number, len(train_idx), len(test_idx), selected, len(selected),
                         conditional_attribute_count, len(selected) / conditional_attribute_count, elapsed, scores)
    times = [r['selection_seconds'] for r in records]
    selected_count_mean = float(np.mean([r['selected_count'] for r in records]))
    selection_rate_mean = selected_count_mean / conditional_attribute_count
    summary = {'k': k, 'conditional_attribute_count': conditional_attribute_count,
               'selected_count_mean': selected_count_mean,
               'selection_rate_mean': selection_rate_mean, 'selection_seconds_mean': float(np.mean(times)),
               'selection_seconds_std': float(np.std(times)), 'accuracy': {}, 'folds': records}
    args.logger.info('k=%d Feature selection count mean:%.8f,conditional attributes:%d,'
                     'selection rate mean:%.8f (%.4f%%) (10 folds)',
                     k, selected_count_mean, conditional_attribute_count,
                     selection_rate_mean, 100 * selection_rate_mean)
    args.logger.info('k=%d Feature selection time mean:%.6f s,std:%.6f s',
                     k, summary['selection_seconds_mean'], summary['selection_seconds_std'])
    for name in records[0]['accuracy']:
        scores = [r['accuracy'][name] for r in records]
        summary['accuracy'][name] = {'mean': float(np.mean(scores)), 'std': float(np.std(scores))}
        args.logger.info('k=%d %s Accuracy mean:%.8f,std:%.8f (10 folds)',
                         k, name, np.mean(scores), np.std(scores))
    results['parameters'].append(summary)
    return results
