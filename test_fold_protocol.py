"""Regression tests for training-fold isolation, selected columns, and TCD queries."""
import logging
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from conf import parse_args
from classification import evaluate_parameter, impute_fold, preprocess_features
from datasets import data_minmax_stand, load_data
from tcd_knn import TCDKNNClassifier, minimax_closure, extend_query


def arguments(model='tcdfs_k'):
    return SimpleNamespace(model=model, tcdfs_k=2, seed=42,
                           dataname='test', logger=logging.getLogger('tests'),
                           dataconf={'test': {'label_attributes_index': []}})


class FoldProtocolTests(unittest.TestCase):
    def test_training_only_selection_shared_folds_and_statistics(self):
        args = arguments()
        X = np.column_stack([np.arange(40), np.arange(40) ** 2])
        y = np.array(['a', 'b'] * 20)
        calls, predictions = [], []

        def select(a, data, labels):
            ids = data[0].astype(int)
            self.assertEqual(len(ids), 36)
            np.testing.assert_array_equal(y[ids], labels)
            calls.append((a.tcdfs_k, ids))
            return [1]

        def evaluate(a, train, test, labels, test_labels):
            ids = calls[-1][1]
            held = np.setdiff1d(np.arange(40), ids)
            self.assertEqual(train.shape, (36, 1))
            self.assertEqual(test.shape, (4, 1))
            lo, hi = X[ids, 1].min(), X[ids, 1].max()
            np.testing.assert_allclose(test[:, 0], (X[held, 1] - lo) / (hi - lo))
            np.testing.assert_array_equal(test_labels, y[held])
            score = (len(predictions) % 10) / 10
            predictions.append(score)
            return {'test': score}

        result = evaluate_parameter(args, X, y, selector=select, evaluator=evaluate)
        self.assertEqual(len(calls), 10)
        self.assertEqual(len(result['parameters']), 1)
        self.assertEqual(result['parameters'][0]['k'], 2)
        args.tcdfs_k = 3
        second = evaluate_parameter(args, X, y, selector=select, evaluator=evaluate)
        self.assertEqual(len(second['parameters']), 1)
        self.assertEqual(second['parameters'][0]['k'], 3)
        self.assertEqual(len(calls), 20)
        for i in range(10):
            np.testing.assert_array_equal(calls[i][1], calls[i+10][1])
        for group in result['parameters']:
            held = [i for fold in group['folds'] for i in fold['test_indices']]
            self.assertEqual(sorted(held), list(range(40)))
            self.assertAlmostEqual(group['accuracy']['test']['mean'], .45)
            self.assertAlmostEqual(group['accuracy']['test']['std'], np.std(np.arange(10)/10))
            self.assertAlmostEqual(group['selection_seconds_mean'],
                                   np.mean([f['selection_seconds'] for f in group['folds']]))

    def test_selection_rate_uses_original_attributes_and_variable_subsets(self):
        args = arguments()
        args.dataconf['test']['label_attributes_index'] = [1]
        X = np.column_stack([np.arange(40), ['a', 'b', 'c', 'd'] * 10,
                             np.arange(40) ** 2]).astype(object)
        y = np.array([0, 1] * 20)
        selected_counts = []

        def select(*unused):
            selected = [1] if len(selected_counts) % 2 == 0 else [0, 1]
            selected_counts.append(len(selected))
            return selected

        with self.assertLogs('tests', level='INFO') as logs:
            result = evaluate_parameter(args, X, y, selector=select,
                                        evaluator=lambda *unused: {'test': .5})
        summary = result['parameters'][0]
        self.assertEqual(summary['conditional_attribute_count'], 3)
        self.assertEqual(summary['selected_count_mean'], 1.5)
        self.assertEqual(summary['selection_rate_mean'], .5)
        for fold, count in zip(summary['folds'], selected_counts):
            self.assertEqual(fold['selected_count'], count)
            self.assertAlmostEqual(fold['selection_rate'], count / 3)
        self.assertTrue(any('selection rate mean:0.50000000 (50.0000%)' in line
                            for line in logs.output))

    def test_cli_accepts_one_k_and_rejects_grid(self):
        args = parse_args(['--tcdfs_k', '5'])
        self.assertEqual(args.tcdfs_k, 5)
        self.assertFalse(hasattr(args, 'tcdfs_k_grid'))
        with patch('sys.stderr', new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as error:
                parse_args(['--tcdfs_k_grid', '2', '3'])
        self.assertEqual(error.exception.code, 2)

    def test_imputation_scaling_and_unknown_categories(self):
        train = np.array([[0, 'a', np.nan], [2, 'b', np.nan], [np.nan, 'a', np.nan]], dtype=object)
        test = np.array([[100, 'new', 50], [np.nan, 'b', np.nan]], dtype=object)
        train, test = impute_fold(train, test, [1])
        self.assertEqual(train[2, 0], 1)
        self.assertEqual(test[1, 0], 1)
        self.assertEqual(test[1, 2], 0)
        a, b = preprocess_features(train, test, [0, 1], [1])
        self.assertEqual(a.shape[1], 3)
        self.assertEqual(b[0, 0], 50)
        np.testing.assert_array_equal(b[0, 1:], [0, 0])
        numeric, _ = preprocess_features(train, test, [0], [1])
        self.assertEqual(numeric.shape[1], 1)
        categorical, _ = preprocess_features(train, test, [1], [1])
        self.assertEqual(categorical.shape[1], 2)

    def test_single_numeric_attribute_keeps_original_position(self):
        args = arguments()
        args.dataconf['test']['label_attributes_index'] = [0]
        data = pd.DataFrame([['a', 'b', 'a'], [2, 4, 6]])
        scaled = data_minmax_stand(args, data)
        np.testing.assert_array_equal(scaled.iloc[0], ['a', 'b', 'a'])
        np.testing.assert_allclose(scaled.iloc[1].astype(float), [0, .5, 1])

    def test_loader_does_not_impute_globally(self):
        args = arguments()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'missing.data'
            path.write_text('id1,?,a\nid2,100,b\n', encoding='utf-8')
            args.dataconf['test'].update(filepath=str(path), seperator=',', header=None,
                                        remove_attribute_index=[0], label_index=2, ori_lab_condattri_idx=[])
            X, y = load_data(args)
            self.assertTrue(np.isnan(X.iloc[0, 0]))
            self.assertEqual(X.iloc[0, 1], 100)
            self.assertEqual(y.tolist(), ['a', 'b'])

    def test_invalid_k_and_empty_subset_fail_without_skipping_folds(self):
        args = arguments()
        X, y = np.arange(40).reshape(20, 2), np.array([0, 1]*10)
        args.tcdfs_k = 19
        with self.assertRaises(ValueError):
            evaluate_parameter(args, X, y)
        args.tcdfs_k = 2
        with self.assertRaisesRegex(ValueError, 'Empty feature subset'):
            evaluate_parameter(args, X, y, selector=lambda *a: [])

    def test_actual_models_on_mixed_data(self):
        rng = np.random.RandomState(3)
        X = np.empty((30, 3), dtype=object)
        X[:, 0] = rng.rand(30)
        X[:, 1] = ['a', 'b'] * 15
        X[:, 2] = rng.rand(30)
        X[0, 2] = np.nan
        y = np.array([0, 1] * 15)
        for model in ['tcdfs_k', 'atcdfs_k']:
            args = arguments(model)
            args.tcdfs_k = 2
            args.dataconf['test']['label_attributes_index'] = [1]
            result = evaluate_parameter(args, X, y)
            group = result['parameters'][0]
            self.assertEqual(len(group['folds']), 10)
            self.assertEqual(len(group['accuracy']), 5)
            for scores in group['accuracy'].values():
                self.assertTrue(0 <= scores['mean'] <= 1)


class TCDTests(unittest.TestCase):
    def test_query_extension_matches_individual_augmented_closure(self):
        train = np.array([[0.], [.2], [.7], [1.]])
        distances = np.abs(train[:, 0, None] - train[:, 0])
        closure = minimax_closure(distances)
        edges = np.abs(train[:, 0] - .5)
        augmented = np.zeros((5, 5))
        augmented[:4, :4] = distances
        augmented[4, :4] = augmented[:4, 4] = edges
        np.testing.assert_allclose(extend_query(edges, closure), minimax_closure(augmented)[4, :4])

    def test_predictions_independent_of_other_test_samples(self):
        train = np.array([[0., 0.], [.2, 0.], [.7, 0.], [1., 0.]])
        y = [0, 0, 1, 1]
        for mode in [False, True]:
            model = TCDKNNClassifier(3, mode).fit(train, y)
            single = model.query_distances([[.5, 1.]])
            batch = model.query_distances([[.5, 1.], [.3, 0.], [.6, 0.]])
            np.testing.assert_allclose(single[0], batch[0])
            self.assertTrue(np.isfinite(batch).all())
            self.assertEqual(model.predict([[.5, 1.]])[0], model.predict([[.5, 1.], [.3, 0.]])[0])
            np.testing.assert_array_equal(model.X_train_, train)


if __name__ == '__main__':
    unittest.main()
