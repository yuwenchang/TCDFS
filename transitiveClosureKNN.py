"""Compatibility entry point for classifier-only TCKNN cross-validation.

Use run.py to repeat feature selection inside every training fold.
Input features must be raw, samples-by-features data; categorical indices
must refer to these input columns.
"""
from tcd_knn import FoldTCDCrossValidator


class TCKNNCrossValidator(FoldTCDCrossValidator):
    attribute_level = False
