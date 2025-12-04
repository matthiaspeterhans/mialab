import numpy as np
import pymia.data.conversion as conversion
import SimpleITK as sitk
from scipy.ndimage import binary_dilation
from sklearn.ensemble import RandomForestClassifier

import mialab.data.structure as structure

# Constants to align structure with other models
BG = 0
WM = 1
GM = 2
HYP = 3
AMY = 4
THAL = 5
FOREGROUND_LABELS = [WM, GM, HYP, AMY, THAL]
N_LABELS = 6

class PerLabelRF:
    """
    per-label random forest ensemble: one binary RF per foreground label.
    Methods: fit(X, y), predict(X), predict_proba(X)
    """
    def __init__(self, n_estimators=50, max_depth=20, random_state=42, n_jobs=-1, max_features=None, debug=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.max_features = max_features
        self.debug = debug
        self.models_ = {} 

    def fit(self, X, y):
        y = np.asarray(y).astype(np.int16)
        if self.debug:
            unique, counts = np.unique(y, return_counts=True)
            print("GT training distribution:", dict(zip(unique, counts)))

        self.models_.clear()
        for idx, label in enumerate(FOREGROUND_LABELS, start=1):
            y_binary = (y == label).astype(int)
            if self.debug:
                uniq_b, cnt_b = np.unique(y_binary, return_counts=True)
                print(f"Training distribution for label {label} vs rest:", dict(zip(uniq_b, cnt_b)))
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                n_jobs=self.n_jobs,
                random_state=self.random_state + idx,
                max_features=self.max_features
            )
            clf.fit(X, y_binary)
            self.models_[label] = clf
        return self

    def predict_proba(self, X):
        n = X.shape[0]
        # columns: [BG, WM, GM, HYP, AMY, THAL]
        proba = np.zeros((n, N_LABELS), dtype=np.float32)

        # Fill foreground probabilities from binary classifiers' positive class
        for label in FOREGROUND_LABELS:
            clf = self.models_.get(label, None)
            if clf is None:
                continue
            p = clf.predict_proba(X)[:, 1]
            proba[:, label] = p

        # BG probability set to 0, then normalize rows
        row_sum = proba.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        proba /= row_sum
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


