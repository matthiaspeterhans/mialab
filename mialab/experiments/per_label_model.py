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
        # To avoid spamming on every test case, print per-image debug only for first K images
        self.debug_max_images = 2
        self._debug_image_counter = 0
        self.models_ = {} 

    def fit(self, X, y):
        y = np.asarray(y).astype(np.int16)
        if self.debug:
            unique, counts = np.unique(y, return_counts=True)
            cnt = dict(zip(unique.tolist(), counts.tolist()))
            print("GT training distribution:", dict(zip(unique, counts)))
            print("[PerLabelRF][fit] GT training distribution (multi-class):", cnt)
            total = len(y)
            fg = np.mean(y != BG) * 100.0
            print(f"[PerLabelRF][fit] Foreground fraction: {fg:.2f}% ({int(np.sum(y!=BG))}/{total})")

        self.models_.clear()
        for idx, label in enumerate(FOREGROUND_LABELS, start=1):
            y_binary = (y == label).astype(int)
            if self.debug:
                uniq_b, cnt_b = np.unique(y_binary, return_counts=True)
                print(f"Training distribution for label {label} vs rest:", dict(zip(uniq_b, cnt_b)))
                n_pos = int(np.sum(y_binary == 1))
                n_neg = int(np.sum(y_binary == 0))
                print(f"[PerLabelRF][fit] Label {label} vs rest: pos={n_pos}, neg={n_neg}, "
                      f"pos%={100.0*n_pos/max(1,(n_pos+n_neg)):.4f}%")
                # A simple imbalance warning (useful for presentation)
                if n_pos > 0:
                    ratio = n_neg / n_pos
                    if ratio > 500:
                        print(f"[PerLabelRF][fit]  WARNING: extreme imbalance for label {label}: "
                              f"neg/pos ≈ {ratio:.1f}")
            if label in [HYP, AMY]:
                cw = {0: 1.0, 1: 10.0}  # pos = label
            else:
                cw = "balanced"
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                n_jobs=self.n_jobs,
                random_state=self.random_state + idx,
                max_features=self.max_features,
                class_weight=cw,
            )
            clf.fit(X, y_binary)
            self.models_[label] = clf
        if self.debug:
            print(f"[PerLabelRF][fit] Trained {len(self.models_)} binary RFs (labels={FOREGROUND_LABELS}).")
            print("[PerLabelRF][fit] Fusion note: BG prob is 0 before normalization; "
                  "BG only wins if all foreground probs are ~0 for a voxel.")
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
            p = clf.predict_proba(X)[:, 1].astype(np.float32)
            proba[:, label] = p

        # BG probability set to 0, then normalize rows
        row_sum = proba.sum(axis=1, keepdims=True)
        zero_rows = (row_sum[:, 0] <= 1e-8)
        n_zero = int(np.sum(zero_rows))
        row_sum[row_sum == 0] = 1.0
        proba /= row_sum
        if self.debug and self._debug_image_counter < self.debug_max_images:
            self._debug_image_counter += 1

            maxp = np.max(proba, axis=1)
            argm = np.argmax(proba, axis=1)
            unique, counts = np.unique(argm, return_counts=True)
            pred_dist = dict(zip(unique.tolist(), counts.tolist()))

            print(f"[PerLabelRF][predict_proba] Voxels={n}")
            print(f"[PerLabelRF][predict_proba] Foreground-sum~0 voxels: {n_zero} "
                  f"({100.0*n_zero/max(1,n):.4f}%) -> these default to BG via argmax")
            print(f"[PerLabelRF][predict_proba] max prob stats: "
                  f"min={float(np.min(maxp)):.3f}, mean={float(np.mean(maxp)):.3f}, max={float(np.max(maxp)):.3f}")
            print(f"[PerLabelRF][predict_proba] predicted label distribution (after fusion argmax): {pred_dist}")
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
