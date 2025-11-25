import numpy as np
from sklearn.ensemble import RandomForestClassifier

BG = 0  # Background label
WM = 1  # White Matter label
GM = 2  # Gray Matter label
HYP = 3  # Hippocampus label
AMY = 4  # Amygdala label
THAL = 5  # Thalamus label

LABELS_LARGE = [BG, WM, GM, THAL]
LABELS_SMALL = [BG, HYP, AMY]

class GroupedRandomForest:
    def __init__(self, n_estimators_large=50, n_estimators_small=50, max_depth=20, random_state = 42, max_features=None):
        self.forest_large = RandomForestClassifier(
            n_estimators=n_estimators_large,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state,
            max_features=max_features, 
            class_weight='balanced'
        )
        self.forest_small = RandomForestClassifier(
            n_estimators=n_estimators_small,
            max_depth=10,
            n_jobs=-1,
            random_state=random_state,
            max_features=max_features,
            class_weight='balanced'
        )

    def _make_labels_large(self, y):
        y_large = y.copy()
        mask_large = np.isin(y_large, LABELS_LARGE)
        y_large[~mask_large] = BG
        return y_large
    
    def _make_labels_small(self, y):
        y_small = y.copy()
        mask_small = np.isin(y_small, LABELS_SMALL)
        y_small[~mask_small] = BG
        return y_small
    
    def fit(self, X, y):
        y = np.asarray(y, dtype=np.int16)

        y_large = self._make_labels_large(y)
        y_small = self._make_labels_small(y)

        self.forest_large.fit(X, y_large)
        self.forest_small.fit(X, y_small)

        self.classes_large_ = list(self.forest_large.classes_)  
        self.classes_small_ = list(self.forest_small.classes_) 
        return self
    
    def predict_proba(self, X):
        proba_large = self.forest_large.predict_proba(X)
        proba_small = self.forest_small.predict_proba(X)

        n = X.shape[0]
        proba_final = np.zeros((n, 6), dtype=np.float32)
        
        for label in [BG, WM, GM, THAL]:
            if label in self.classes_large_:
                idx = self.classes_large_.index(label)
                proba_final[:, label] = proba_large[:, idx]
                
        for label in [HYP, AMY]:
            if label in self.classes_small_:
                idx = self.classes_small_.index(label)
                proba_final[:, label] = proba_small[:, idx]
                
        row_sum = proba_final.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        proba_final /= row_sum
        return proba_final
        
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)