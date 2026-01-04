import numpy as np
from sklearn.ensemble import RandomForestClassifier

BG = 0
WM = 1
GM = 2
HYP = 3
AMY = 4
THAL = 5

LABELS_LARGE = [BG, WM, GM]
LABELS_SMALL = [BG, HYP, AMY, THAL]


class GroupedRandomForest:
    def __init__(self, n_estimators_large=50, n_estimators_small=50,
                 max_depth=20, random_state=42, max_features=None):

        self.forest_large = RandomForestClassifier(
            n_estimators=n_estimators_large,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state,
            max_features=max_features,
        )

        self.forest_small = RandomForestClassifier(
            n_estimators=n_estimators_small,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state,
            max_features=max_features,
        )

    def _make_labels_large(self, y):
        y_large = y.copy()
        y_large[~np.isin(y_large, LABELS_LARGE)] = BG
        return y_large

    def _make_labels_small(self, y):
        y_small = y.copy()
        y_small[~np.isin(y_small, LABELS_SMALL)] = BG
        return y_small

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.int16)

        self.forest_large.fit(X, self._make_labels_large(y))
        self.forest_small.fit(X, self._make_labels_small(y))

        self.classes_large_ = list(self.forest_large.classes_)
        self.classes_small_ = list(self.forest_small.classes_)
        return self

    def _safe_col(proba, classes, label, n):
        if label in classes:
            return proba[:, classes.index(label)].astype(np.float32)
        return np.zeros((n,), dtype=np.float32)
    
    def predict_proba(self, X):
        parts = self.predict_proba_parts(X)
        return parts["proba_final"]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba_parts(self, X):
        proba_large = self.forest_large.predict_proba(X)
        proba_small = self.forest_small.predict_proba(X)
        n = X.shape[0]

        p_wm = self._safe_col(proba_large, self.classes_large_, WM, n)
        p_gm = self._safe_col(proba_large, self.classes_large_, GM, n)

        rest = np.clip(1.0 - p_wm - p_gm, 0.0, 1.0)

        p_bg_s   = self._safe_col(proba_small, self.classes_small_, BG, n)
        p_hyp_s  = self._safe_col(proba_small, self.classes_small_, HYP, n)
        p_amy_s  = self._safe_col(proba_small, self.classes_small_, AMY, n)
        p_thal_s = self._safe_col(proba_small, self.classes_small_, THAL, n)

        denom = p_bg_s + p_hyp_s + p_amy_s + p_thal_s
        denom[denom == 0] = 1.0

        q_bg   = p_bg_s / denom
        q_hyp  = p_hyp_s / denom
        q_amy  = p_amy_s / denom
        q_thal = p_thal_s / denom

        proba_final = np.zeros((n, 6), dtype=np.float32)
        proba_final[:, WM]   = p_wm
        proba_final[:, GM]   = p_gm
        proba_final[:, BG]   = rest * q_bg
        proba_final[:, HYP]  = rest * q_hyp
        proba_final[:, AMY]  = rest * q_amy
        proba_final[:, THAL] = rest * q_thal

        row_sum = proba_final.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        proba_final /= row_sum

        return {
            "p_wm": p_wm,
            "p_gm": p_gm,
            "rest": rest,
            "q_bg": q_bg,
            "q_hyp": q_hyp,
            "q_amy": q_amy,
            "q_thal": q_thal,
            "proba_final": proba_final
        }
