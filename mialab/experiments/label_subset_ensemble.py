import numpy as np
from sklearn.ensemble import RandomForestClassifier

BG = 0  # Background
WM = 1  # White Matter
GM = 2  # Gray Matter
HYP = 3  # Hippocampus
AMY = 4  # Amygdala
THAL = 5  # Thalamus

FOREGROUND_LABELS = [WM, GM, HYP, AMY, THAL]
RARE_LABELS = [HYP, AMY, THAL] #THAL
N_LABELS = 6


class LabelSubsetEnsembleRF:

    def __init__(self,
                 n_models=5,
                 n_estimators=50,
                 max_depth=20,
                 n_jobs=-1,
                 random_state=42,
                 max_features=None,
                 min_labels_per_model=2,
                 max_labels_per_model=4):
        self.n_models = n_models
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.max_features = max_features
        self.min_labels_per_model = min_labels_per_model
        self.max_labels_per_model = max_labels_per_model

        self.models_ = []    
        self.rng_ = np.random.RandomState(random_state)

    def _sample_label_subset(self):
        k = self.rng_.randint(self.min_labels_per_model,
                              self.max_labels_per_model + 1)
        n_rare = 2 if k >= 2 else 1
        rare_part = self.rng_.choice(RARE_LABELS, size=n_rare, replace=False)

        remaining_choices = [l for l in FOREGROUND_LABELS if l not in rare_part]
        rest = self.rng_.choice(remaining_choices, size=k - n_rare, replace=False)

        subset = np.concatenate((rare_part, rest))
        #subset = self.rng_.choice(FOREGROUND_LABELS, size=k, replace=False)
        allowed_labels = np.concatenate(([BG], subset))
        return allowed_labels

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.int16)
        self.models_ = []

        for m in range(self.n_models):
            allowed_labels = self._sample_label_subset()

            y_sub = y.copy()
            mask_allowed = np.isin(y_sub, allowed_labels)
            y_sub[~mask_allowed] = BG

            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                n_jobs=self.n_jobs,
                random_state=self.random_state + m, 
                max_features=self.max_features,
            )
            rf.fit(X, y_sub)

            self.models_.append((rf, allowed_labels))

        return self

    def predict(self, X):
        n = X.shape[0]
        votes = np.zeros((n, N_LABELS), dtype=np.int32)

        for rf, allowed_labels in self.models_:
            y_pred = rf.predict(X)
            votes[np.arange(n), y_pred] += 1

        final_labels = np.argmax(votes, axis=1)
        return final_labels

    def predict_proba(self, X):
        n = X.shape[0]
        votes = np.zeros((n, N_LABELS), dtype=np.int32)

        for rf, allowed_labels in self.models_:
            y_pred = rf.predict(X)
            votes[np.arange(n), y_pred] += 1

        proba = votes.astype(np.float32) / max(self.n_models, 1)

        row_sum = proba.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        proba /= row_sum

        return proba
