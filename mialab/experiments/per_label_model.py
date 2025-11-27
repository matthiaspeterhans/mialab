import numpy as np
import pymia.data.conversion as conversion
import SimpleITK as sitk
from scipy.ndimage import binary_dilation
from sklearn.ensemble import RandomForestClassifier

import mialab.data.structure as structure

# 1: 'WhiteMatter'
# 2: 'GreyMatter'
# 3: 'Hippocampus'
# 4: 'Amygdala'
# 5: 'Thalamus'

def train_all(images, n_estimators=50, max_depth=20, debug: bool = False):

    ''' One model per label (binary classifiers: label vs rest)'''

    X = []
    y = []

    for img in images:
        X.append(img.feature_matrix[0]) # features
        y.append(img.feature_matrix[1].squeeze()) # GT labels
    X = np.concatenate(X)
    y = np.concatenate(y)
    if debug:
        unique, counts = np.unique(y, return_counts=True)
        print("GT training distribution:", dict(zip(unique, counts)))

    models = {}
    for label in range(1,6): # labels 1 to 5
        # Create binary labels for current label vs rest
        y_binary = (y == label).astype(int)

        if debug:
            unique, counts = np.unique(y_binary, return_counts=True)
            print(f"Training distribution for label {label} vs rest:", dict(zip(unique, counts)))

        model = RandomForestClassifier(
            max_features=images[0].feature_matrix[0].shape[1],
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X, y_binary)
        models[label] = model

    return models

# Prediction function for per-label models
def predict_all(models, img, debug: bool = False):
    """
    Predict using the per-label models.
    For each label, get the probability map from the corresponding model.
    Then assign the label with the highest probability at each voxel.
    """

    X = img.feature_matrix[0]  # features
    shape = sitk.GetArrayFromImage(img.images[structure.BrainImageTypes.T1w]).shape
    
    prob_maps = []

    for label, clf in models.items():
        probs = clf.predict_proba(X)[:, 1]  # probability of the positive class
        prob_map = probs.reshape(shape)
        prob_maps.append(prob_map)

    prob_maps = np.stack(prob_maps, axis=-1)  # shape: (H, W, D, num_labels)
    predicted_labels = np.argmax(prob_maps, axis=-1) + 1  # +1 to match label indexing

    if debug:
        print("Predicted labels shape:", predicted_labels.shape)

    return predicted_labels
