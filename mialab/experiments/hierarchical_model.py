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

LARGE_LABELS = {1, 2, 5} # WM, GM, Thalamus
SMALL_LABELS = {3, 4} # Hippocampus, Amygdala


#  Stage 1: Train Random Forest on ALL tissues (baseline-style)
def train_large_rf(images, n_estimators=50, max_depth=20, debug: bool = False):
    """
    Stage-1 RF: train exactly like the baseline multi-class model.
    It learns all labels 0,1,2,3,4,5.
    """
    X = []
    y = []

    for img in images:
        X.append(img.feature_matrix[0]) # features
        y.append(img.feature_matrix[1].squeeze()) # GT labels
    X = np.concatenate(X)
    y = np.concatenate(y)

    if debug:
        unique, counts = np.unique(y, return_counts=True)
        print("Stage1 GT training distribution:", dict(zip(unique, counts)))
        # Output should be something like {0:...,1:...,2:...,3:...,4:...,5:...}

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X, y)
    return model


#  Stage 2: Train Random Forest for small tissues (3,4) vs background
def train_small_rf(images, large_predictions, n_estimators=50, max_depth=20, debug: bool = False):
    """
    Stage-2 RF: binary-ish classifier in 3-way label space {0,3,4}:
        - 3,4 stay as 3,4 (small tissues)
        - everything else becomes 0 (background)
    We do NOT use an ROI here to avoid accidentally dropping all small labels.
    The ROI is only applied at PREDICTION time when fusing with Stage-1.
    """
    X = []
    y = []

    for img in images:
        xi = img.feature_matrix[0]
        yi = img.feature_matrix[1].squeeze()
        # Map labels: keep 3/4, everything else → 0
        yi_mod = np.where(np.isin(yi, list(SMALL_LABELS)), yi, 0)

        X.append(xi)
        y.append(yi_mod)

    X = np.concatenate(X)
    y = np.concatenate(y)

    if len(X) == 0:
        raise RuntimeError("Stage-2 SMALL RF: no training samples at all.")

    if debug:
        # Debug: Stage-2 label distribution (we WANT 0,3,4 here)
        unique, counts = np.unique(y, return_counts=True)
        print("[Stage-2] small-tissue training distribution:", dict(zip(unique, counts)))
        # Example: {0: 13000, 3: 500, 4: 200}

    # Class weights: emphasize small tissues 3 & 4
    #class_weight = {0: 1.0, 3: 5.0, 4: 5.0}
    class_weight = {0: 1.0, 3: 10.0, 4: 10.0} # increases sensitivity to small tissues

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        class_weight=class_weight
    )
    model.fit(X, y)

    if debug:
        print("[Stage-2] classes_ in trained RF:", model.classes_)  # should be [0,3,4]

    return model


#  Stage 1 Prediction
def predict_large(model, img, debug: bool = False):
    """
    Predict full multi-class segmentation with Stage-1 RF.
    """
    pred = model.predict(img.feature_matrix[0])
    if debug:
        unique, counts = np.unique(pred, return_counts=True)
        print("Stage1 distribution:", dict(zip(unique, counts)))
    return pred


#  Stage 2 Prediction (inside ROI)
def predict_small(model, img, pred_large, debug: bool = False, force_manual_roi: bool = False):
    """
    Stage-2 prediction:
      - uses RF trained on labels {0,3,4}
      - predicts for all voxels
      - BUT we will only allow replacing Stage-1 labels inside ROI during fusion.

    ROI options:
      - If force_manual_roi=True: a big circle around the image center (debug only)
      - Else: ROI = voxels that Stage-1 predicted as GM or Thalamus (2 or 5)
              (these are the regions where Hippocampus/Amygdala usually live).
    """
    n_vox = pred_large.shape[0]

    if force_manual_roi:
        # ---- Manual debug ROI: big 2D circle repeated over all slices ----
        Xs, Ys, Zs = img.image_properties.size  # (X,Y,Z)
        cx, cy = Xs // 2, Ys // 2
        radius = min(Xs, Ys) // 4

        coords = np.indices((Xs, Ys))
        xx, yy = coords[0], coords[1]

        circular_mask_2d = (xx - cx) ** 2 + (yy - cy) ** 2 < radius ** 2
        roi_mask_3d = np.repeat(circular_mask_2d[..., None], Zs, axis=2)
        roi_mask = roi_mask_3d.flatten()

        # Make sure ROI length matches pred_large length; if not, trim
        if roi_mask.shape[0] > n_vox:
            roi_mask = roi_mask[:n_vox]
        elif roi_mask.shape[0] < n_vox:
            pad_len = n_vox - roi_mask.shape[0]
            roi_mask = np.concatenate([roi_mask, np.zeros(pad_len, dtype=bool)])

        if debug:
            print("Manual ROI size:", int(np.sum(roi_mask)))
    else:
        # Normal ROI: where Stage-1 thinks it's GM or Thalamus
        roi_mask = np.isin(pred_large, [1, 2, 5])
        #roi_mask = binary_dilation(roi_mask, iterations=2).flatten()
        roi_mask_3d = roi_mask.reshape(img.image_properties.size[::-1])
        roi_mask_3d = binary_dilation(roi_mask_3d, iterations=2)
        roi_mask = roi_mask_3d.flatten()

        if debug:
            print("Normal ROI size (Stage-1 GM/Thal):", int(np.sum(roi_mask)))

    # Stage-2 predictions & probabilities for all voxels
    probs = model.predict_proba(img.feature_matrix[0])   # shape: [N, num_classes]
    pred_small = model.predict(img.feature_matrix[0])    # argmax labels: 0/3/4

    classes = model.classes_  # e.g. [0, 3, 4] or maybe [0,3] etc.

    # Find which probability columns correspond to 3 & 4
    small_class_indices = []
    for lab in SMALL_LABELS:
        idx = np.where(classes == lab)[0]
        if idx.size > 0:
            small_class_indices.append(idx[0])

    if len(small_class_indices) == 0:
        # Worst case: RF never saw labels 3 or 4.
        # Then Stage-2 never predicts small tissue; treat confidence as 0.
        small_conf = np.zeros(pred_small.shape[0], dtype=float)
        if debug:
            print("[Stage-2] WARNING: no classes 3/4 in model.classes_.")
    else:
        # Max probability over the small-tissue classes
        small_conf = np.max(probs[:, small_class_indices], axis=1)

    if debug:
        if np.sum(roi_mask) > 0:
            print("Stage2 mean confidence inside ROI:",
                  float(np.mean(small_conf[roi_mask])))
        else:
            print("Stage2 mean confidence inside ROI: ROI empty!")

        uniq2, cnt2 = np.unique(pred_small, return_counts=True)
        print("Stage2 prediction label distribution:", dict(zip(uniq2, cnt2)))

    return pred_small, small_conf, roi_mask


#  Fuse predictions from large + small stage
def fuse_predictions(pred_large, pred_small, small_conf, roi_mask, debug: bool = False,
                     conf_threshold: float = 0.005):
    """
    Fusion rule:
      - Start from Stage-1 prediction (full multi-class).
      - Inside ROI, if Stage-2 is confident enough (small_conf > threshold),
        overwrite Stage-1 with Stage-2 prediction.
      - Outside ROI, keep Stage-1 as is.

    This way:
      - Large structures (1,2,5) are primarily Stage-1’s job.
      - Small structures (3,4) can be refined by Stage-2 in a focused region.
    """
    final = pred_large.copy()

    # Replace only where:
    #  - Stage-2 confidence is high enough
    #  - AND inside ROI
    #  - AND Stage-2 actually predicts a small tissue (3 or 4)
    mask_replace = (small_conf > conf_threshold) & roi_mask & np.isin(pred_small, list(SMALL_LABELS))
    final[mask_replace] = pred_small[mask_replace]

    if debug:
        print("Voxels replaced by Stage2:", int(np.sum(mask_replace)))
        uniq_f, cnt_f = np.unique(final, return_counts=True)
        print("Final label distribution after fusion:", dict(zip(uniq_f, cnt_f)))

    return final


#  Convert numpy prediction to SimpleITK image
def convert_to_image(pred, img_properties):
    return conversion.NumpySimpleITKImageBridge.convert(pred.astype(np.uint8), img_properties)