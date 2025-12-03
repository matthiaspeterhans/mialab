import numpy as np
import pymia.data.conversion as conversion
import SimpleITK as sitk
from scipy.ndimage import binary_dilation
from sklearn.ensemble import RandomForestClassifier

import mialab.data.structure as structure

# Label mapping (for readability)
# 0: background
# 1: WhiteMatter
# 2: GreyMatter
# 3: Hippocampus
# 4: Amygdala
# 5: Thalamus

LARGE_LABELS = {1, 2, 5}  # WM, GM, Thalamus
SMALL_LABELS = {3, 4}      # Hippocampus, Amygdala


#  Helper: build ROI from Stage-1 predictions
def build_stage1_roi(pred_large, img, include_small_labels=True, dilation_iters=2, debug=False):
    """
    Build a loose ROI in *atlas space* from Stage-1 predictions.

    Args:
        pred_large (np.ndarray): 1D array of Stage-1 predicted labels for each voxel.
        img (structure.BrainImage): Image object (for shape info).
        include_small_labels (bool): Whether to include labels 3 and 4 in the seed.
        dilation_iters (int): Number of binary dilation iterations in voxel units.
        debug (bool): Print stats.

    Returns:
        np.ndarray (bool): 1D ROI mask (True = inside ROI).
    """
    if include_small_labels:
        seed_labels = [1, 2, 3, 4, 5]
    else:
        seed_labels = [1, 2, 5]

    roi_mask = np.isin(pred_large, seed_labels)

    # Reshape to 3D, apply dilation, then flatten back
    roi_mask_3d = roi_mask.reshape(img.image_properties.size[::-1])  # (z, y, x)
    roi_mask_3d = binary_dilation(roi_mask_3d, iterations=dilation_iters)
    roi_mask = roi_mask_3d.flatten()

    if debug:
        print(f"[ROI] Stage-1 ROI size: {int(np.sum(roi_mask))} voxels "
              f"({np.mean(roi_mask) * 100:.2f}% of image)")

    return roi_mask


def build_center_disk_roi(img, radius_fraction=0.4, debug=False):
    """
    Manual / heuristic ROI: a big disk from the image center in (x,y),
    repeated across all slices. Mainly useful as a debugging baseline.

    Args:
        img (structure.BrainImage): Image object.
        radius_fraction (float): Disk radius fraction of min(x,y).
        debug (bool): Print stats.

    Returns:
        np.ndarray (bool): 1D ROI mask.
    """
    # Image size in SimpleITK order: (x, y, z)
    Xs, Ys, Zs = img.image_properties.size
    cx, cy = Xs // 2, Ys // 2
    radius = int(min(Xs, Ys) * radius_fraction)

    coords = np.indices((Xs, Ys))
    xx, yy = coords[0], coords[1]
    circular_mask_2d = (xx - cx) ** 2 + (yy - cy) ** 2 < radius ** 2

    # Repeat across z
    roi_mask_3d = np.repeat(circular_mask_2d[..., None], Zs, axis=2)  # (x,y,z)
    # Convert to (z,y,x) order when flattening
    roi_mask_3d = np.transpose(roi_mask_3d, (2, 1, 0))  # (z,y,x)
    roi_mask = roi_mask_3d.flatten()

    if debug:
        print(f"[ROI] Center-disk ROI size: {int(np.sum(roi_mask))} voxels "
              f"({np.mean(roi_mask) * 100:.2f}% of image)")

    return roi_mask


#  Stage 1: Train Random Forest on ALL tissues (baseline-style)
def train_large_rf(images,
                   n_estimators=50,
                   max_depth=20,
                   max_features=None,
                   seed=42,
                   debug=False):
    """
    Stage-1 RF: train exactly like the baseline multi-class model.
    It learns all labels 0,1,2,3,4,5.

    Args:
        images (List[BrainImage]): Training images with feature_matrix filled.
        n_estimators (int): Number of trees.
        max_depth (int): Max tree depth.
        max_features: RF max_features parameter (e.g. n_features or 'sqrt').
        seed (int): Random seed.
        debug (bool): Debug prints.

    Returns:
        RandomForestClassifier: Stage-1 trained model.
    """
    X_list = []
    y_list = []

    for img in images:
        X_list.append(img.feature_matrix[0])
        y_list.append(img.feature_matrix[1].squeeze())

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    if debug:
        unique, counts = np.unique(y, return_counts=True)
        print("[Stage-1] GT training distribution:", dict(zip(unique, counts)))

    if max_features is None:
        max_features = X.shape[1]

    model = RandomForestClassifier(
        max_features=max_features,
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=seed
    )
    model.fit(X, y)

    if debug:
        print("[Stage-1] classes_:", model.classes_)

    return model



#  Helper: Build Stage-2 features with Stage-1 probabilities (auto-context)
def _augment_features_with_stage1_probs(X, probs_large, classes_large, debug=False):
    """
    Concatenate Stage-1 probabilities to original features.

    Args:
        X (np.ndarray): Original feature matrix [N, F].
        probs_large (np.ndarray): Stage-1 probabilities [N, C].
        classes_large (np.ndarray): Stage-1 classes_ array [C].
        debug (bool): Debug prints.

    Returns:
        np.ndarray: Augmented feature matrix [N, F + C_sel].
    """
    # Here we simply use probabilities for *all* Stage-1 classes.
    # If you want only a subset (e.g. 2,3,4,5), you can filter cols here.
    # Sanity checks
    if X.shape[0] != probs_large.shape[0]:
        raise RuntimeError("Mismatch between X and Stage-1 probs rows.")

    X_aug = np.concatenate([X, probs_large], axis=1)

    if debug:
        print(f"[Stage-2] Augmented features: original F={X.shape[1]}, "
              f"Stage-1 prob dims={probs_large.shape[1]}, total={X_aug.shape[1]}")
        print(f"[Stage-2] Stage-1 classes in probs_large: {classes_large}")

    return X_aug


#  Stage 2: Train Random Forest for small tissues (3,4) vs background
def train_small_rf(images,
                   model_large,
                   n_estimators=50,
                   max_depth=20,
                   max_features=None,
                   seed=42,
                   use_spatial_downsampling=True,
                   debug=False):
    """
    Stage-2 RF: 3-way label space {0,3,4}:
        - 3,4 stay as 3,4 (small tissues)
        - everything else becomes 0 (background)

    We *train* on:
        - all small-structure voxels (labels 3,4),
        - plus background voxels that Stage 1 considers "brain-ish"
          (predicted as 1,2,3,4,5) if use_spatial_downsampling=True,
        - else: all sampled voxels.

    NOTE:
      For training images, feature_matrix contains ONLY sampled voxels
      (due to RandomizedTrainingMaskGenerator). Therefore we cannot reshape
      predictions to a full 3D volume. ROI is defined in sample-index space.
    """
    X_all = []
    y_all = []

    classes_large = model_large.classes_

    for img_idx, img in enumerate(images):
        X_orig = img.feature_matrix[0] # [N_samp, F]
        y_orig = img.feature_matrix[1].squeeze() # [N_samp]
        # Map labels: keep 3/4, everything else → 0
        y_mod = np.where(np.isin(y_orig, list(SMALL_LABELS)), y_orig, 0)

        # Stage-1 probabilities on this (sampled) feature set
        probs_large = model_large.predict_proba(X_orig)

        if debug and img_idx == 0:
            print("[Stage-2 train] X_orig shape:", X_orig.shape)
            print("[Stage-2 train] y_orig shape:", y_orig.shape)
            print("[Stage-2 train] image size (x,y,z):", img.image_properties.size)
            print("[Stage-2 train] full voxel count:", np.prod(img.image_properties.size))

        # Augment features with Stage-1 probabilities (auto-context)
        X_aug = _augment_features_with_stage1_probs(
            X_orig, probs_large, classes_large,
            debug=(debug and img_idx == 0)
        )

        if use_spatial_downsampling:
            # Stage-1 predictions for these *sampled* voxels
            pred_large_img = model_large.predict(X_orig)

            # Define a 1D ROI mask in sample index space:
            # keep voxels Stage-1 thinks are non-background (1,2,3,4,5)
            roi_mask = np.isin(pred_large_img, [1, 2, 3, 4, 5])

            if debug and img_idx == 0:
                print(f"[Stage-2 train] Sampled voxels: {len(pred_large_img)}")
                print(f"[Stage-2 train] ROI (non-bg) voxels: "
                      f"{np.sum(roi_mask)} ({np.mean(roi_mask) * 100:.2f}%)")

            keep_mask = (y_mod != 0) | roi_mask # keep all small (3,4) + tissue-like negatives
            X_use = X_aug[keep_mask]
            y_use = y_mod[keep_mask]
        else:
            X_use = X_aug
            y_use = y_mod

        X_all.append(X_use)
        y_all.append(y_use)

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)

    if len(X) == 0:
        raise RuntimeError("Stage-2 SMALL RF: no training samples at all.")

    if debug:
        unique, counts = np.unique(y, return_counts=True)
        print("[Stage-2] small-tissue training distribution (0,3,4):",
              dict(zip(unique, counts)))

    if max_features is None:
        max_features = X.shape[1]

    # Emphasize small structures
    class_weight = {0: 1.0, 3: 10.0, 4: 10.0}

    model_small = RandomForestClassifier(
        max_features=max_features,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
        class_weight=class_weight
    )
    model_small.fit(X, y)

    if debug:
        print("[Stage-2] classes_ in trained RF:", model_small.classes_)

    return model_small



#  Stage 1 Prediction
def predict_large(model, img, debug=False):
    """
    Predict full multi-class segmentation with Stage-1 RF.

    Args:
        model: Stage-1 RandomForestClassifier.
        img: BrainImage with feature_matrix.

    Returns:
        np.ndarray: predicted labels (0..5).
    """
    X = img.feature_matrix[0]
    pred = model.predict(X)
    if debug:
        unique, counts = np.unique(pred, return_counts=True)
        print("[Stage-1] prediction label distribution:", dict(zip(unique, counts)))
    return pred


#  Stage 2 Prediction (with auto-context, inside ROI)
def predict_small(model_small,
                  model_large,
                  img,
                  pred_large,
                  roi_mode="stage1",  # "stage1" or "center"
                  debug=False):
    """
    Stage-2 prediction:
      - uses RF trained on labels {0,3,4} with auto-context features.
      - predicts for all voxels (on augmented features).
      - BUT we will only allow replacing Stage-1 labels inside ROI during fusion.

    ROI options:
      - "stage1": ROI built from Stage-1 predictions (labels 1,2,3,4,5 + dilation).
      - "center": big center disk (debug / sanity check).

    Returns:
        pred_small: Stage-2 predicted labels (0/3/4) for all voxels.
        small_conf: max probability over {3,4} for each voxel.
        roi_mask: 1D bool ROI mask.
    """
    X_orig = img.feature_matrix[0]
    n_vox = X_orig.shape[0]

    # ROI
    if roi_mode == "stage1":
        roi_mask = build_stage1_roi(pred_large, img,
                                    include_small_labels=True,
                                    dilation_iters=2,
                                    debug=debug)
        if debug:
            gt = img.feature_matrix[1].squeeze()  # GT labels for all voxels in test
            roi_recall_hippo = np.mean(roi_mask[gt == 3]) if np.any(gt == 3) else np.nan
            roi_recall_amyg  = np.mean(roi_mask[gt == 4]) if np.any(gt == 4) else np.nan
            print(f"[ROI] recall hippo={roi_recall_hippo:.3f}, amyg={roi_recall_amyg:.3f}")
    elif roi_mode == "center":
        roi_mask = build_center_disk_roi(img, radius_fraction=0.4, debug=debug)
    else:
        raise ValueError(f"Unknown roi_mode: {roi_mode}")

    # create augmented features using Stage-1 probabilities (auto-context)
    probs_large = model_large.predict_proba(X_orig)
    X_aug = _augment_features_with_stage1_probs(X_orig, probs_large, model_large.classes_, debug=debug)

    # Stage-2 predictions & probabilities
    probs_small = model_small.predict_proba(X_aug) # [N, num_classes]
    pred_small = model_small.predict(X_aug) # argmax: 0/3/4

    classes_small = model_small.classes_  # e.g. [0,3,4]

    # indices for hippocampus/amygdala
    small_class_indices = []
    for lab in SMALL_LABELS:
        idx = np.where(classes_small == lab)[0]
        if idx.size > 0:
            small_class_indices.append(idx[0])

    if len(small_class_indices) == 0:
        # Stage-2 never saw labels 3 or 4
        small_conf = np.zeros(n_vox, dtype=float)
        if debug:
            print("[Stage-2] WARNING: no small labels (3/4) in model_small.classes_.")
    else:
        small_conf = np.max(probs_small[:, small_class_indices], axis=1)

    if debug:
        if np.sum(roi_mask) > 0:
            print("[Stage-2] mean small-conf inside ROI:",
                  float(np.mean(small_conf[roi_mask])))
        else:
            print("[Stage-2] ROI empty!")
        uniq2, cnt2 = np.unique(pred_small, return_counts=True)
        print("[Stage-2] prediction label distribution:", dict(zip(uniq2, cnt2)))

    return pred_small, small_conf, roi_mask


#  Fusion of Stage 1 + Stage 2
def fuse_predictions(pred_large,
                     pred_small,
                     small_conf,
                     roi_mask,
                     conf_threshold=0.05,
                     debug=False):
    """
    Fusion rule:
      - Start from Stage-1 prediction (full multi-class).
      - Inside ROI, if Stage-2 is confident enough (small_conf > threshold)
        AND predicts a small label (3 or 4),
        overwrite Stage-1 with Stage-2 prediction.
      - Outside ROI, keep Stage-1 as is.

    Args:
        pred_large (np.ndarray): Stage-1 labels.
        pred_small (np.ndarray): Stage-2 labels (0,3,4).
        small_conf (np.ndarray): max prob over {3,4} from Stage-2.
        roi_mask (np.ndarray, bool): ROI mask.
        conf_threshold (float): Confidence threshold for replacement.
        debug (bool): Debug prints.

    Returns:
        np.ndarray: final fused labels.
    """
    final = pred_large.copy()

    # replace where Stage-2 is confident and predicts small structure inside ROI
    mask_replace = (small_conf > conf_threshold) & roi_mask & np.isin(pred_small, list(SMALL_LABELS))
    final[mask_replace] = pred_small[mask_replace]

    if debug:
        mask_small = np.isin(pred_small, [3, 4]) & roi_mask
        print("[Fusion] small_conf min/mean/max on replaced candidates:",
            float(np.min(small_conf[mask_small])),
            float(np.mean(small_conf[mask_small])),
            float(np.max(small_conf[mask_small])))
        print("[Fusion] Voxels replaced by Stage-2:", int(np.sum(mask_replace)))
        uniq_f, cnt_f = np.unique(final, return_counts=True)
        print("[Fusion] Final label distribution:", dict(zip(uniq_f, cnt_f)))

    return final


#  Convert numpy prediction to SimpleITK image
def convert_to_image(pred, img_properties):
    return conversion.NumpySimpleITKImageBridge.convert(pred.astype(np.uint8), img_properties)