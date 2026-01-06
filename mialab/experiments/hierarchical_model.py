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

def _count_map(arr):
    u, c = np.unique(arr, return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))

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
        # seed_labels = [3, 4]
    else:
        # seed_labels = [1, 2, 5]
        seed_labels = [3, 4]

    roi_mask = np.isin(pred_large, seed_labels)

    # Reshape to 3D, apply dilation, then flatten back
    roi_mask_3d = roi_mask.reshape(img.image_properties.size[::-1])  # (z, y, x)
    roi_mask_3d = binary_dilation(roi_mask_3d, iterations=dilation_iters)
    roi_mask = roi_mask_3d.flatten()

    if debug:
        n_roi = int(np.sum(roi_mask))
        n_all = int(roi_mask.size)
        print(f"[ROI] stage1-seed={seed_labels}, dilation={dilation_iters}")
        print(f"[ROI] ROI size: {n_roi} voxels ({100.0*n_roi/max(1,n_all):.2f}% of image)"
              f"({np.mean(roi_mask) * 100:.2f}% of image)")

        # show what Stage-1 predicted inside ROI (helps explain “where it looks”)
        pred_in = pred_large[roi_mask]
        print("[ROI] Stage-1 label dist inside ROI:", _count_map(pred_in))
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
        n_roi = int(np.sum(roi_mask))
        n_all = int(roi_mask.size)
        print(f"[ROI] center-disk radius_fraction={radius_fraction}")
        print(f"[ROI] ROI size: {n_roi} voxels ({100.0*n_roi/max(1,n_all):.2f}% of image)")

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
        print("[Stage-1][train] GT distribution:", _count_map(y))
        print(f"[Stage-1][train] X shape={X.shape}, features={X.shape[1]}")

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
        print(f"[Stage-2][auto-context] original F={X.shape[1]}, prob dims={probs_large.shape[1]}, total={X_aug.shape[1]}")
        print(f"[Stage-2][auto-context] Stage-1 classes order: {classes_large}")
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
    kept_total = 0
    pos_total = 0
    for img_idx, img in enumerate(images):
        X_orig = img.feature_matrix[0] # [N_samp, F]
        y_orig = img.feature_matrix[1].squeeze() # [N_samp]
        # Map labels: keep 3/4, everything else → 0
        y_mod = np.where(np.isin(y_orig, list(SMALL_LABELS)), y_orig, 0)

        # Stage-1 probabilities on this (sampled) feature set
        probs_large = model_large.predict_proba(X_orig)

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

            keep_mask = (y_mod != 0) | roi_mask # keep all small (3,4) + tissue-like negatives
            X_use = X_aug[keep_mask]
            y_use = y_mod[keep_mask]
        else:
            X_use = X_aug
            y_use = y_mod

        kept_total += int(np.sum(keep_mask))
        pos_total += int(np.sum(y_use != 0))
        if debug and img_idx == 0:
            print(f"[Stage-2][train] First train img: sampled voxels={len(y_mod)}")
            print(f"[Stage-2][train] Kept voxels={int(np.sum(keep_mask))} "
                  f"({100.0*np.mean(keep_mask):.2f}%) via {'ROI negatives' if use_spatial_downsampling else 'no downsampling'}")
            print("[Stage-2][train] y_mod dist (0/3/4):", _count_map(y_mod))
            print("[Stage-2][train] y_use dist (0/3/4):", _count_map(y_use))

        X_all.append(X_use)
        y_all.append(y_use)

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)

    if len(X) == 0:
        raise RuntimeError("Stage-2 SMALL RF: no training samples at all.")
    
    if debug:
        print("[Stage-2][train] Final training dist (0/3/4):", _count_map(y))
        print(f"[Stage-2][train] Total kept voxels={kept_total}, positives(3/4)={pos_total} "
              f"({100.0*pos_total/max(1,kept_total):.4f}%)")
        print(f"[Stage-2][train] X shape={X.shape}, features={X.shape[1]}")

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
        print("[Stage-1][predict] pred dist:", _count_map(pred))
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
            gt = img.feature_matrix[1].squeeze()
            n_h = int(np.sum(gt == 3))
            n_a = int(np.sum(gt == 4))
            in_h = int(np.sum(roi_mask[gt == 3])) if n_h > 0 else 0
            in_a = int(np.sum(roi_mask[gt == 4])) if n_a > 0 else 0
            rec_h = (in_h / n_h) if n_h > 0 else np.nan
            rec_a = (in_a / n_a) if n_a > 0 else np.nan
            print(f"[ROI] GT Hip voxels={n_h}, in ROI={in_h}, recall={rec_h:.3f}")
            print(f"[ROI] GT Amy voxels={n_a}, in ROI={in_a}, recall={rec_a:.3f}")
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
        inside = roi_mask
        n_in = int(np.sum(inside))
        print(f"[Stage-2][predict] voxels total={n_vox}, ROI voxels={n_in} ({100.0*n_in/max(1,n_vox):.2f}%)")
        if n_in > 0:
            print("[Stage-2][predict] mean small_conf inside ROI:", float(np.mean(small_conf[inside])))
        print("[Stage-2][predict] pred dist:", _count_map(pred_small))

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
    mask_candidate = roi_mask & np.isin(pred_small, list(SMALL_LABELS))
    mask_replace = mask_candidate & (small_conf > float(conf_threshold))
    final[mask_replace] = pred_small[mask_replace]

    if debug:
        n_roi = int(np.sum(roi_mask))
        n_cand = int(np.sum(mask_candidate))
        n_rep = int(np.sum(mask_replace))
        print(f"[Fusion] conf_threshold={conf_threshold}")
        print(f"[Fusion] ROI voxels={n_roi}, candidates (Stage2 says small in ROI)={n_cand}, replaced={n_rep} "
              f"({100.0*n_rep/max(1,n_roi):.2f}% of ROI)")

        # replacement breakdown
        if n_rep > 0:
            rep_labels = pred_small[mask_replace]
            rep_dist = _count_map(rep_labels)
            print("[Fusion] replaced label breakdown:", rep_dist)

            conf_vals = small_conf[mask_replace]
            print("[Fusion] small_conf on replaced: "
                  f"min={float(np.min(conf_vals)):.3f}, mean={float(np.mean(conf_vals)):.3f}, max={float(np.max(conf_vals)):.3f}")
        else:
            print("[Fusion] no voxels replaced at this threshold.")

        print("[Fusion] final pred dist:", _count_map(final))

    return final


#  Convert numpy prediction to SimpleITK image
def convert_to_image(pred, img_properties):
    return conversion.NumpySimpleITKImageBridge.convert(pred.astype(np.uint8), img_properties)
