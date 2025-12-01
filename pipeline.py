"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import sys
import timeit
import warnings

import numpy as np
import pandas as pd
import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer
import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble

try:
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
    from mialab.experiments import hierarchical_model as hmodel
    from mialab.experiments.grouped_random_forest import GroupedRandomForest
    from mialab.experiments.label_subset_ensemble import LabelSubsetEnsembleRF
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
    from mialab.experiments import hierarchical_model as hmodel
    from mialab.experiments.grouped_random_forest import GroupedRandomForest
    from mialab.experiments.label_subset_ensemble import LabelSubsetEnsembleRF

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load

def main_wrapper(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str, model_type: str, debug: bool = False, seed: int = 42, multi_seed_runs: int = 1):
    """Wrapper that allows multi-seed experiments."""
    all_results = [] # aggregated list of rows to write to CSV
    for run_idx in range(multi_seed_runs):
        current_seed = seed + run_idx
        print(f"\n========== Running seed {current_seed} ({run_idx+1}/{multi_seed_runs}) ==========\n")
        np.random.seed(current_seed)
        # run the original pipeline (but modified to RETURN results)
        detailed_rows = main_single_run(
            result_dir,
            data_atlas_dir,
            data_train_dir,
            data_test_dir,
            model_type,
            debug,
            current_seed,
            multi_seed_runs
        )
        all_results.extend(detailed_rows)
    df = pd.DataFrame(all_results)

    out_path = os.path.join(result_dir, "multi_seed_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved aggregated results to:\n{out_path}")

def main_single_run(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str, model_type: str, debug: bool = False, seed: int = 42, multi_seed_runs: int = 1):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Decision forest classifier model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)

    print('-' * 5, f'Training {model_type} model...')

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_train_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())
    pre_process_params = {'skullstrip_pre': True,
                          'normalization_pre': True,
                          'registration_pre': True,
                          'histogram_matching_pre': False,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True}

    # load images for training and pre-process
    try:
        import multiprocessing
        multi_process = multiprocessing.cpu_count() > 2
        print('Using multi-processing with', multiprocessing.cpu_count(), 'cores.')
    except Exception:
        multi_process = False
        print('Using single processing.')
    images = putil.pre_process_batch(crawler.data, pre_process_params, 
    multi_process=multi_process)

    if debug:
        # GT label distribution after preprocessing
        all_labels = np.concatenate([img.feature_matrix[1].squeeze() for img in images])
        unique, counts = np.unique(all_labels, return_counts=True)
        print("GT distribution after preprocessing:", dict(zip(unique, counts)))
    # generate feature matrix and label vector
    data_train = np.concatenate([img.feature_matrix[0] for img in images])
    labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()

    #warnings.warn('Random forest parameters not properly set.')
    # forest = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1],
    #                                             n_estimators=1,
    #                                             max_depth=5)
    start_time = timeit.default_timer()
    forest = None
    forest_large = None
    forest_small = None
    if model_type == 'baseline':
        forest = sk_ensemble.RandomForestClassifier(
            max_features=images[0].feature_matrix[0].shape[1],
            n_estimators=50,
            max_depth=20,
            n_jobs=-1,
            random_state=42)
        forest.fit(data_train, labels_train)

    elif model_type == 'hierarchical':
        forest_large = hmodel.train_large_rf(images, debug=debug)

        # Stage 1 train predictions (needed for Stage 2)
        train_preds_large = [
            forest_large.predict(img.feature_matrix[0]) 
            for img in images
        ]
        forest_small = hmodel.train_small_rf(images, train_preds_large, debug=debug)
    
    elif model_type == 'grouped':
        forest = GroupedRandomForest(max_features=images[0].feature_matrix[0].shape[1],
                                 n_estimators_large=50, 
                                 n_estimators_small=50,
                                 max_depth=20,
                                 random_state=42)
        forest.fit(data_train, labels_train)
    
    elif model_type == 'random_subset_ensemble':
        forest = LabelSubsetEnsembleRF(n_models=8, n_estimators=50, max_depth=20,
                                    n_jobs=-1,
                                    random_state=42,
                                    max_features=images[0].feature_matrix[0].shape[1],
                                    min_labels_per_model=2,
                                    max_labels_per_model=5)
        forest.fit(data_train, labels_train)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
    if debug:
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

    save_segmentations = (multi_seed_runs == 1) # create result directory only if saving segmentations
    if save_segmentations:
        # create a result directory with timestamp
        t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + f"_seed_{seed}"
        result_dir = os.path.join(result_dir, t)
        os.makedirs(result_dir, exist_ok=True)
    else:
        result_dir = None  # segmentations will not be saved

    print('-' * 5, 'Testing...')

    # initialize evaluator
    evaluator = putil.init_evaluator()

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_test_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())

    # load images for testing and pre-process
    pre_process_params['training'] = False
    images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=multi_process)

    images_prediction = []
    images_probabilities = []

    for img in images_test:
        print('-' * 10, 'Testing', img.id_)

        start_time = timeit.default_timer()
        if model_type == 'baseline' or model_type == 'grouped' or model_type == 'random_subset_ensemble':
            predictions = forest.predict(img.feature_matrix[0])
            probabilities = forest.predict_proba(img.feature_matrix[0])
            # convert prediction and probabilities back to SimpleITK images
            image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8), img.image_properties)
            image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)
            # evaluate segmentation without post-processing
            evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

            images_prediction.append(image_prediction)
            images_probabilities.append(image_probabilities)
        elif model_type == 'hierarchical':
            # Stage 1 prediction
            pred_large = hmodel.predict_large(forest_large, img, debug=debug)
            # Stage 2 prediction
            pred_small, small_conf, roi_mask = hmodel.predict_small(forest_small, img, pred_large, debug=debug)
            # Fuse
            pred_final = hmodel.fuse_predictions(pred_large, pred_small, small_conf, roi_mask, debug=debug)
            # Convert to image
            image_prediction = hmodel.convert_to_image(pred_final, img.image_properties)
            image_probabilities = None  # not used for simple post-processing
            evaluator.evaluate(image_prediction,
                       img.images[structure.BrainImageTypes.GroundTruth],
                       img.id_)
            images_prediction.append(image_prediction)
            images_probabilities.append(None)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        if debug:
            print(' Time elapsed:', timeit.default_timer() - start_time, 's')

    # post-process segmentation and evaluate with post-processing
    if model_type == 'hierarchical':
        # no post-processing at first while debugging
        post_process_params = {'simple_post': False}
    else:
        post_process_params = {'simple_post': True}
    images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities, post_process_params, multi_process=multi_process)

    for i, img in enumerate(images_test):
        evaluator.evaluate(images_post_processed[i], img.images[structure.BrainImageTypes.GroundTruth],
                           img.id_ + '-PP')

        # save results
        # sitk.WriteImage(images_prediction[i], os.path.join(result_dir, images_test[i].id_ + '_SEG.mha'), True)
        # sitk.WriteImage(images_post_processed[i], os.path.join(result_dir, images_test[i].id_ + '_SEG-PP.mha'), True)
        if save_segmentations:
            out_seg = os.path.join(result_dir, images_test[i].id_ + '_SEG.mha')
            out_pp = os.path.join(result_dir, images_test[i].id_ + '_SEG-PP.mha')
            sitk.WriteImage(images_prediction[i], out_seg, True)
            sitk.WriteImage(images_post_processed[i], out_pp, True)
    if save_segmentations:
        # use two writers to report the results
        os.makedirs(result_dir, exist_ok=True)  # generate result directory, if it does not exists
        result_file = os.path.join(result_dir, 'results.csv')
        writer.CSVWriter(result_file).write(evaluator.results)

    print('\nSubject-wise results...')
    writer.ConsoleWriter().write(evaluator.results)
    functions = {'MEAN': np.mean, 'STD': np.std}

    if save_segmentations:
        # report also mean and standard deviation among all subjects
        result_summary_file = os.path.join(result_dir, 'results_summary.csv')
        writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
    print('\nAggregated statistic results...')
    writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

    # Extract detailed metric rows for this seed
    detailed_rows = []
    for r in evaluator.results:
        subject = r.id_
        label = r.label
        metric = r.metric
        value = float(r.value)

        detailed_rows.append({
            "model": model_type,
            "seed": seed,
            "subject": subject,
            "label": label,
            "metric": metric,
            "value": value
        })
    
    # clear results such that the evaluator is ready for the next evaluation
    evaluator.clear()

    return detailed_rows

if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/train/')),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/test/')),
        help='Directory with testing data.'
    )

    parser.add_argument(
        '--model_type',
        type=str,
        default='baseline',
        choices=['baseline', 'per_label', 'grouped', 'hierarchical', 'random_subset_ensemble'],
        help='Choose which segmentation model to run.'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug prints during training and testing.'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility.'
    )

    parser.add_argument(
        '--multi_seed_runs',
        type=int,
        default=1,
        help='Run pipeline multiple times with different seeds (for boxplot experiments).'
    )
    args = parser.parse_args()
    main_wrapper(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir, args.model_type, args.debug, args.seed, args.multi_seed_runs)