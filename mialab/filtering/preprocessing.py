"""The pre-processing module contains classes for image pre-processing.

Image pre-processing aims to improve the image quality (image intensities) for subsequent pipeline steps.
"""
import warnings

import numpy as np
import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk


class ImageNormalization(pymia_fltr.Filter):
    """Represents a normalization filter."""

    def __init__(self):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: pymia_fltr.FilterParams = None) -> sitk.Image:
        """Executes a normalization on an image.

        Args:
            image (sitk.Image): The image.
            params (FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The normalized image.
        """

        img_arr = sitk.GetArrayFromImage(image)

        # todo: normalize the image using numpy
        #warnings.warn('No normalization implemented. Returning unprocessed image.')

        # normalizes img intensity values so that all subjects are comparable.
        # MRI intensities are not absolute — scanners and settings vary, so you standardize them.
        
        # z-score normalization
        mean = img_arr.mean()
        std = img_arr.std()
        img_norm = (img_arr - mean) / std
        #img_norm = np.clip(img_norm, -3, 3)

        # use non zero voxels to compute mean and std
        # brain_voxels = img_arr[img_arr > 0]
        # mean = brain_voxels.mean()
        # std = brain_voxels.std()
        # img_norm = np.zeros_like(img_arr)
        # img_norm[img_arr > 0] = (img_arr[img_arr > 0] - mean) / std
        # img_norm = np.clip(img_norm, -2, 2)

        # min max normalization to [0, 1]
        # brain_voxels = img_arr[img_arr > 0]
        # min_val = brain_voxels.min()
        # max_val = brain_voxels.max()
        # img_norm = np.zeros_like(img_arr)
        # img_norm[img_arr > 0] = (img_arr[img_arr > 0] - min_val) / (max_val - min_val)

        # robust normalization using median and IQR
        # brain_voxels = img_arr[img_arr > 0]
        # median = np.median(brain_voxels)
        # q75, q25 = np.percentile(brain_voxels, [75 ,25])
        # iqr = q75 - q25
        # img_norm = np.zeros_like(img_arr)
        # img_norm[img_arr > 0] = (img_arr[img_arr > 0] - median) / iqr
        # img_norm = np.clip(img_norm, -3, 3)

        img_out = sitk.GetImageFromArray(img_norm)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageNormalization:\n' \
            .format(self=self)


class SkullStrippingParameters(pymia_fltr.FilterParams):
    """Skull-stripping parameters."""

    def __init__(self, img_mask: sitk.Image):
        """Initializes a new instance of the SkullStrippingParameters

        Args:
            img_mask (sitk.Image): The brain mask image.
        """
        self.img_mask = img_mask


class SkullStripping(pymia_fltr.Filter):
    """Represents a skull-stripping filter."""

    def __init__(self):
        """Initializes a new instance of the SkullStripping class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: SkullStrippingParameters = None) -> sitk.Image:
        """Executes a skull stripping on an image.

        Args:
            image (sitk.Image): The image.
            params (SkullStrippingParameters): The parameters with the brain mask.

        Returns:
            sitk.Image: The normalized image.
        """
        mask = params.img_mask  # the brain mask

        # todo: remove the skull from the image by using the brain mask (1 = brain, 0 = non-brain)
        #warnings.warn('No skull-stripping implemented. Returning unprocessed image.')

        # multiply img by mask voxel-wise
        mask = sitk.Cast(mask, sitk.sitkUInt8)
        image = sitk.Mask(image, mask)

        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'SkullStripping:\n' \
            .format(self=self)


class ImageRegistrationParameters(pymia_fltr.FilterParams):
    """Image registration parameters."""

    def __init__(self, atlas: sitk.Image, transformation: sitk.Transform, is_ground_truth: bool = False):
        """Initializes a new instance of the ImageRegistrationParameters

        Args:
            atlas (sitk.Image): The atlas image.
            transformation (sitk.Transform): The transformation for registration.
            is_ground_truth (bool): Indicates weather the registration is performed on the ground truth or not.
        """
        self.atlas = atlas
        self.transformation = transformation
        self.is_ground_truth = is_ground_truth


class ImageRegistration(pymia_fltr.Filter):
    """Represents a registration filter."""

    def __init__(self):
        """Initializes a new instance of the ImageRegistration class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: ImageRegistrationParameters = None) -> sitk.Image:
        """Registers an image.

        Args:
            image (sitk.Image): The image.
            params (ImageRegistrationParameters): The registration parameters.

        Returns:
            sitk.Image: The registered image.
        """

        # todo: replace this filter by a registration. Registration can be costly, therefore, we provide you the
        # transformation, which you only need to apply to the image!
        #warnings.warn('No registration implemented. Returning unregistered image')

        atlas = params.atlas
        transform = params.transformation
        is_ground_truth = params.is_ground_truth  # the ground truth will be handled slightly different

        # note: if you are interested in registration, and want to test it, have a look at
        # pymia.filtering.registration.MultiModalRegistration. Think about the type of registration, i.e.
        # do you want to register to an atlas or inter-subject? Or just ask us, we can guide you ;-)
        
        # placing rotated or scaled img exactly over a standard template by applying geometric transformation
        if is_ground_truth:
            interp = sitk.sitkNearestNeighbor
        else:
            interp = sitk.sitkLinear
        image = sitk.Resample(image, atlas, transform, interp, 0.0, image.GetPixelID())

        return image

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'ImageRegistration:\n' \
            .format(self=self)

class HistogramMatchingParameters(pymia_fltr.FilterParams):
    def __init__(self,
                 reference_image: sitk.Image,
                 histogram_levels: int = 256,
                 match_points: int = 10,
                 threshold_at_mean_intensity: bool = True):
        """
        Args:
            reference_image: Referece Image (e.g. Atlas-T1 or Atlas-T2).
            histogram_levels: Number of histogram levels.
            match_points: Number of match points.
            threshold_at_mean_intensity: Values < mean intensity are ignored.
        """
        self.reference_image = reference_image
        self.histogram_levels = histogram_levels
        self.match_points = match_points
        self.threshold_at_mean_intensity = threshold_at_mean_intensity

class HistogramMatching(pymia_fltr.Filter):

    def __init__(self):
        super().__init__()

    def execute(self, image: sitk.Image, params: HistogramMatchingParameters = None) -> sitk.Image:
        if params is None or params.reference_image is None:
            raise ValueError("HistogramMatching: reference_image must be set.")
        img_in  = sitk.Cast(image, sitk.sitkFloat32)
        ref_in  = sitk.Cast(params.reference_image, sitk.sitkFloat32)

        hm = sitk.HistogramMatchingImageFilter()
        hm.SetNumberOfHistogramLevels(params.histogram_levels)
        hm.SetNumberOfMatchPoints(params.match_points)
        hm.SetThresholdAtMeanIntensity(params.threshold_at_mean_intensity)

        out = hm.Execute(img_in, ref_in)
        out.CopyInformation(image)

        out = sitk.Cast(out, image.GetPixelID())
        return out

    def __str__(self):
        return 'HistogramMatching:\n' \
            .format(self=self)