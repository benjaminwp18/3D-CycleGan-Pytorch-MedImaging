
import os
import shutil
from time import time
import re
import argparse
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from utils.NiftiDataset import *
from pathlib import Path


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def lstFiles(Path):

    images_list = []  # create an empty list, the raw image data files is stored here
    for dirName, subdirList, fileList in os.walk(Path):
        for filename in fileList:
            if ".nii.gz" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
            elif ".nii" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
            elif ".mhd" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
            elif '.hdr' in filename.lower():
                images_list.append(os.path.join(dirName, filename))

    images_list = sorted(images_list, key=numericalSort)
    return images_list


def Align(image, reference):

    image_array = sitk.GetArrayFromImage(image)

    label_origin = reference.GetOrigin()
    label_direction = reference.GetDirection()
    label_spacing = reference.GetSpacing()

    image = sitk.GetImageFromArray(image_array)
    image.SetOrigin(label_origin)
    image.SetSpacing(label_spacing)
    image.SetDirection(label_direction)

    return image


def CropBackground(image, label):
    size_new = (240, 240, 120)

    def Normalization(image):
        """
        Normalize an image to 0 - 255 (8bits)
        """
        normalizeFilter = sitk.NormalizeImageFilter()
        resacleFilter = sitk.RescaleIntensityImageFilter()
        resacleFilter.SetOutputMaximum(255)
        resacleFilter.SetOutputMinimum(0)

        image = normalizeFilter.Execute(image)  # set mean and std deviation
        image = resacleFilter.Execute(image)  # set intensity 0-255

        return image

    image2 = Normalization(image)
    label2 = Normalization(label)

    threshold = sitk.BinaryThresholdImageFilter()
    threshold.SetLowerThreshold(20)
    threshold.SetUpperThreshold(255)
    threshold.SetInsideValue(1)
    threshold.SetOutsideValue(0)

    roiFilter = sitk.RegionOfInterestImageFilter()
    roiFilter.SetSize([size_new[0], size_new[1], size_new[2]])

    image_mask = threshold.Execute(image2)
    image_mask = sitk.GetArrayFromImage(image_mask)
    image_mask = np.transpose(image_mask, (2, 1, 0))

    import scipy
    centroid = scipy.ndimage.measurements.center_of_mass(image_mask)

    x_centroid = int(centroid[0])
    y_centroid = int(centroid[1])

    roiFilter.SetIndex([int(x_centroid - (size_new[0]) / 2), int(y_centroid - (size_new[1]) / 2), 0])

    label_crop = roiFilter.Execute(label)
    image_crop = roiFilter.Execute(image)

    return image_crop, label_crop


def Registration(image, label):

    image, image_sobel, label, label_sobel,  = image, image, label, label

    Gaus = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    image_sobel = Gaus.Execute(image_sobel)
    label_sobel = Gaus.Execute(label_sobel)

    fixed_image = label_sobel
    moving_image = image_sobel

    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()
    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    registration_method.SetInterpolator(sitk.sitkLinear)
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))

    image = sitk.Resample(image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                     moving_image.GetPixelID())

    return image, label


parser = argparse.ArgumentParser()
# parser.add_argument('--images', default='./Data_folder/T1', help='path to the images a (early frames)')
# parser.add_argument('--labels', default='./Data_folder/T2', help='path to the images b (late frames)')
# parser.add_argument('--split', default=50, help='number of images for testing')
# parser.add_argument('--resolution', default=(1.6,1.6,1.6), help='new resolution to resample the all data')
# args = parser.parse_args()

DIR_PATHS = (
    Path('Data_folder') / 'train' / 'images',
    Path('Data_folder') / 'train' / 'labels',
    Path('Data_folder') / 'test' / 'images',
    Path('Data_folder') / 'test' / 'labels',
)

RESOLUTION = (1.6, 1.6, 1.6)

if __name__ == "__main__":
    image_lists: list[list[str]] = list([lstFiles(path) for path in DIR_PATHS])

    reference_image = image_lists[0][0]    # setting a reference image to have all data in the same coordinate system
    reference_image = sitk.ReadImage(reference_image)
    reference_image = resample_sitk_image(reference_image, spacing=RESOLUTION, interpolator='linear')

    for image_list in image_lists:
        for image_path in image_list:
            print(image_path)
            image = sitk.ReadImage(image_path)

            # Skip registration as we have no good way to pair
            # label, reference_image = Registration(label, reference_image)
            # image, label = Registration(image, label)

            image = resample_sitk_image(image, spacing=RESOLUTION, interpolator='linear')

            # image = Align(image, reference_image)
            # label = Align(label, reference_image)

            sitk.WriteImage(image, Path(image_path).parent / (Path(image_path).stem + '.nii'))