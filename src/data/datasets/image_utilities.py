import numpy as np
import SimpleITK as sitk

from data import LV_LABEL

DEFAULT_CROP_SIZE = 512

# --- slice extraction utilities ----

def select_frame(volume_4d: sitk.Image, frame_idx: int) -> sitk.Image:
    """
    Extract a 3D frame from a 4D image.

    Parameters
    ----------
    volume_4d : sitk.Image
        4D input image with dimensions ordered as (x, y, z, t).
    frame_idx : int
        Index of the time frame to extract along the t-axis.

    Returns
    -------
    sitk.Image
        3D image corresponding to the selected time frame.
    """
    size = list(volume_4d.GetSize())
    extractor = sitk.ExtractImageFilter()
    extractor.SetSize(size[:3] + [0])  # keep x, y, z dimensions, set t to 0
    extractor.SetIndex([0, 0, 0, frame_idx])  # start at the desired time frame
    return extractor.Execute(volume_4d)

def extract_slice(
    image_3d: sitk.Image,
    mask_3d: sitk.Image,
    frame_idx: int,
) -> tuple[sitk.Image, sitk.Image]:
    """
    Extract the same 2D slice from both image and mask, given a frame index.

    Parameters
    ----------
    image_3d : sitk.Image
        3D input image (x, y, z).
    mask_3d : sitk.Image
        3D mask image, spatially aligned with `image_3d`.
    frame_idx : int
        Index of the slice along the z-axis to extract.

    Returns
    -------
    tuple[sitk.Image, sitk.Image]
        2D image slice and corresponding 2D mask slice at `frame_idx`.
    """
    size = list(image_3d.GetSize())
    extractor = sitk.ExtractImageFilter()
    extractor.SetSize(size[:2] + [0])  # keep x, y dimensions, set z to 0
    extractor.SetIndex([0, 0, frame_idx])  # start at the desired frame index and time frame (0)
    return extractor.Execute(image_3d), extractor.Execute(mask_3d)

# --- resampling utilities ---

def resample_slice(
    image: sitk.Image,
    target_spacing: tuple[float, float] = (1.0, 1.0),
    is_label: bool = False,
) -> sitk.Image:
    """
    Resample a 2D image to the target spacing.

    Parameters
    ----------
    image : sitk.Image
        2D image to be resampled.
    target_spacing : tuple[float, float], optional
        Desired output spacing in millimetres (sx, sy). Defaults to (1.0, 1.0).
    is_label : bool, optional
        If True, use nearest-neighbour interpolation to preserve label values.
        If False, use linear interpolation for intensity images.

    Returns
    -------
    sitk.Image
        Resampled 2D image with the requested spacing.
    """
    original_spacing = image.GetSpacing() 
    original_size = image.GetSize()

    new_size = [int(round(original_size[i] * (original_spacing[i] / target_spacing[i]))) for i in range(2)] 

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())

    if is_label:
        # If the image is a mask, use nearest neighbor interpolation to preserve label values.
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # If the image is data, use linear interpolation for better quality.
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)


# --- cropping utilities ---

def get_lv_center(mask_2d: sitk.Image) -> tuple[int, int]:
    """
    Get the center of mass of the left ventricle (LV) in a 2D mask.

    Parameters
    ----------
    mask_2d : sitk.Image
        2D mask image containing the LV label.

    Returns
    -------
    tuple[int, int]
        Integer coordinates ``(x, y)`` of the LV centre of mass.

    Raises
    ------
    ValueError
        If the LV label is not present in the mask.
    """
    mask_np = sitk.GetArrayFromImage(mask_2d)  # convert to numpy array
    coords = np.argwhere(mask_np == LV_LABEL)  # get coordinates of LV pixels

    if len(coords) == 0:
        raise ValueError("LV not found in mask.")
    
    y_mean, x_mean = coords.mean(axis=0)  # compute mean coordinates (center of mass)
    return int(x_mean), int(y_mean)

def crop_around_center(
    image: sitk.Image,
    center: tuple[int, int],
    size: int = DEFAULT_CROP_SIZE,
) -> sitk.Image:
    """
    Crop image centred at a given point and pad as needed.

    Parameters
    ----------
    image : sitk.Image
        2D input image to be cropped.
    center : tuple[int, int]
        Centre of the crop in pixel coordinates ``(x, y)``.
    size : int, optional
        Desired output size (both width and height) in pixels. Defaults to ``DEFAULT_CROP_SIZE``.

    Returns
    -------
    sitk.Image
        Cropped (and possibly padded) 2D image of size ``(size, size)``.
    """
    width, height = image.GetSize()
    cx, cy = center
    half = size // 2

    start_x = max(cx - half, 0)
    start_y = max(cy - half, 0)
    end_x = min(cx + half, width)
    end_y = min(cy + half, height)

    extractor = sitk.RegionOfInterestImageFilter()
    extractor.SetIndex([start_x, start_y])
    extractor.SetSize([end_x - start_x, end_y - start_y])
    cropped = extractor.Execute(image)

    return sitk.ConstantPad(
        cropped, 
        [0, 0],
        [size - cropped.GetSize()[0], size - cropped.GetSize()[1]],
        0, 
    )

# --- normalization utilities ---

def normalize(image: sitk.Image) -> sitk.Image:
    """
    Normalize intensities to [0,1].

    Parameters
    ----------
    image : sitk.Image
        Input image whose intensity range will be normalised.

    Returns
    -------
    sitk.Image
        Image with intensities scaled to the range \[0, 1\]. Constant images
        are mapped to all zeros.
    """
    image_np = sitk.GetArrayFromImage(image).astype(np.float32)
    
    min_val = image_np.min()
    max_val = image_np.max()

    if max_val - min_val > 0:
        # Scale to [0, 1] when there is variation in intensities.
        image_np = (image_np - min_val) / (max_val - min_val)
    else:
        # For constant images, return zeros.
        image_np[:] = 0.0

    return sitk.GetImageFromArray(image_np)