import numpy as np
import SimpleITK as sitk

from data import LV_LABEL

# --- slice extraction utilities ----

def select_frame(volume_4d: sitk.Image, frame_idx: int) -> sitk.Image:
    """
    Extract a 3D frame from a 4D image.
    Volume is extracted at time frame_idx from a (x, y, z, t) image.
    """
    size = list(volume_4d.GetSize())
    extractor = sitk.ExtractImageFilter()
    extractor.SetSize(size[:3] + [0])  # keep x, y, z dimensions, set t to 0
    extractor.SetIndex([0, 0, 0, frame_idx])  # start at the desired time frame
    return extractor.Execute(volume_4d)

def extract_slice(image_3d: sitk.Image, mask_3d:sitk.Image, frame_idx: int):
    """
    Extract the same 2D slice from both image and mask, given a frame index.
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
    """
    original_spacing = image.GetSpacing() 
    original_size = image.GetSize()

    new_size = [int(round(original_size[i] * (original_spacing[i] / target_spacing[i]))) for i in range(2)] 

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())

    if is_label:  # if the image is a mask, use nearest neighbor interpolation to preserve label values
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:  # if the image is data, use linear interpolation for better quality
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)


# --- cropping utilities ---

def get_lv_center(mask_2d: sitk.Image) -> tuple[float, float]:
    """
    Get the center of mass of the left ventricle (LV) in a 2D mask.
    """
    mask_np = sitk.GetArrayFromImage(mask_2d)  # convert to numpy array
    coords = np.argwhere(mask_np == LV_LABEL)  # get coordinates of LV pixels

    if len(coords) == 0: # if there are no LV pixels, return the center of the image
        raise ValueError("LV not found in mask.")
    
    y_mean, x_mean = coords.mean(axis=0)  # compute mean coordinates (center of mass)
    return int(x_mean), int(y_mean)

def crop_around_center(image: sitk.Image, center, size=512) -> sitk.Image:
    """
    Crop image centered at given (x,y).
    Pads if necessary.
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
    """
    image_np = sitk.GetArrayFromImage(image).astype(np.float32)
    
    min_val = image_np.min()
    max_val = image_np.max()

    if max_val - min_val > 0: image_np = (image_np - min_val) / (max_val - min_val)  # scale to [0,1]
    else: image_np[:] = 0.0

    return sitk.GetImageFromArray(image_np)