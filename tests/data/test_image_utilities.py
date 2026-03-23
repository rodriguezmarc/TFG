import numpy as np
import SimpleITK as sitk
import pytest

from data import LV_LABEL
from data.image_utilities import (
    select_frame, 
    extract_slice,
    resample_slice,
    get_lv_center,
    crop_around_center,
    normalize
)

SLICE_INDEX = 15
CROP_SIZE = 128
CROP_CENTER = (128, 128)
PADDING_CENTER = (10, 10)

# 1. select_frame FUNCTIONAL
def test_select_frame_returns_3d_frame(synthetic_image):  # given an image
    """
    Extract frame index 2 from a 4D image.
    Expected a 3D SimpleITK image that matches spatial dimensions of the original.
    """
    frame = select_frame(synthetic_image, frame_idx=2)
    original_size = synthetic_image.GetSize()
    frame_size = frame.GetSize()

    assert isinstance(frame, sitk.Image), (f"frame should be a SimpleITK frame instance, but got {type(frame)}")
    assert frame.GetDimension() == 3, (f"frame should be 3D, but got dimension {frame.GetDimension()}")
    assert frame_size == original_size[:3], (f"frame size should match original spatial dimensions {original_size[:3]}, but got {frame_size}")

# 2. extract_slice FUNCTIONAL
def test_extract_slice_returns_matching_slices(synthetic_frame_and_mask):  # given a frame and its mask
    """
    Extract slice with idx 15 from a frame and its mask.
    Expected both 2D SimpleITK image that share identical dimensions and match the original's.
    """
    frame, mask_3d = synthetic_frame_and_mask
    slice_, mask_2d = extract_slice(frame, mask_3d, SLICE_INDEX)

    # instances
    assert isinstance(slice_, sitk.Image), (f"slice_ should be a SimpleITK image instance, but got {type(slice_)}")
    assert isinstance(mask_2d, sitk.Image), (f"mask_2d should be a SimpleITK image instance, but got {type(mask_2d)}")
    
    # dimensions
    assert slice_.GetDimension() == 2, (f"slice_ should be 2D, but got dimension {slice_.GetDimension()}")
    assert mask_2d.GetDimension() == 2, (f"mask_2d should be 2D, but got dimension {mask_2d.GetDimension()}")

    # spatial size
    assert slice_.GetSize() == frame.GetSize()[:2], (f"slice size should match frame's size {frame.GetSize()[:2]}, but got {slice_.GetSize()}")
    assert mask_2d.GetSize() == mask_3d.GetSize()[:2], (f"mask_2d size should match mask_3d's size {mask_3d.GetSize()[:2]}, but got {mask_2d.GetSize()}")
    assert slice_.GetSize() == mask_2d.GetSize(), (f"slice and mask_2d should have the same size, but got {slice_.GetSize()} vs {mask_2d.GetSize()}")

# 3. resample_slice FUNCTIONAL
def test_resample_slice_changes_spacing(synthetic_slice):
    """
    Resample image from spacing (2.0, 2.0) to (1.0, 1.0).
    Expected spatial resolution to be doubled in each dimension.
    """
    resampled = resample_slice(synthetic_slice, target_spacing=(1.0, 1.0))
    original_size = synthetic_slice.GetSize()
    new_size = resampled.GetSize()

    assert isinstance(resampled, sitk.Image), (f"resampled should be a SimpleITK image instance, but got {type(resampled)}")
    assert resampled.GetSpacing() == (1.0, 1.0), (f"resampled image should have the spacing of (1.0, 1.0), but got {resampled.GetSpacing()}" )
    expected_size = tuple(dim * 2 for dim in original_size)
    assert new_size == expected_size, (
        f"resampled image size should be double the original size {original_size} due to halving the spacing, but got {new_size}"
    )

# 4. resample_slice PRESERVES VALUES
def test_resample_slice_label_preserves_values(synthetic_slice):
    """
    Resample mask from spacing (2.0, 2.0) to (1.0, 1.0)
    Expected label vales to remain exactly {0, LV_LABEL}
    """
    mask_arr = np.zeros((64, 64), dtype=np.uint8)
    mask_arr[20:30, 20:30] = LV_LABEL  # create a square region with the label value
    mask = sitk.GetImageFromArray(mask_arr)
    mask.SetSpacing((2.0, 2.0))

    resampled = resample_slice(mask, (1.0,1.0), is_label=True)
    unique_vals = np.unique(sitk.GetArrayFromImage(resampled))

    assert set(unique_vals).issubset({0, LV_LABEL}), (f"resampled mask should only contain values 0 and {LV_LABEL}, but got {unique_vals}")  

# 5. get_lv_center FUNCTIONL
def test_get_lv_center_returns_center_near_inserted_region(synthetic_3d_image_and_mask):
    """
    LV inserted at region 30:35.
    Expected center (32, 32).
    """
    _, mask_3d = synthetic_3d_image_and_mask
    
    # extract slice with LV
    extractor = sitk.ExtractImageFilter()
    extractor.SetSize([64, 64, 0])
    extractor.SetIndex([0, 0, 15])
    
    mask_2d = extractor.Execute(mask_3d)
    cx, cy = get_lv_center(mask_2d)

    # region inserted at 30:35, so expected center is around 32
    assert 31 <= cx <= 33, (f"LV center x-coordinate should be around 32, but got {cx}")
    assert 31 <= cy <= 33, (f"LV center y-coordinate should be around 32, but got {cy}")

# 6. get_lv_center WHEN NO MASK
def test_get_lv_center_raises_when_no_lv_present():
    """
    If no mask voxels exist, should raise ValueError.
    """
    empty = sitk.GetImageFromArray(np.zeros((64, 64), dtype=np.uint8))
    with pytest.raises(ValueError, match="LV not found in mask."): get_lv_center(empty)

# 7. crop_around_center FUNCTIONAL
def test_crop_around_center_basic():
    """
    Crop 128x128 region centered at (128, 128) from a 256x256 image.
    Expected output size of 128x128.
    """
    arr = np.random.rand(256, 256).astype(np.float32)
    img = sitk.GetImageFromArray(arr)

    cropped = crop_around_center(img, center=CROP_CENTER, size=CROP_SIZE)
    
    assert cropped.GetSize() == (CROP_SIZE, CROP_SIZE), (f"Cropped image should have size {(CROP_SIZE, CROP_SIZE)}, but got {cropped.GetSize()}") 


# 8. crop_around_center WITH PADDING
def test_crop_around_center_with_padding_preserves_original_region_and_pads_with_zeros():
    """
    Crop 128x128 region centered at (10,10) from a 100x100 image.
    Since crop exceeds boundaries, padding should be applied.
    Expected output size of 128x128, where padded region is full of 0.
    Expected original data to be preserved in the top-left area.
    """
    arr = np.ones((100, 100), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)

    cropped = crop_around_center(img, center=PADDING_CENTER, size=CROP_SIZE)
    cropped_np = sitk.GetArrayFromImage(cropped)
    nonzero_count = np.count_nonzero(cropped_np)
    
    assert cropped.GetSize() == (CROP_SIZE, CROP_SIZE), (f"Cropped image should have size {(CROP_SIZE, CROP_SIZE)} even with padding, but got {cropped.GetSize()}")
    assert set(np.unique(cropped_np)).issubset({0.0, 1.0}), (
        f"Cropped image should only contain original ones and zero padding, but got {np.unique(cropped_np)}"
    )
    assert nonzero_count > 0, "Cropped image should preserve some original non-zero pixels."

# 9. normalize FUNCTIONAL
def test_normalize_range():
    """
    Normalize array with minimum value of 0 and maximum of 15.
    Expected output range of [0,1].
    """
    arr = np.array([[0, 5], [10, 15]], dtype=np.float32)
    img = sitk.GetImageFromArray(arr)

    norm = normalize(img)
    norm_np = sitk.GetArrayFromImage(norm)

    assert np.isclose(norm_np.min(), 0.0), (f"Normalized image minimum should be 0.0, but got {norm_np.min()}"  )
    assert np.isclose(norm_np.max(), 1.0), (f"Normalized image maximum should be 1.0, but got {norm_np.max()}" )

# 10. normalize WITH CONSTANT VALUE
def test_normalize_constant():
    """
    Given input image with a constant value, normalization should return an image of all zeros.
    """
    arr = np.ones((10, 10), dtype=np.float32) * 5
    img = sitk.GetImageFromArray(arr)

    norm = normalize(img)
    norm_np = sitk.GetArrayFromImage(norm)

    assert np.all(norm_np == 0.0), (f"Normalized image should be all zeros when input is constant, but got unique values {np.unique(norm_np)}") 