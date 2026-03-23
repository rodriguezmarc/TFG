import numpy as np
import SimpleITK as sitk
import pytest

from data import LV_LABEL

SPACING=(2.0, 2.0)
SIZE = (10, 20, 30, 4)

@pytest.fixture
def synthetic_image():
    """
    Create a synthetic 4D image (x, y, z, t).
    Shape (10, 20, 30, 4)
    """
    arr = np.random.rand(*SIZE).astype(np.float32)
    img = sitk.GetImageFromArray(arr, isVector=False)
    return img

@pytest.fixture
def synthetic_frame_and_mask():
    """
    Create 3D image + mask with known LV location
    """
    image_arr = np.random.rand(30, 64, 64).astype(np.float32)
    mask_arr = np.zeros((30, 64, 64), dtype=np.uint8)

    # Insert LV_LABEL in a known region
    mask_arr[15, 30:35, 30:35] = LV_LABEL
    
    image = sitk.GetImageFromArray(image_arr, isVector=False)
    mask = sitk.GetImageFromArray(mask_arr, isVector=False)
    return image, mask

@pytest.fixture
def synthetic_slice():
    """
    Create a synthetic 2D image (x, y).
    Shape (64, 64)
    """
    arr = np.linspace(0, 100, 64*64).reshape(64,64).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(SPACING)
    return img


@pytest.fixture
def synthetic_3d_image_and_mask():
    """
    Backward-compatible alias for tests expecting this fixture name.
    """
    image_arr = np.random.rand(30, 64, 64).astype(np.float32)
    mask_arr = np.zeros((30, 64, 64), dtype=np.uint8)
    mask_arr[15, 30:35, 30:35] = LV_LABEL

    image = sitk.GetImageFromArray(image_arr, isVector=False)
    mask = sitk.GetImageFromArray(mask_arr, isVector=False)
    return image, mask