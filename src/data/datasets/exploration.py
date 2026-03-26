import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# --- LOADING --
img4d_path = "data/patient101/patient101_4d.nii.gz"
img_obj = nib.load(img4d_path)
img4d = img_obj.get_fdata()

mask_paths = {
    "ED": "data/patient101/patient101_frame01_gt.nii.gz",
    "ES": "data/patient101/patient101_frame14_gt.nii.gz"
}
masks = {k: nib.load(v).get_fdata() for k, v in mask_paths.items()}

# --- Setup ---
init_slice = img4d.shape[2] // 2
init_frame = 0

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
img_display = ax.imshow(img4d[:,:,init_slice, init_frame], cmap="gray")

ax.set_title(f"Frame: {init_frame}, Slice: {init_slice}")
ax.axis("off")

# --- Mask Overlay ---
label_colors = np.array([  # color map for mask labels
    [0, 0, 0, 0],       # Background (transparent)
    [1, 0, 0, 0.4],     # RV (red)
    [0, 1, 0, 0.4],     # Myocardium (green)
    [0, 0, 1, 0.4]      # LV (blue)
])

mask_overlay_img = np.zeros((*img4d.shape[:2], 4)) # RGBA
mask_overlay = ax.imshow(mask_overlay_img)

# --- Sliders ---
ax_frame = plt.axes([0.2, 0.1, 0.6, 0.03])
ax_slice = plt.axes([0.2, 0.05, 0.6, 0.03])

slider_frame = Slider(
    ax_frame, 
    "Frame",
    0,
    img4d.shape[3] - 1,
    valinit=init_frame,
    valstep=1
)

slider_slice = Slider(
    ax_slice, 
    "Slice",
    0,
    img4d.shape[2] - 1,
    valinit=init_slice,
    valstep=1
)

# --- Update Function ---
def update(val):
    frame = int(slider_frame.val)
    slice_idx = int(slider_slice.val)

    img_display.set_data(img4d[:,:,slice_idx, frame])

    mask_overlay_img[:] = 0
    if frame == 0: mask_slice = masks["ED"][:,:,slice_idx].astype(int)
    elif frame == 13: mask_slice = masks["ES"][:,:,slice_idx].astype(int)
    else: mask_slice = np.zeros_like(mask_overlay_img[:,:,0], dtype=int)
    
    # Colorize
    for label in [1, 2, 3]: mask_overlay_img[mask_slice == label] = label_colors[label]
    mask_overlay.set_data(mask_overlay_img)

slider_frame.on_changed(update)
slider_slice.on_changed(update)

plt.show()