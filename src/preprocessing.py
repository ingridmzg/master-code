import matplotlib.pyplot as plt #Plotting tools
import numpy as np
import hyperspy.api as hs
import math

# center profiles plot
def plot_center_profiles(dp_before, dp_after, title):
    before = dp_before.data
    after = dp_after.data

    cy = before.shape[0] // 2
    cx = before.shape[1] // 2

    plt.figure(figsize=(7, 4))
    plt.plot(before[cy, :], label="Original horizontal")
    plt.plot(after[cy, :], label="Centered horizontal")
    plt.yscale("symlog")
    plt.axvline(cx, linestyle="--")
    plt.title(title + " horizontal profile")
    plt.xlabel("CameraX pixel")
    plt.ylabel("Intensity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title}_horizontal.svg", format="svg", bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(before[:, cx], label="Original vertical")
    plt.plot(after[:, cx], label="Centered vertical")
    plt.yscale("symlog")
    plt.axvline(cy, linestyle="--")
    plt.title(title + " vertical profile")
    plt.xlabel("CameraY pixel")
    plt.ylabel("Intensity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title}_vertical.svg", format="svg", bbox_inches="tight")
    plt.show()

# print summary statistics for BeamShift object
def print_shift_stats(name, shifts):
    """
    BeamShift:
    The two signal components correspond to the estimated beam shifts
    in the detector x and y directions.
    """
    try:
        shifts.compute()
    except Exception:
        pass

    arr = np.asarray(shifts.data)

    sx = arr[:, :, 0]
    sy = arr[:, :, 1]

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print("NaNs:", np.isnan(arr).sum())
    print("Infs:", np.isinf(arr).sum())

    print("sx mean/std:", np.nanmean(sx), np.nanstd(sx))
    print("sy mean/std:", np.nanmean(sy), np.nanstd(sy))

    print("sx min/max:", np.nanmin(sx), np.nanmax(sx))
    print("sy min/max:", np.nanmin(sy), np.nanmax(sy))

    print("sx p5/p95:", np.nanpercentile(sx, 5), np.nanpercentile(sx, 95))
    print("sy p5/p95:", np.nanpercentile(sy, 5), np.nanpercentile(sy, 95))

# crop zero filled borders
def make_crop_and_nav_mask_from_applied_shifts(
    signal_centered,
    applied_shifts,
    extra_margin=2,
    percentile=None,
    square_crop=True,
):
    """
    Make a signal-axis crop and a navigation mask based on the shifts applied
    during direct-beam centering.

    Parameters:
   -  signal_centered: The centered 4D-STEM/SPED signal.
   -  applied_shifts: The BeamShift object used in signal.center_direct_beam(shifts=...).
    - extra_margin: Extra pixels added to the crop.
    - percentile: If None, use maximum absolute shift.
    - square_crop: If True, crop the same number of pixels from all signal-axis edges.
    - This gives a square diffraction pattern

    Returns:
    - signal_cropped: Centered signal cropped in signal axes.
    - nav_mask: Boolean navigation mask. True = keep this scan position.
    - crop_x, crop_y: Number of pixels cropped from each side of CameraX/CameraY.

    """
    try:
        applied_shifts.compute()
    except Exception:
        pass

    arr = np.asarray(applied_shifts.data)

    sx = arr[:, :, 0]
    sy = arr[:, :, 1]

    if percentile is None:
        max_abs_sx = np.nanmax(np.abs(sx))
        max_abs_sy = np.nanmax(np.abs(sy))
    else:
        max_abs_sx = np.nanpercentile(np.abs(sx), percentile)
        max_abs_sy = np.nanpercentile(np.abs(sy), percentile)

    crop_x_raw = int(math.ceil(max_abs_sx)) + extra_margin
    crop_y_raw = int(math.ceil(max_abs_sy)) + extra_margin

    if square_crop:
        crop = max(crop_x_raw, crop_y_raw)
        crop_x = crop
        crop_y = crop
    else:
        crop_x = crop_x_raw
        crop_y = crop_y_raw

    print("Applied-shift statistics")
    print("sx min/max:", np.nanmin(sx), np.nanmax(sx))
    print("sy min/max:", np.nanmin(sy), np.nanmax(sy))
    print("sx max abs used:", max_abs_sx)
    print("sy max abs used:", max_abs_sy)

    print()
    print("Recommended signal-axis crop")
    print(f"raw crop_x = {crop_x_raw} pixels from left/right")
    print(f"raw crop_y = {crop_y_raw} pixels from top/bottom")

    if square_crop:
        print(f"square crop = {crop_x} pixels from all signal-axis edges")
    else:
        print(f"crop_x = {crop_x} pixels from left/right")
        print(f"crop_y = {crop_y} pixels from top/bottom")

    # Crop signal axes only.
    # isig order follows signal coordinates: isig[CameraX, CameraY]
    signal_cropped = signal_centered.isig[
        crop_x:-crop_x,
        crop_y:-crop_y
    ]

    # Navigation mask: True means keep this scan position.
    # If percentile=None, all positions should be valid by construction.
    nav_keep = (
        (np.abs(sx) <= crop_x - extra_margin) &
        (np.abs(sy) <= crop_y - extra_margin)
    )

    nav_mask = hs.signals.Signal2D(nav_keep.T).T

    print()
    print("Navigation mask")
    print("Kept scan positions:", np.sum(nav_keep))
    print("Total scan positions:", nav_keep.size)
    print("Fraction kept:", np.sum(nav_keep) / nav_keep.size)

    print()
    print("Cropped signal")
    print(signal_cropped)
    print(signal_cropped.axes_manager)

    return signal_cropped, nav_mask, crop_x, crop_y

