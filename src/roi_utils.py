import json
import hyperspy.api as hs
from pathlib import Path



def save_rois_by_tilt(roi_by_tilt, path):

    data = {}

    for tilt, rois in roi_by_tilt.items():

        data[str(tilt)] = {}

        for key, roi in rois.items():

            data[str(tilt)][key] = {
                "left": roi.left,
                "right": roi.right,
                "top": roi.top,
                "bottom": roi.bottom,
            }

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved ROIs to {path}")


def load_rois(path):

    with open(path, "r") as f:
        data = json.load(f)

    roi_by_tilt = {}

    for tilt_str, rois in data.items():

        tilt = int(tilt_str)

        roi_by_tilt[tilt] = {}

        for key, values in rois.items():

            roi_by_tilt[tilt][key] = hs.roi.RectangularROI(
                left=values["left"],
                right=values["right"],
                top=values["top"],
                bottom=values["bottom"]
            )

    return roi_by_tilt

def extract_roi_subsets(signals, roi_by_tilt):

    roi_subsets = {}

    for tilt in sorted(signals.keys()):

        print(f"\nProcessing tilt {tilt}")

        signal = signals[tilt]
        rois = roi_by_tilt[tilt]

        roi_subsets[tilt] = {}

        for key, roi in rois.items():

            subset = roi(signal)

            # compute lazy signal
            subset.compute()

            roi_subsets[tilt][key] = subset

            print(f"ROI {key}: extracted")

    return roi_subsets


def load_polar_rois(polar_root):

    roi_polar = {}

    for tilt_dir in sorted(polar_root.glob("tilt_*")):

        if not tilt_dir.is_dir():
            continue

        tilt = int(tilt_dir.name.split("_")[1])

        roi_polar[tilt] = {}

        files = sorted(tilt_dir.glob("ROI_*_polar.hspy"))

        print(f"\nTilt {tilt}")
        print(f"Found {len(files)} files")

        for file in files:

            roi_key = file.stem.split("_")[1]

            print(f"Loading ROI {roi_key}")

            signal = hs.load(str(file), lazy=True)

            roi_polar[tilt][roi_key] = signal

        print("complete")

    return roi_polar