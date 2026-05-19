import json
import hyperspy.api as hs

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

def run_template_matching(
    roi_polar,
    simulations,
    n_keep=8,
    frac_keep=1.0,
):

    roi_results = {}

    for tilt in sorted(roi_polar.keys()):

        roi_results[tilt] = {}

        for roi_key, signal_pol in roi_polar[tilt].items():

            print(f"Tilt {tilt}, ROI {roi_key}")

            res = signal_pol.get_orientation(
                simulation=simulations,
                n_best=n_keep,
                frac_keep=frac_keep,
            )

            roi_results[tilt][roi_key] = res

            print("TM complete")

    return roi_results