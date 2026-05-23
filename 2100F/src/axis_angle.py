import numpy as np
from sklearn.decomposition import PCA
from orix.quaternion import Orientation

"""
Axis-angle candidate selection

This method is based on "Denis Lobe masterwork": https://github.com/denislobe/Masteroppgave/tree/main

1. Convert orix axis-angle objects to xyz arrays
2. Extract n_best candidates for one pixel through the tilt series
3. Fit a PCA line to candidate-0 in axis-angle space
4. Replace outlier tilts with the candidate closest to the fitted line
5. Run this for one twin
6. Optional wrapper for all twins
7. Print tilt-step misorientation for validation

This module contains the axis-angle method that was tested during development
of the representative-orientation workflow. It operates on a manually selected
representative pixel.

For each ROI/twin, the n-best template-matching candidates are collected
through the tilt series. Candidate 0 is converted to axis-angle coordinates,
a line is fitted to the resulting trajectory, and candidate-0 outliers are
replaced by the available candidate closest to the fitted line.
This method is kept for documentation and comparison. The final automated
method used for the thesis is implemented separately in
"roi_trajectory_selection.py".

"""


def axang_to_xyz(axang):
    """
    Convert an orix axis-angle object to an array of shape (n, 3).
    """
    xyz = np.asarray(axang.xyz)
    xyz = np.squeeze(xyz)

    if xyz.ndim == 1:
        xyz = xyz[None, :]

    if xyz.ndim == 2 and xyz.shape[0] == 3 and xyz.shape[-1] != 3:
        xyz = xyz.T

    if xyz.shape[-1] != 3:
        raise ValueError(f"Unexpected axis-angle shape: {xyz.shape}")

    return xyz


def get_candidates_for_pixel(
    roi_results,
    twin_key,
    px,
    py,
    symmetry,
):
    """
    Collect n-best template-matching candidates for one pixel through the
    tilt series.
    """
    tilts = np.array(sorted(roi_results.keys()))

    candidates = []

    for tilt in tilts:

        ori_map = roi_results[tilt][twin_key].to_single_phase_orientations()

        if px >= ori_map.shape[0] or py >= ori_map.shape[1]:
            raise IndexError(
                f"Pixel ({px}, {py}) outside ROI at tilt {tilt}, "
                f"shape={ori_map.shape[:2]}"
            )

        candidates.append(ori_map[px, py])

    candidate_oris = Orientation(
        np.stack([ori.data for ori in candidates]),
        symmetry=symmetry,
    )

    return tilts, candidate_oris


def fit_axis_angle_line(points):
    """
    Fit a line to axis-angle points using PCA.
    """
    centroid = points.mean(axis=0)

    pca = PCA(n_components=1)
    pca.fit(points)

    direction = pca.components_[0]
    direction = direction / np.linalg.norm(direction)

    return centroid, direction


def point_line_distance(points, centroid, direction):
    """
    Calculate perpendicular distance from points to a fitted line.
    """
    diff = points - centroid

    return np.linalg.norm(
        np.cross(diff, direction),
        axis=1,
    )


def select_candidates_axis_angle(
    candidate_oris,
    symmetry,
    threshold_percentile=70,
):
    """
    Select a candidate sequence using axis-angle line fitting.

    Candidate 0 is used as the initial trajectory. A PCA line is fitted to
    this trajectory in axis-angle space. Points far from the fitted line are
    treated as outliers and replaced by the candidate closest to the fitted
    line.
    """
    n_tilts = candidate_oris.shape[0]

    candidate0 = candidate_oris[:, 0]
    xyz0 = axang_to_xyz(candidate0.to_axes_angles())

    centroid, direction = fit_axis_angle_line(xyz0)

    distances = point_line_distance(
        xyz0,
        centroid,
        direction,
    )

    threshold = np.percentile(
        distances,
        threshold_percentile,
    )

    inlier_mask = distances < threshold

    selected_oris = Orientation(
        candidate_oris[:, 0].data.copy(),
        symmetry=symmetry,
    )

    chosen_idx = np.zeros(
        n_tilts,
        dtype=int,
    )

    for i in range(n_tilts):

        if inlier_mask[i]:
            continue

        candidates_i = candidate_oris[i]

        xyz_i = axang_to_xyz(candidates_i.to_axes_angles())

        candidate_distances = point_line_distance(
            xyz_i,
            centroid,
            direction,
        )

        best_idx = int(np.argmin(candidate_distances))

        selected_oris[i] = candidates_i[best_idx]
        chosen_idx[i] = best_idx

    return {
        "axis_oris": selected_oris,
        "chosen_idx": chosen_idx,
        "distances": distances,
        "inlier_mask": inlier_mask,
        "line_centroid": centroid,
        "line_direction": direction,
        "threshold": threshold,
    }


def run_axis_angle_selection(
    roi_results,
    twin_key,
    px,
    py,
    symmetry,
    threshold_percentile=70,
):
    """
    Run axis-angle candidate selection for one twin/ROI and one representative
    pixel.
    """
    print(f"\n=== Twin {twin_key}: axis-angle selection ===")
    print(f"Pixel: ({px}, {py})")

    tilts, candidate_oris = get_candidates_for_pixel(
        roi_results=roi_results,
        twin_key=twin_key,
        px=px,
        py=py,
        symmetry=symmetry,
    )

    selection = select_candidates_axis_angle(
        candidate_oris=candidate_oris,
        symmetry=symmetry,
        threshold_percentile=threshold_percentile,
    )

    for tilt, idx, dist, is_inlier in zip(
        tilts,
        selection["chosen_idx"],
        selection["distances"],
        selection["inlier_mask"],
    ):
        print(
            f"Tilt {tilt:>3}: "
            f"candidate={idx}, "
            f"distance={dist:.4f}, "
            f"inlier={is_inlier}"
        )

    return {
        "tilts": tilts,
        "pixel": (px, py),
        "center_pixel": (px, py),
        "candidate_oris": candidate_oris,
        "center_oris": candidate_oris[:, 0],
        "axis_oris": selection["axis_oris"],
        "chosen_idx": selection["chosen_idx"],
        "dist": selection["distances"],
        "inlier_mask": selection["inlier_mask"],
        "line_centroid": selection["line_centroid"],
        "line_direction": selection["line_direction"],
        "threshold": selection["threshold"],
        "method": "axis_angle_pca_line",
    }


def run_axis_angle_selection_all_twins(
    roi_results,
    twin_keys,
    representative_pixels,
    symmetry,
    threshold_percentile=70,
):
    """
    Run axis-angle candidate selection for all twins using predefined
    representative pixels.
    """
    results = {}

    for twin_key in twin_keys:

        px, py = representative_pixels[twin_key]

        results[twin_key] = run_axis_angle_selection(
            roi_results=roi_results,
            twin_key=twin_key,
            px=px,
            py=py,
            symmetry=symmetry,
            threshold_percentile=threshold_percentile,
        )

    return results


def smallest_symmetry_misorientation(ori_a, ori_b, symmetry):
    """
    Smallest symmetry-equivalent misorientation angle between two orientations.
    """
    ori_a = Orientation(
        ori_a.data.squeeze(),
        symmetry=symmetry,
    )

    ori_b = Orientation(
        ori_b.data.squeeze(),
        symmetry=symmetry,
    )

    angles = np.array(
        [float(ori_a.angle_with(eq, degrees=True)) for eq in ori_b.equivalent()]
    )

    return float(angles.min())


def tilt_step_misorientation(
    orientations,
    tilts,
    symmetry,
):
    """
    Calculate neighbouring-step misorientation for an orientation trajectory.
    """
    values = []

    for i in range(len(tilts) - 1):

        mis = smallest_symmetry_misorientation(
            orientations[i],
            orientations[i + 1],
            symmetry=symmetry,
        )

        values.append(mis)

    return np.array(values)


def print_tilt_step_misorientation(
    trajectory_results,
    twin_keys,
    symmetry,
):
    """
    Print neighbouring-step misorientation for selected trajectories.
    """
    for twin_key in twin_keys:

        result = trajectory_results[twin_key]

        tilts = result["tilts"]
        oris = result["axis_oris"]

        values = tilt_step_misorientation(
            orientations=oris,
            tilts=tilts,
            symmetry=symmetry,
        )

        print(f"\n=== Twin {twin_key}: tilt-step misorientation ===")

        for i, mis in enumerate(values):
            print(f"{tilts[i]:>3} → {tilts[i + 1]:>3}: " f"{mis:.2f}°")
