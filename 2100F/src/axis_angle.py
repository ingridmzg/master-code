import numpy as np
from sklearn.decomposition import PCA
from orix.quaternion import Orientation


def axang_to_xyz(axang):
    xyz = np.array(axang.xyz)

    if xyz.ndim == 2 and xyz.shape[0] == 3:
        xyz = xyz.T

    xyz = np.squeeze(xyz)

    if xyz.shape[-1] != 3:
        raise ValueError(f"Unexpected axis-angle shape: {xyz.shape}")

    return xyz


def get_candidates_for_pixel(roi_results, roi_key, px, py, symmetry):
    tilt_keys = np.array(sorted(roi_results.keys()))
    all_candidates = []

    for tilt in tilt_keys:
        ori_full = roi_results[tilt][roi_key].to_single_phase_orientations()
        all_candidates.append(ori_full[px, py])

    candidates = Orientation(
        np.stack([o.data for o in all_candidates]),
        symmetry=symmetry
    )

    return tilt_keys, candidates


def select_candidates_axis_angle(candidate_oris, symmetry, threshold_percentile=70):
    tilt_n = candidate_oris.shape[0]

    candidate0 = candidate_oris[:, 0]
    xyz = axang_to_xyz(candidate0.to_axes_angles())

    pca = PCA(n_components=1)
    pca.fit(xyz)

    centroid = xyz.mean(axis=0)
    direction = pca.components_[0]
    direction /= np.linalg.norm(direction)

    diff = xyz - centroid
    dist = np.linalg.norm(np.cross(diff, direction), axis=1)

    threshold = np.percentile(dist, threshold_percentile)
    inlier_mask = dist < threshold

    fixed_oris = Orientation(
        candidate_oris[:, 0].data.copy(),
        symmetry=symmetry
    )

    chosen_idx = np.zeros(tilt_n, dtype=int)

    for i in range(tilt_n):

        if inlier_mask[i]:
            chosen_idx[i] = 0
            continue

        candidates = candidate_oris[i]
        cand_xyz = axang_to_xyz(candidates.to_axes_angles())

        diff = cand_xyz - centroid
        cand_dist = np.linalg.norm(np.cross(diff, direction), axis=1)

        best_idx = int(np.argmin(cand_dist))

        fixed_oris[i] = candidates[best_idx]
        chosen_idx[i] = best_idx

    return fixed_oris, chosen_idx, dist, inlier_mask



def run_axis_angle_selection(
    roi_results,
    twin_key,
    px,
    py,
    symmetry,
    threshold_percentile=70
):

    print(f"\n=== Twin {twin_key} ===")
    print(f"Pixel: ({px}, {py})")

    tilt_keys, candidate_oris = get_candidates_for_pixel(
        roi_results,
        twin_key,
        px,
        py,
        symmetry=symmetry
    )

    axis_oris, chosen_idx, dist, inlier_mask = select_candidates_axis_angle(
        candidate_oris,
        symmetry=symmetry,
        threshold_percentile=threshold_percentile
    )

    center_oris = candidate_oris[:, 0]

    for tilt, idx, d, ok in zip(tilt_keys, chosen_idx, dist, inlier_mask):
        print(
            f"Tilt {tilt:>3}: "
            f"candidate={idx}, "
            f"dist={d:.4f}, "
            f"inlier={ok}"
        )

    result = {
        "tilts": tilt_keys,
        "pixel": (px, py),
        "candidate_oris": candidate_oris,
        "center_oris": center_oris,
        "axis_oris": axis_oris,
        "chosen_idx": chosen_idx,
        "dist": dist,
        "inlier_mask": inlier_mask,
    }

    return result