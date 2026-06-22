import numpy as np
from orix.quaternion import Orientation

"""
ROI-based orientaiton trajectory selection using symmetry-equivalent misorientation continuity

This module selects representative orientation trajectories from template-
matched SPED ROIs. For each ROI, pixels are first screened using the
highest-ranked template-matching candidate. The full n-best candidate sequence
is then optimized only for the best few pixels.

Misorientation calculations are based on orix orientation objects and
symmetry-equivalent orientations.

- collect ROI orientation score/data
- screen pixels using candidate 0
- optimize n_best candidate sequence for shortlisted pixels
- select the smoothest trajectory
- print validation
"""


def collect_roi_data(roi_results, twin_key):
    """
    Collect orientation maps and candidate-0 score maps for one ROI through
    the tilt series.

    The ROI shape can vary slightly between tilts. The returned common shape
    is therefore the overlapping navigation area shared by all tilts.
    """
    tilts = np.array(sorted(roi_results.keys()))

    orientations = {}
    scores = {}

    min_x = np.inf
    min_y = np.inf

    for tilt in tilts:
        result = roi_results[tilt][twin_key]

        ori_map = result.to_single_phase_orientations()

        score_map = result.data[:, :, 0, 1]

        if hasattr(score_map, "compute"):
            score_map = score_map.compute()

        score_map = np.asarray(score_map)

        if score_map.shape != ori_map.shape[:2]:
            if score_map.T.shape == ori_map.shape[:2]:
                score_map = score_map.T
            else:
                raise ValueError(
                    f"Shape mismatch at tilt {tilt}: "
                    f"orientations={ori_map.shape[:2]}, "
                    f"scores={score_map.shape}"
                )

        orientations[tilt] = ori_map
        scores[tilt] = score_map

        min_x = min(min_x, ori_map.shape[0])
        min_y = min(min_y, ori_map.shape[1])

    common_shape = (int(min_x), int(min_y))

    return tilts, orientations, scores, common_shape


def smallest_symmetry_misorientation(ori_a, ori_b, symmetry):
    """
    Smallest symmetry-equivalent misorientation angle between two orientations.

    The orientations are rebuilt with the supplied crystal symmetry. The
    symmetry-equivalent variants of the second orientation are generated with
    orix, and the smallest angle to the first orientation is returned.
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


def trajectory_score(jumps):
    """
    Score trajectory continuity.

    The score does not force the expected experimental tilt step. It penalizes
    large isolated jumps and variation in neighbouring-step misorientation.
    """
    jumps = np.asarray(jumps)

    median = np.median(jumps)
    mad = np.median(np.abs(jumps - median))

    return float(mad + 0.5 * np.max(jumps))


def tilt_step_misorientation(
    orientations,
    tilts,
    symmetry,
):
    """
    Calculate misorientation between neighbouring tilt steps.
    """
    jumps = []

    n_tilts = len(tilts)

    for i in range(n_tilts - 1):

        mis = smallest_symmetry_misorientation(
            orientations[i],
            orientations[i + 1],
            symmetry=symmetry,
        )

        jumps.append(mis)

    return np.array(jumps)


def select_smooth_candidate_sequence(candidate_oris, symmetry, rank_weight=0.02):
    """
    Select the "smoothest" candidate sequence for one pixel.

    Template matching gives several orientation candidates at each tilt. This
    function finds the path through those candidates with the lowest total
    neighbouring-step misorientation. A small penalty is added for using
    lower-ranked candidates.
    """
    n_tilts = candidate_oris.shape[0]
    n_candidates = candidate_oris.shape[1]

    cost = np.full((n_tilts, n_candidates), np.inf)
    back = np.full((n_tilts, n_candidates), -1, dtype=int)

    cost[0, :] = rank_weight * np.arange(n_candidates)

    for i in range(1, n_tilts):
        for j in range(n_candidates):

            previous = candidate_oris[i - 1]
            current = candidate_oris[i, j]

            jumps = np.array(
                [
                    smallest_symmetry_misorientation(
                        previous[k],
                        current,
                        symmetry=symmetry,
                    )
                    for k in range(n_candidates)
                ]
            )

            values = cost[i - 1, :] + jumps**2 + rank_weight * j

            back[i, j] = int(np.argmin(values))
            cost[i, j] = float(values[back[i, j]])

    chosen_idx = np.zeros(n_tilts, dtype=int)
    chosen_idx[-1] = int(np.argmin(cost[-1]))

    for i in range(n_tilts - 1, 0, -1):
        chosen_idx[i - 1] = back[i, chosen_idx[i]]

    selected_oris = Orientation(
        np.stack(
            [candidate_oris[i, chosen_idx[i]].data.squeeze() for i in range(n_tilts)]
        ),
        symmetry=symmetry,
    )

    return selected_oris, chosen_idx, cost


def run_roi_trajectory_selection(
    roi_results,
    twin_key,
    symmetry,
    score_percentile=30,
    stride=2,
    keep_n=5,
    rank_weight=0.02,
):
    """
    Search one ROI for the pixel and candidate sequence giving the smoothest
    orientation trajectory through the tilt series.

    The search is done in two stages:
    1. All pixels are screened using candidate 0 only.
    2. The full n-best candidate optimization is run for the best pixels.
    """
    print(f"\n=== Twin {twin_key}: ROI trajectory selection ===")

    tilts, orientations, scores, common_shape = collect_roi_data(
        roi_results=roi_results,
        twin_key=twin_key,
    )

    nx, ny = common_shape

    mean_score = np.mean(
        np.stack([scores[tilt][:nx, :ny] for tilt in tilts]),
        axis=0,
    )

    score_threshold = np.percentile(mean_score, score_percentile)

    screened = []

    for px in range(0, nx, stride):
        for py in range(0, ny, stride):

            if mean_score[px, py] < score_threshold:
                continue

            candidate0_oris = [orientations[tilt][px, py, 0] for tilt in tilts]

            jumps = tilt_step_misorientation(
                candidate0_oris,
                tilts,
                symmetry=symmetry,
            )

            screened.append(
                {
                    "pixel": (px, py),
                    "score": trajectory_score(jumps),
                    "jumps": jumps,
                    "max_jump": float(jumps.max()),
                    "mean_score": float(mean_score[px, py]),
                }
            )

    screened = sorted(
        screened,
        key=lambda item: item["score"],
    )

    shortlisted = screened[:keep_n]

    if len(shortlisted) == 0:
        raise RuntimeError(
            f"No pixels passed the score threshold for Twin {twin_key}. "
            "Try lowering score_percentile or checking the ROI."
        )

    print(f"Screened pixels: {len(screened)}")
    print(f"Shortlisted pixels: {len(shortlisted)}")

    best = None
    tested_full = []

    for item in shortlisted:

        px, py = item["pixel"]

        candidate_oris = Orientation(
            np.stack([orientations[tilt][px, py].data for tilt in tilts]),
            symmetry=symmetry,
        )

        selected_oris, chosen_idx, cost = select_smooth_candidate_sequence(
            candidate_oris=candidate_oris,
            symmetry=symmetry,
            rank_weight=rank_weight,
        )

        jumps = tilt_step_misorientation(
            selected_oris,
            tilts,
            symmetry=symmetry,
        )

        full_item = {
            "pixel": (px, py),
            "score": trajectory_score(jumps),
            "jumps": jumps,
            "max_jump": float(jumps.max()),
            "chosen_idx": chosen_idx,
            "axis_oris": selected_oris,
            "candidate_oris": candidate_oris,
            "center_oris": candidate_oris[:, 0],
            "cost": cost,
        }

        tested_full.append(full_item)

        print(
            f"pixel={(px, py)}, "
            f"score={full_item['score']:.2f}, "
            f"max jump={full_item['max_jump']:.2f}°, "
            f"chosen={chosen_idx}"
        )

        if best is None or full_item["score"] < best["score"]:
            best = full_item

    if best is None:
        raise RuntimeError(f"No valid trajectory found for Twin {twin_key}")

    result = {
        "tilts": tilts,
        "pixel": best["pixel"],
        "representative_pixel": best["pixel"],
        "candidate_oris": best["candidate_oris"],
        "center_oris": best["center_oris"],
        "selected_oris": best["axis_oris"],
        "axis_oris": best["axis_oris"],
        "chosen_idx": best["chosen_idx"],
        "trajectory_jumps": best["jumps"],
        "trajectory_score": best["score"],
        "method": "roi_trajectory_selection",
    }

    diagnostics = {
        "screened": screened,
        "shortlisted": shortlisted,
        "tested_full": tested_full,
        "mean_score_map": mean_score,
        "common_shape": common_shape,
        "score_threshold": float(score_threshold),
    }

    print(f"\nSelected Twin {twin_key} pixel: {best['pixel']}")
    print("Selected jumps:", np.round(best["jumps"], 2))

    return result, diagnostics


def run_roi_trajectory_selection_all_twins(
    roi_results,
    twin_keys,
    symmetry,
    score_percentile=30,
    stride=2,
    keep_n=5,
    rank_weight=0.02,
):
    """
    Run ROI trajectory selection for all twins.
    """
    results = {}
    representative_pixels = {}
    diagnostics = {}

    for twin_key in twin_keys:

        result, diag = run_roi_trajectory_selection(
            roi_results=roi_results,
            twin_key=twin_key,
            symmetry=symmetry,
            score_percentile=score_percentile,
            stride=stride,
            keep_n=keep_n,
            rank_weight=rank_weight,
        )

        results[twin_key] = result
        representative_pixels[twin_key] = result["pixel"]
        diagnostics[twin_key] = diag

    return results, representative_pixels, diagnostics


def print_tilt_step_misorientation(
    trajectory_results,
    twin_keys,
    symmetry,
):
    """
    Print neighbouring-step misorientation for selected orientation trajectories.
    """
    for twin_key in twin_keys:

        result = trajectory_results[twin_key]

        tilts = result["tilts"]
        oris = result["selected_oris"]

        jumps = tilt_step_misorientation(
            oris,
            tilts,
            symmetry=symmetry,
        )

        print(f"\n Twin {twin_key}: tilt-step misorientation")

        for i, mis in enumerate(jumps):

            print(f"{tilts[i]:>3} → {tilts[i + 1]:>3}: " f"{mis:.2f}°")
