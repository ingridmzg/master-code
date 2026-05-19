import itertools
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from orix.quaternion import Orientation
from orix.vector import Vector3d

# ----------------------DATA EXTRACTION ----------------------------


def get_selected_orientations(axis_angle_results, twin_key):
    """
    Return the selected orientations for one twin.

    Parameters
    ----------
    axis_angle_results : dict
        Output from the axis-angle candidate-selection workflow.
    twin_key : str
        Twin label, for example "A", "B", "C", or "D".

    Returns
    -------
    tilts : np.ndarray
        Tilt angles.
    oris : Orientation
        Selected orientations.
    chosen_idx : np.ndarray
        Selected candidate index at each tilt.
    """
    result = axis_angle_results[twin_key]

    return (
        result["tilts"],
        result["axis_oris"],
        result["chosen_idx"],
    )


def get_selected_vectors(axis_angle_results, twin_key):
    """
    Convert selected orientations to beam-direction vectors.

    The selected orientation `g` was applied to the crystal reference direction `[001]` to obtain the corresponding beam-direction vector. This gives the zone-axis direction associated with each tilt step.
    """
    tilts, oris, chosen_idx = get_selected_orientations(axis_angle_results, twin_key)

    vectors = oris * Vector3d.zvector()

    return tilts, vectors, chosen_idx


# ----------------------GEOMETRY ----------------------------


def fit_arc_pole(vectors):
    """
    Fit a great-circle pole to a set of stereographic trajectory vectors.

    The fitted pole is the normal to the best-fit plane through the
    beam-direction vectors.
    """
    X = vectors.data.reshape(-1, 3)
    X = X / np.linalg.norm(X, axis=1)[:, None]

    _, _, vh = np.linalg.svd(X, full_matrices=False)

    pole = vh[-1]
    pole = pole / np.linalg.norm(pole)

    return Vector3d(pole)


def pole_for_plot(pole):
    """
    Choose the pole sign used for plotting.

    The poles +p and -p define the same great circle. This function only
    selects the sign that is easier to display in the selected stereographic
    hemisphere.
    """
    p = pole.data.reshape(3)

    if p[2] < 0:
        p = -p

    return Vector3d(p)


def smallest_angle(angle):
    """
    Return the smallest equivalent angle between two unoriented axes.
    """
    return min(angle, 180 - angle)


def neighbour_angle_jumps(vectors):
    """
    Calculate angular changes between neighbouring beam-direction vectors.

    These jumps are used as a continuity check for the stereographic
    trajectory through the tilt series.
    """
    jumps = []

    for i in range(vectors.shape[0] - 1):
        angle = float(vectors[i].angle_with(vectors[i + 1], degrees=True))

        jumps.append(angle)

    return np.array(jumps)

def smallest_symmetry_misorientation(ori_a, ori_b, symmetry):
    """
    Calculate the smallest symmetry-equivalent misorientation angle
    between two orientations of the same crystal symmetry.
    """
    ori_a = Orientation(
        ori_a.data.squeeze(),
        symmetry=symmetry
    )

    ori_b = Orientation(
        ori_b.data.squeeze(),
        symmetry=symmetry
    )

    equivalents_b = ori_b.equivalent()

    angles = np.array([
        float(
            ori_a.angle_with(
                eq,
                degrees=True
            )
        )
        for eq in equivalents_b
    ])

    return float(angles.min())

# ------------------ Symmetry-equivalent branch selection -------------------



def get_continuous_equivalent_branch(
    axis_angle_results,
    twin_key,
    symmetry,
):
    """
    Construct a continuous symmetry-equivalent orientation branch.

    In cubic symmetry, one physical orientation can be represented by
    several symmetry-equivalent orientations. Near a fundamental-zone
    boundary, the plotted representative can switch branch.

    This function keeps the physical solution unchanged, but chooses a
    continuous symmetry-equivalent representative for stereographic
    visualization.
    """
    tilts, oris, chosen_idx = get_selected_orientations(axis_angle_results, twin_key)

    oris_continuous = [oris[0]]

    for i in range(1, oris.shape[0]):

        previous = oris_continuous[-1]
        current = oris[i]

        equivalents = current.equivalent()

        angles = np.array(
            [float(previous.angle_with(eq, degrees=True)) for eq in equivalents]
        )

        best = equivalents[np.argmin(angles)]
        oris_continuous.append(best)

    oris_continuous = Orientation(
        np.stack([ori.data.squeeze() for ori in oris_continuous]), symmetry=symmetry
    )

    vectors_continuous = oris_continuous * Vector3d.zvector()

    return tilts, oris_continuous, vectors_continuous, chosen_idx


def choose_plotting_branch(
    axis_angle_results,
    twin_key,
    symmetry,
    jump_threshold=12,
    improvement_margin=3,
):
    """
    Choose the branch used for stereographic plotting.

    The original selected branch is retained when it is already continuous.
    If the original branch contains a large jump, a continuous
    symmetry-equivalent branch is tested. The continuous branch is used
    only if it clearly improves the maximum neighbour-to-neighbour jump.
    """
    tilts_orig, vectors_orig, chosen_idx_orig = get_selected_vectors(
        axis_angle_results, twin_key
    )

    jumps_orig = neighbour_angle_jumps(vectors_orig)
    max_orig = jumps_orig.max()

    tilts_cont, oris_cont, vectors_cont, chosen_idx_cont = (
        get_continuous_equivalent_branch(
            axis_angle_results=axis_angle_results,
            twin_key=twin_key,
            symmetry=symmetry,
        )
    )

    jumps_cont = neighbour_angle_jumps(vectors_cont)
    max_cont = jumps_cont.max()

    use_continuous = (
        max_orig > jump_threshold and max_cont < max_orig - improvement_margin
    )

    if use_continuous:
        return {
            "tilts": tilts_cont,
            "vectors": vectors_cont,
            "orientations": oris_cont,
            "chosen_idx": chosen_idx_cont,
            "branch_type": "continuous_equivalent",
            "jumps": jumps_cont,
            "max_original_jump": float(max_orig),
            "max_continuous_jump": float(max_cont),
        }

    return {
        "tilts": tilts_orig,
        "vectors": vectors_orig,
        "orientations": None,
        "chosen_idx": chosen_idx_orig,
        "branch_type": "original",
        "jumps": jumps_orig,
        "max_original_jump": float(max_orig),
        "max_continuous_jump": float(max_cont),
    }


def build_plotting_branches(
    axis_angle_results,
    twin_keys,
    symmetry,
    jump_threshold=12,
    improvement_margin=3,
):
    """
    Build stereographic plotting branches and fitted arc poles for all twins.
    """
    vectors_plot_by_twin = {}
    arc_poles_plot = {}

    for twin_key in twin_keys:

        branch = choose_plotting_branch(
            axis_angle_results=axis_angle_results,
            twin_key=twin_key,
            symmetry=symmetry,
            jump_threshold=jump_threshold,
            improvement_margin=improvement_margin,
        )

        vectors_plot_by_twin[twin_key] = branch
        arc_poles_plot[twin_key] = fit_arc_pole(branch["vectors"])

    return vectors_plot_by_twin, arc_poles_plot


def build_vectors_and_arc_poles(axis_angle_results, twin_keys):
    """
    Build original selected vectors and fitted arc poles for all twins.

    This is useful for comparing the unreduced plotting branches against
    the original selected orientation representation.
    """
    vectors_by_twin = {}
    arc_poles = {}

    for twin_key in twin_keys:

        tilts, vectors, chosen_idx = get_selected_vectors(axis_angle_results, twin_key)

        vectors_by_twin[twin_key] = {
            "tilts": tilts,
            "vectors": vectors,
            "chosen_idx": chosen_idx,
        }

        arc_poles[twin_key] = fit_arc_pole(vectors)

    return vectors_by_twin, arc_poles


# ---------------------- PLOTTING ----------------------------


def plot_ipf_overview(
    axis_angle_results,
    twin_keys,
    point_group,
):
    """
    Plot selected orientations for all twins in side-by-side m-3m IPFs.

    This plot is an overview of how the selected orientations are
    represented inside the cubic fundamental zone.
    """
    cmap = plt.cm.plasma

    fig = plt.figure(figsize=(14, 4))

    for i, twin_key in enumerate(twin_keys):

        tilts, oris, chosen_idx = get_selected_orientations(
            axis_angle_results, twin_key
        )

        norm = plt.Normalize(vmin=tilts.min(), vmax=tilts.max())

        colours = cmap(norm(tilts))

        ax = fig.add_subplot(
            1, len(twin_keys), i + 1, projection="ipf", symmetry=point_group
        )

        ax.scatter(oris, color=colours, s=90, alpha=0.9)

        ax.set_title(f"Twin {twin_key}", fontsize=12)

    fig.subplots_adjust(right=0.88)

    cax = fig.add_axes([0.91, 0.18, 0.015, 0.64])

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)

    cbar.set_label("Tilt")
    cbar.set_ticks(tilts)
    cbar.set_ticklabels([str(t) for t in tilts])

    fig.suptitle("m-3m IPFs", fontsize=14)

    return fig


def plot_unreduced_stereographic_trajectory(
    vectors,
    tilts,
    title="Continuous unreduced stereographic trajectory",
    cmap_name="plasma",
    marker_size=120,
    label_points=True,
):
    """
    Plot one stereographic trajectory without IPF symmetry reduction.

    The input vectors should already represent the chosen plotting branch.
    """
    cmap = plt.get_cmap(cmap_name)

    norm = plt.Normalize(vmin=tilts.min(), vmax=tilts.max())

    colours = cmap(norm(tilts))

    fig = plt.figure(figsize=(7, 6))

    ax = fig.add_subplot(111, projection="stereographic")

    ax.scatter(vectors, color=colours, s=marker_size, alpha=0.95)

    if label_points:

        for vector, tilt in zip(vectors, tilts):
            ax.text(vector, s=str(tilt), size=10, offset=(0, 0.04))

    ax.set_title(title, fontsize=14)

    ax.stereographic_grid(True)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, shrink=0.85)

    cbar.set_label("Tilt")

    plt.tight_layout()

    return fig


def plot_stereographic_arcs(
    twin_list,
    vectors_plot_by_twin,
    arc_poles,
    colors,
    title,
):
    """
    Plot stereographic trajectories, fitted great circles, and arc poles
    for selected twins.
    """
    first_key = twin_list[0]

    fig = vectors_plot_by_twin[first_key]["vectors"].scatter(
        c=colors[first_key],
        s=90,
        axes_labels=["RD", "TD", None],
        show_hemisphere_label=True,
        return_figure=True,
    )

    ax = fig.axes[0]

    for twin_key in twin_list[1:]:

        vectors_plot_by_twin[twin_key]["vectors"].scatter(
            figure=fig, c=colors[twin_key], s=90
        )

    for twin_key in twin_list:

        pole = pole_for_plot(arc_poles[twin_key])

        pole.scatter(figure=fig, c=colors[twin_key], marker="*", s=250, reproject=True)

        pole.draw_circle(
            figure=fig, color=colors[twin_key], linewidth=2, reproject=True
        )

    ax.set_title(title)
    ax.stereographic_grid(True)

    return fig


# ---------------------- ANALYSIS ----------------------------


def calculate_arc_pole_angles(arc_poles, twin_keys):
    """
    Calculate angular separations between fitted arc poles.
    """
    arc_angle_dict = {}

    for a, b in itertools.combinations(twin_keys, 2):

        angle = float(arc_poles[a].angle_with(arc_poles[b], degrees=True))

        angle_small = smallest_angle(angle)

        arc_angle_dict[f"{a}{b}"] = {
            "angle": angle,
            "smallest_equivalent": angle_small,
        }

    return arc_angle_dict


def calculate_misorientation_vs_tilt(axis_angle_results, twin_keys):
    """
    Calculate pairwise misorientation between selected twin orientations.
    """
    tilts = axis_angle_results[twin_keys[0]]["tilts"]

    misorientation_dict = {}

    for a, b in itertools.combinations(twin_keys, 2):

        key = f"{a}{b}"
        misorientation_dict[key] = []

        oris_a = axis_angle_results[a]["axis_oris"]
        oris_b = axis_angle_results[b]["axis_oris"]

        for i, tilt in enumerate(tilts):

            misorientation = float(oris_a[i].angle_with(oris_b[i], degrees=True))

            misorientation_dict[key].append(misorientation)

    return tilts, misorientation_dict


def save_stereographic_summary(
    path,
    tilts,
    arc_angle_dict,
    misorientation_dict,
):
    """
    Save stereographic angle and misorientation results to JSON.
    """
    summary = {
        "tilts": [int(t) for t in tilts],
        "arc_pole_angles": arc_angle_dict,
        "misorientation_vs_tilt": {
            key: [float(x) for x in values]
            for key, values in misorientation_dict.items()
        },
    }

    with open(path, "w") as f:
        json.dump(summary, f, indent=4)

    return summary
