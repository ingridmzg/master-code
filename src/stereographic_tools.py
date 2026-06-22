import itertools
import json
from fractions import (
    Fraction, # For converting vector directions to approximate integer [uvw]
)  
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from orix.quaternion import Orientation, Rotation
from orix.vector import Vector3d


# Return the selected orientations for one twin region.
def get_selected_orientations(results, twin_key):

    result = results[twin_key]

    if "selected_oris" in result:
        orientations = result["selected_oris"]
    else:
        orientations = result["axis_oris"]

    return (
        result["tilts"],
        orientations,
        result["chosen_idx"],
    )


def get_selected_vectors(results, twin_key):
    """
    Convert selected orientations to beam-direction vectors.

    The stereographic trajectory is constructed from

        v = g * [001], where g is the selected crystal orientation.
    """
    tilts, orientations, chosen_idx = get_selected_orientations(
        results,
        twin_key,
    )

    vectors = orientations * Vector3d.zvector()

    return tilts, vectors, chosen_idx


# Fit a great-circle pole to a set of beam-direction vectors. The fitted pole is the normal of the best-fit plane through the vectors.
def fit_arc_pole(vectors):

    data = np.asarray(vectors.data).reshape(
        -1, 3
    )  # reshape because orix Vector3d with multiple vectors has shape (3, N)
    data = (
        data / np.linalg.norm(data, axis=1)[:, None]
    )  # Normalize vectors to unit length

    _, _, vh = (
        np.linalg.svd(  # using single-value decomposition to find the best-fit plane normal. Alternatively, could use PCA or eigen-decomposition of the covariance matrix. chose svd for numerical stability and simplicity.
            data,
            full_matrices=False,
        )
    )

    pole = vh[
        -1
    ]  # the best fit plane normal is the last row of vh, corresponding to the smallest singular value. This is the direction with the least variance among the vectors
    pole = pole / np.linalg.norm(pole)  # Normalize the pole to unit length

    return Vector3d(pole)


# Choose a display sign for a pole
def pole_for_plot(pole):
    """
    The poles +p and -p define the same great circle. For consistent plotting, the pole is flipped if its z component is negative.
    This ensures that all poles are plotted in the upper hemisphere, avoiding point-wise reprojection artefacts while preserving the great circle geometry.
    """
    data = np.asarray(pole.data).reshape(3)

    if data[2] < 0:
        data = -data

    return Vector3d(data)


# Return the smallest equivalent angle between two unoriented axes
def smallest_axis_angle(angle):

    return min(angle, 180.0 - angle)


def neighbour_angle_jumps(vectors):
    """
    Calculate angular changes between neighbouring beam-direction vectors in a stereographic trajectory.
    This quantifies the smoothness of the trajectory and can be used to identify discontinuities or artefacts.
    """
    jumps = []

    for i in range(vectors.shape[0] - 1):  # loop over neighbouring pairs of vectors
        angle = float(
            vectors[i].angle_with(  # calculate the angle between neighbouring vectors
                vectors[
                    i + 1
                ],  # the angle is calculated in 3D space, to capture the true angular change in orientation
                degrees=True,
            )
        )

        jumps.append(angle)

    return np.array(jumps)


def orient_vectors_to_plot_hemisphere(vectors):
    """
    Put one complete trajectory in a consistent plotting hemisphere.

    The same sign is applied to the full trajectory. This avoids point-by-point
    reprojection artefacts while preserving the trajectory shape.
    """
    data = np.asarray(vectors.data).reshape(-1, 3)
    data = data / np.linalg.norm(data, axis=1)[:, None]

    mean_direction = data.mean(axis=0)

    if mean_direction[2] < 0:
        data = -data

    return Vector3d(data)


def get_stereographic_branch(
    results,
    twin_key,
    symmetry,
    reference_direction=Vector3d.zvector(),
):
    """
    Construct a continuous stereographic plotting branch.

    The selected physical orientations are not changed. For each selected
    orientation, symmetry-equivalent representatives are generated with Orix.
    The representative used for plotting is chosen by minimizing the angular change of the beam-direction vector relative to the previous tilt step.
    """
    result = results[twin_key]

    tilts, orientations, chosen_idx = get_selected_orientations(
        results,
        twin_key,
    )

    branch_orientations = []

    first = Orientation(
        orientations[0].data.squeeze(), # reshape because orix Orientation with multiple orientations has shape (3, N) for the axis and (N,) for the angle, so we need to squeeze to get a single orientation
        symmetry=symmetry,
    )

    # the first orientation is used as the starting point for the branch, and its beam-direction vector is used as the reference for choosing subsequent orientations

    branch_orientations.append(first)
    previous_vector = first * reference_direction 

    for i in range(1, orientations.shape[0]):

        current = Orientation(
            orientations[i].data.squeeze(),
            symmetry=symmetry,
        )

        equivalents = current.equivalent() # generate all symmetry-equivalent orientations for the current tilt step
        candidate_vectors = equivalents * reference_direction

        angles = previous_vector.angle_with(
            candidate_vectors,
            degrees=True,
        )

        best_idx = int(np.argmin(angles))

        best_orientation = equivalents[best_idx]
        best_vector = candidate_vectors[best_idx]

        branch_orientations.append(best_orientation)
        previous_vector = best_vector

    branch_orientations = Orientation(
        np.stack([orientation.data.squeeze() for orientation in branch_orientations]),
        symmetry=symmetry,
    )

    branch_vectors = branch_orientations * reference_direction

    branch_vectors = orient_vectors_to_plot_hemisphere(
        branch_vectors,
    )

    jumps = neighbour_angle_jumps(
        branch_vectors,
    )

    return {
        "tilts": tilts,
        "orientations": branch_orientations,
        "vectors": branch_vectors,
        "chosen_idx": chosen_idx,
        "branch_type": "stereographic_branch",
        "jumps": jumps,
        "max_vector_jump": float(np.max(jumps)),
        "representative_pixel": result.get(
            "representative_pixel",
            result.get("pixel", None),
        ),
    }

# Build stereographic branches and fitted arc poles for all twin regions
def build_stereographic_branches(
    results,
    twin_keys,
    symmetry,
):
    vectors_by_twin = {}
    arc_poles = {}

    for twin_key in twin_keys:

        branch = get_stereographic_branch(
            results=results,
            twin_key=twin_key,
            symmetry=symmetry,
        )

        vectors_by_twin[twin_key] = branch
        arc_poles[twin_key] = fit_arc_pole(
            branch["vectors"],
        )

    return vectors_by_twin, arc_poles


def plot_ipf_overview(
    results,
    twin_keys,
    point_group,
    title="IPF overview",
):
    cmap = plt.cm.plasma

    fig = plt.figure(figsize=(14, 4))

    for i, twin_key in enumerate(twin_keys):

        tilts, orientations, _ = get_selected_orientations(
            results,
            twin_key,
        )

        norm = plt.Normalize( # normalize tilts for consistent colouring across subplots
            vmin=tilts.min(),
            vmax=tilts.max(),
        )

        colours = cmap(norm(tilts))

        ax = fig.add_subplot(
            1,
            len(twin_keys),
            i + 1,
            projection="ipf",
            symmetry=point_group,
        )

        ax.scatter(
            orientations,
            color=colours,
            s=90,
            alpha=0.9,
        )

        ax.set_title(
            f"Twin {twin_key}",
            fontsize=12,
        )

    fig.subplots_adjust(
        right=0.88,
    )

    cax = fig.add_axes(
        [0.91, 0.18, 0.015, 0.64],
    )

    sm = mpl.cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        cax=cax,
    )

    cbar.set_label("Tilt")
    cbar.set_ticks(tilts)
    cbar.set_ticklabels([str(t) for t in tilts])

    fig.suptitle(
        title,
        fontsize=14,
    )

    return fig

# plot one stereographic trajectory
def plot_stereographic_trajectory(
    vectors,
    tilts,
    title="Stereographic trajectory",
    cmap_name="plasma",
    marker_size=120,
    label_points=True,
):
    cmap = plt.get_cmap(cmap_name)

    norm = plt.Normalize(
        vmin=tilts.min(),
        vmax=tilts.max(),
    )

    colours = cmap(norm(tilts))

    fig = plt.figure(figsize=(7, 6))

    ax = fig.add_subplot(111, projection="stereographic")

    ax.scatter(
        vectors,
        color=colours,
        s=marker_size,
        alpha=0.95,
    )

    if label_points:

        for vector, tilt in zip(vectors, tilts):
            ax.text(
                vector,
                s=str(tilt),
                size=10,
                offset=(0, 0.04),
            )

    ax.set_title(
        title,
        fontsize=14,
    )

    ax.stereographic_grid(True)

    sm = mpl.cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        shrink=0.85,
    )

    cbar.set_label("Tilt")

    plt.tight_layout()

    return fig

# Plot stereographic trajectories, fitted great circles, and arc poles.
def plot_stereographic_arcs(
    twin_list,
    vectors_by_twin,
    arc_poles,
    colors,
    title,
    show_labels=False,
):
    
    first_key = twin_list[0]

    fig = vectors_by_twin[first_key]["vectors"].scatter(
        c=colors[first_key],
        s=90,
        axes_labels=["RD", "TD", None], # rolling direction, transverse direction, and no label for the z-axis (which points out of the plane)
        show_hemisphere_label=True,
        return_figure=True,
    )

    ax = fig.axes[0]

    for twin_key in twin_list[1:]:

        vectors_by_twin[twin_key]["vectors"].scatter(
            figure=fig,
            c=colors[twin_key],
            s=90,
        )

    for twin_key in twin_list:

        branch = vectors_by_twin[twin_key]

        if show_labels:
            for vector, tilt in zip(branch["vectors"], branch["tilts"]):
                ax.text(
                    vector,
                    s=str(tilt),
                    size=8,
                    offset=(0, 0.035),
                )

        pole = pole_for_plot(
            arc_poles[twin_key],
        )

        pole.scatter(
            figure=fig,
            c=colors[twin_key],
            marker="*",
            s=250,
            reproject=True,
        )

        pole.draw_circle(
            figure=fig,
            color=colors[twin_key],
            linewidth=2,
            reproject=True,
        )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[twin_key],
            markersize=9,
            label=f"Twin {twin_key}",
        )
        for twin_key in twin_list
    ]

    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
    )

    ax.set_title(title)
    ax.stereographic_grid(True)

    return fig

# Calculate angular separations between fitted arc poles
def calculate_arc_pole_angles(
    arc_poles,
    twin_keys,
):
    arc_angle_dict = {}

    for a, b in itertools.combinations(twin_keys, 2):

        angle = float(
            arc_poles[a].angle_with(
                arc_poles[b],
                degrees=True,
            )
        )

        arc_angle_dict[f"{a}{b}"] = {
            "angle": angle,
            "smallest_equivalent": smallest_axis_angle(angle),
        }

    return arc_angle_dict

# Convert a 3D vector direction to approximate integer [uvw]
def vector_to_uvw(
    vector,
    max_denominator=8,
):
    
    data = np.asarray(vector.data).reshape(3)
    data = data / np.linalg.norm(data)

    nonzero = np.where(np.abs(data) > 1e-8)[0]

    if len(nonzero) == 0:
        return np.array([0, 0, 0], dtype=int)

    if data[nonzero[0]] < 0:
        data = -data

    scale_index = np.argmax(np.abs(data))

    scaled = data / data[scale_index]

    fractions = [Fraction(float(x)).limit_denominator(max_denominator) for x in scaled] # convert the scaled vector components to fractions with limited denominators, which allows us to find a common denominator and convert all components to integers while preserving their ratios

    denominators = [f.denominator for f in fractions] # find the least common multiple of the denominators to convert all fractions to a common denominator

    lcm = np.lcm.reduce(denominators) # the smallest integer that all denominators divide into, allowing us to convert all fractions to integers without losing the relative ratios

    uvw = np.array([int(f.numerator * lcm / f.denominator) for f in fractions])

    nonzero_int = uvw[np.abs(uvw) > 0]

    if len(nonzero_int) > 0:
        gcd = np.gcd.reduce(
            np.abs(nonzero_int),
        )
        uvw = uvw // gcd

    return uvw

#  Return fitted arc poles as unit vectors and approximate [uvw] directions.
def summarize_arc_poles(
    arc_poles,
    twin_keys,
    max_denominator=8,
):
    summary = {}

    for twin_key in twin_keys:

        pole = arc_poles[twin_key]

        data = np.asarray(pole.data).reshape(3)
        data = data / np.linalg.norm(data)

        summary[twin_key] = {
            "pole_xyz": data,
            "uvw": vector_to_uvw(
                pole,
                max_denominator=max_denominator,
            ),
        }

    return summary

def smallest_symmetry_misorientation(
    ori_a,
    ori_b,
    symmetry,
):
    """
    Calculate the smallest symmetry-equivalent misorientation angle.

    Orix has a misorientation function, but it does not consider symmetry equivalences.
    To find the smallest symmetry-equivalent misorientation, we generate all symmetry-equivalent orientations of ori_b and calculate the misorientation angle with ori_a for each equivalent.
    The minimum angle is the smallest symmetry-equivalent misorientation.
    """
    ori_a = Orientation(
        ori_a.data.squeeze(),
        symmetry=symmetry,
    )

    ori_b = Orientation(
        ori_b.data.squeeze(),
        symmetry=symmetry,
    )

    equivalents_b = ori_b.equivalent()

    angles = np.array(
        [
            float(
                ori_a.angle_with(
                    eq,
                    degrees=True,
                )
            )
            for eq in equivalents_b
        ]
    )

    return float(
        angles.min(),
    )


def calculate_pairwise_misorientation_vs_tilt(
    results,
    twin_keys,
    symmetry,
):
    """
    Calculate pairwise smallest symmetry-equivalent misorientation.
    """
    tilts = results[twin_keys[0]]["tilts"]

    misorientation_dict = {}

    for a, b in itertools.combinations(twin_keys, 2):

        key = f"{a}{b}"
        misorientation_dict[key] = []

        _, oris_a, _ = get_selected_orientations(
            results,
            a,
        )

        _, oris_b, _ = get_selected_orientations(
            results,
            b,
        )

        for i, _ in enumerate(tilts):

            misorientation = smallest_symmetry_misorientation(
                oris_a[i],
                oris_b[i],
                symmetry=symmetry,
            )

            misorientation_dict[key].append(
                misorientation,
            )

        misorientation_dict[key] = np.array(
            misorientation_dict[key],
        )

    return tilts, misorientation_dict


def calculate_tilt_step_misorientation(
    results,
    twin_keys,
    symmetry,
):
    """
    Calculate neighbouring-step misorientation for each twin trajectory.
    """
    step_dict = {}

    for twin_key in twin_keys:

        tilts, orientations, _ = get_selected_orientations(
            results,
            twin_key,
        )

        values = []

        for i in range(len(tilts) - 1):

            misorientation = smallest_symmetry_misorientation(
                orientations[i],
                orientations[i + 1],
                symmetry=symmetry,
            )

            values.append(misorientation)

        step_dict[twin_key] = {
            "tilts": tilts,
            "mid_tilts": 0.5 * (tilts[:-1] + tilts[1:]),
            "values": np.array(values),
        }

    return step_dict


def vector_to_uvw_from_array(
    vector,
    max_denominator=8,
):
    """
    Convert an array-like direction to approximate integer [uvw].
    """
    return vector_to_uvw(
        Vector3d(np.asarray(vector).reshape(3)),
        max_denominator=max_denominator,
    )


def smallest_symmetry_misorientation_axis_angle(
    ori_a,
    ori_b,
    symmetry,
):
    """
    Return the smallest symmetry-equivalent misorientation in axis-angle form.
    """
    ori_a = Orientation(
        ori_a.data.squeeze(),
        symmetry=symmetry,
    )

    ori_b = Orientation(
        ori_b.data.squeeze(),
        symmetry=symmetry,
    )

    equivalents_b = ori_b.equivalent()

    angles = np.array(
        [
            float(
                ori_a.angle_with(
                    eq,
                    degrees=True,
                )
            )
            for eq in equivalents_b
        ]
    )

    best_idx = int(np.argmin(angles))
    best_b = equivalents_b[best_idx]

    misorientation = best_b * ~ori_a

    angle = float(
        np.rad2deg(
            np.asarray(misorientation.angle).squeeze(),
        )
    )

    axis = np.asarray(
        misorientation.axis.data,
    ).reshape(3)

    axis = axis / np.linalg.norm(axis)

    return {
        "angle": angle,
        "axis": axis,
        "uvw": vector_to_uvw_from_array(axis),
        "equivalent_index": best_idx,
    }

# Convert an orix Vector3d to a unit vector in numpy array form
def _unit_vector(vector): 
    data = np.asarray(vector.data).reshape(3)
    return data / np.linalg.norm(data)

def _same_pole_sign(moving_pole, reference_pole):
    moving = _unit_vector(moving_pole)
    reference = _unit_vector(reference_pole)

    if np.dot(moving, reference) < 0:
        moving = -moving

    return Vector3d(moving)

# Rotate one trajectory so that its arc pole overlaps a reference pole
def align_branch_pole_to_reference(
    vectors_by_twin,
    arc_poles,
    moving_key,
    reference_key,
):
    """

    The same rotation is applied to all beam-direction vectors in the moving
    trajectory. Related to phase mapping techniques where a common crystallographic direction is aligned across multiple datasets for comparison.
    Here, we align the fitted arc poles of stereographic trajectories to compare their shapes without the confounding effect of different pole orientations.
    Physically, this could correspond to reorienting the sample or the reference frame so that a specific crystallographic direction (the arc pole) is aligned across different twin regions, allowing for a more direct comparison of their stereographic trajectories and misorientation characteristics.
    """
    reference_pole = pole_for_plot(
        arc_poles[reference_key],
    )

    moving_pole = _same_pole_sign(
        arc_poles[moving_key],
        reference_pole,
    )

    candidate_a = Rotation.from_align_vectors(
        reference_pole,
        moving_pole,
    )

    candidate_b = Rotation.from_align_vectors(
        moving_pole,
        reference_pole,
    )

    angle_a = float(
        (candidate_a * moving_pole).angle_with(
            reference_pole,
            degrees=True,
        )
    )

    angle_b = float(
        (candidate_b * moving_pole).angle_with(
            reference_pole,
            degrees=True,
        )
    )

    if angle_a <= angle_b:
        rotation = candidate_a
    else:
        rotation = candidate_b

    aligned_branch = vectors_by_twin[moving_key].copy()
    aligned_branch["vectors"] = rotation * vectors_by_twin[moving_key]["vectors"]
    aligned_branch["branch_type"] = (
        vectors_by_twin[moving_key]["branch_type"]
        + f"_aligned_to_{reference_key}"
    )

    aligned_pole = rotation * arc_poles[moving_key]

    angle_after = float(
        arc_poles[reference_key].angle_with(
            aligned_pole,
            degrees=True,
        )
    )

    angle_after = smallest_axis_angle(
        angle_after,
    )

    return aligned_branch, aligned_pole, rotation, angle_after

def save_stereographic_summary(
    path,
    twin_keys,
    arc_poles,
    arc_angle_dict,
    misorientation_dict,
    step_misorientation=None,
):
    """
    Save stereographic and misorientation results to JSON.
    """
    arc_pole_summary = summarize_arc_poles(
        arc_poles,
        twin_keys,
    )

    summary = {
        "arc_poles": {
            key: {
                "pole_xyz": [float(x) for x in value["pole_xyz"]],
                "uvw": [int(x) for x in value["uvw"]],
            }
            for key, value in arc_pole_summary.items()
        },
        "arc_pole_angles": {
            key: {
                "angle": float(value["angle"]),
                "smallest_equivalent": float(value["smallest_equivalent"]),
            }
            for key, value in arc_angle_dict.items()
        },
        "misorientation_vs_tilt": {
            key: [float(x) for x in value] for key, value in misorientation_dict.items()
        },
    }

    if step_misorientation is not None:
        summary["tilt_step_misorientation"] = {
            key: [float(x) for x in value["values"]]
            for key, value in step_misorientation.items()
        }

    with open(path, "w") as f:
        json.dump(
            summary,
            f,
            indent=4,
        )

    return summary
def twin_angle_error(ori_a, ori_b):
    angle = ori_a.angle_with(ori_b, degrees=True)
    angle = float(np.asarray(angle).squeeze())
    return angle, abs(angle - 60)

def align_branch_pole_to_center(
    branch,
    pole,
    center_direction=Vector3d.zvector(),
):
    """
    Rotate one stereographic branch so that its fitted arc pole is aligned
    with the centre of the stereogram.

    The arc pole is the normal of the fitted great-circle plane. When this pole
    is aligned with the stereographic centre, the corresponding great circle is
    displayed close to the outer edge of the stereogram. This gives a view
    similar to classical manual stereogram analysis.

    Parameters
    ----------
    branch : dict
        One entry from vectors_by_twin. Must contain "vectors".
    pole : Vector3d
        Fitted arc pole for this branch.
    center_direction : Vector3d
        Direction to align the pole to. Default is z, corresponding to the
        centre of the stereographic projection.

    Returns
    -------
    aligned_branch : dict
        Copy of the branch with rotated vectors.
    aligned_pole : Vector3d
        Rotated arc pole.
    rotation : Rotation
        Rotation used for the alignment.
    angle_after : float
        Angle between aligned pole and centre direction, in degrees.
    """

    pole = pole_for_plot(pole)
    center_direction = pole_for_plot(center_direction)

    candidate_a = Rotation.from_align_vectors(
        center_direction,
        pole,
    )

    candidate_b = Rotation.from_align_vectors(
        pole,
        center_direction,
    )

    angle_a = float(
        (candidate_a * pole).angle_with(
            center_direction,
            degrees=True,
        )
    )

    angle_b = float(
        (candidate_b * pole).angle_with(
            center_direction,
            degrees=True,
        )
    )

    if angle_a <= angle_b:
        rotation = candidate_a
    else:
        rotation = candidate_b

    aligned_branch = branch.copy()
    aligned_branch["vectors"] = rotation * branch["vectors"]
    aligned_branch["branch_type"] = (
        branch.get("branch_type", "branch")
        + "_pole_aligned_to_center"
    )

    aligned_pole = rotation * pole

    angle_after = float(
        aligned_pole.angle_with(
            center_direction,
            degrees=True,
        )
    )

    angle_after = smallest_axis_angle(angle_after)

    return aligned_branch, aligned_pole, rotation, angle_after

def align_all_branch_poles_to_center(
    vectors_by_twin,
    arc_poles,
    twin_keys,
    center_direction=Vector3d.zvector(),
):
    """
    Align each twin trajectory independently so that its fitted arc pole lies
    at the stereogram centre.

    This produces a classical stereogram-style comparison where each trajectory
    lies near the outer edge of the projection.
    """

    aligned_vectors_by_twin = {}
    aligned_arc_poles = {}
    rotations = {}
    alignment_errors = {}

    for twin_key in twin_keys:

        aligned_branch, aligned_pole, rotation, angle_after = (
            align_branch_pole_to_center(
                branch=vectors_by_twin[twin_key],
                pole=arc_poles[twin_key],
                center_direction=center_direction,
            )
        )

        aligned_vectors_by_twin[twin_key] = aligned_branch
        aligned_arc_poles[twin_key] = aligned_pole
        rotations[twin_key] = rotation
        alignment_errors[twin_key] = angle_after

    return (
        aligned_vectors_by_twin,
        aligned_arc_poles,
        rotations,
        alignment_errors,
    )

def plot_center_aligned_stereographic_arcs(
    twin_list,
    aligned_vectors_by_twin,
    aligned_arc_poles,
    colors,
    title="Pole-centred stereographic trajectories",
    show_labels=True,
):
    """
    Plot trajectories after aligning each branch arc pole to the stereogram
    centre. This places the fitted great circles near the edge of the
    stereogram, making the plot resemble manual classical stereogram analysis.
    """

    first_key = twin_list[0]

    fig = aligned_vectors_by_twin[first_key]["vectors"].scatter(
        c=colors[first_key],
        s=90,
        axes_labels=["RD", "TD", None],
        show_hemisphere_label=True,
        return_figure=True,
    )

    ax = fig.axes[0]

    for twin_key in twin_list[1:]:

        aligned_vectors_by_twin[twin_key]["vectors"].scatter(
            figure=fig,
            c=colors[twin_key],
            s=90,
        )

    for twin_key in twin_list:

        branch = aligned_vectors_by_twin[twin_key]

        if show_labels:
            for vector, tilt in zip(branch["vectors"], branch["tilts"]):
                ax.text(
                    vector,
                    s=str(tilt),
                    size=8,
                    offset=(0, 0.035),
                )

        pole = pole_for_plot(
            aligned_arc_poles[twin_key],
        )

        pole.scatter(
            figure=fig,
            c=colors[twin_key],
            marker="*",
            s=250,
            reproject=True,
        )

        pole.draw_circle(
            figure=fig,
            color=colors[twin_key],
            linewidth=2,
            reproject=True,
        )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[twin_key],
            markersize=9,
            label=f"Twin {twin_key}",
        )
        for twin_key in twin_list
    ]

    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
    )

    ax.set_title(title)
    ax.stereographic_grid(True)

    return fig
