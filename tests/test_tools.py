import numpy as np
from ase.build import molecule
from ase.visualize import view

import nqetools as nqe


def test_cluster_atoms():
    print(flush=True)
    # Create an example Atoms object (for instance, two separate water molecules)

    water = molecule("H2O")
    # Translate the second water molecule so it does not overlap with the first.
    water2 = water.copy()
    water2.translate([5, 0, 0])

    # Combine into a single Atoms object.
    combined = water + water2

    # Cluster the atoms.
    clusters = nqe.cluster_atoms(combined)
    print(f"Found {len(clusters)} clusters:", flush=True)
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1} with {len(cluster)} atoms", flush=True)
    assert len(clusters[0]) == 3
    assert len(clusters[1]) == 3
    assert len(clusters) == 2


def test_move_clusters_to_distance():
    print(flush=True)

    idx1 = 0
    idx2 = 0
    target_distance = 10.0  # Desired distance between the target atoms

    # Create two water molecules as clusters
    water1 = molecule("H2O")
    water2 = molecule("H2O")

    # Translate the second water molecule to an arbitrary initial position
    water2.translate([2, 0, 0])  # Initial separation of 2 Angstroms

    # Move the clusters so that the target atoms are at a new separation distance
    moved_atoms = nqe.move_clusters_to_distance(water1, water2,
                                                index1=idx1,
                                                index2=idx2,
                                                target_distance=target_distance)
    view(moved_atoms)

    # Check the distance between the target atoms
    distance = moved_atoms.get_distance(idx1, idx2 + len(water1))
    assert round(distance, 2) == target_distance


def test_move_to_distances():
    print(flush=True)

    index1 = 0
    index2 = 3

    distances = [2.0, 10.0]

    # Create an example Atoms object (for instance, two separate water molecules)
    water = molecule("H2O")
    # Translate the second water molecule so it does not overlap with the first.
    water2 = water.copy()
    water2.translate([5, 0, 0])

    # Combine into a single Atoms object.
    combined = water + water2

    # Move the clusters so that the target atoms are at a new separation distance
    moved_atoms = nqe.move_to_distances(combined,
                                        index1,
                                        index2,
                                        distances)

    # Check the distance between the target atoms
    distance1 = moved_atoms[0].get_distance(index1, index2)
    distance2 = moved_atoms[1].get_distance(index1, index2)
    assert round(distance1, 2) == distances[0]
    assert round(distance2, 2) == distances[1]


def test_get_fes_times():
    print(flush=True)
    dt = 1.0
    n_steps = 5000
    fake_data = [np.array(0.0), np.array(1.0), np.array(2.0), np.array(3.0), np.array(4.0), np.array(5.0)]
    val = nqe.get_fes_times(dt, n_steps, fake_data)
    assert val == fake_data


def test_convert_code_to_string():
    """Tests the conversion of a Python function to its string representation.

    This function defines a simple `add` function and uses the `convert_code_to_string`
    function from the `nqetools` module to convert the `add` function into its string
    representation. The result is then compared to the expected string representation
    using an assertion.

    Asserts:
        The string representation of the `add` function matches the expected string.

    Prints:
        Flushes the output to ensure any print statements are immediately visible.
    """
    print(flush=True)

    def add(a: int, b: int) -> int:
        """Return the sum of *a* and *b*."""
        return a + b

    s = nqe.convert_code_to_string(add)
    assert s == "def add(a: int, b: int) -> int:\n    \"\"\"Return the sum of *a* and *b*.\"\"\"\n    return a + b\n"


def test_get_distance():
    print(flush=True)
    water = molecule("H2O")
    dist = nqe.get_distance(water, 0, 1)
    assert round(dist, 2) == 0.97


def test_closest_corresponding_index():
    print(flush=True)
    water = molecule("H2O")
    water2 = water.copy()
    idx = nqe.closest_corresponding_index(water, water2, 1)
    assert idx == 1


def test_combine_without_overlaps():
    print(flush=True)
    water = molecule("H2O")

    water2 = water.copy()
    water2.translate([5, 0, 0])
    combined = nqe.combine_without_overlaps(water, water2)
    assert len(combined) == 6

    water3 = water.copy()
    water3.translate([0.5, 0, 0])
    combined2 = nqe.combine_without_overlaps(water, water3)
    assert len(combined2) == 3


def test_largest_bonded_cluster_indices():
    print(flush=True)
    # Create an example Atoms object (for instance, two separate water molecules)

    water = molecule("H2O")
    # Translate the second water molecule so it does not overlap with the first.
    water2 = molecule("H2")
    water2.translate([5, 0, 0])

    # Combine into a single Atoms object.
    combined = water + water2

    indices = nqe.largest_bonded_cluster_indices(combined)
    assert len(indices) == 3
    assert indices == [0, 1, 2]
