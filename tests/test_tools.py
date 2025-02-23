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
