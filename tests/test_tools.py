import nqetools as nqe
from ase.build import molecule
from ase.visualize import view


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
