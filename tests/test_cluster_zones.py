# vérifie que le clustering génère bien ses fichiers

import os
from reco_engine.cluster_zones import cluster_zones


def test_cluster_generation():
    cluster_zones(
        zones_csv="zones_index.csv",
        out_csv="outputs/zone_clusters_test.csv",
        out_plot_csv="outputs/zone_clusters_2d_test.csv",
        k=3,
    )

    assert os.path.exists("outputs/zone_clusters_test.csv")
    assert os.path.exists("outputs/zone_clusters_2d_test.csv")
