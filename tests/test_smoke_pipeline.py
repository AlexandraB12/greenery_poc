# vérifie que le pipeline tourne sans crash

import os
import pandas as pd
from reco_engine.demo_reco import run_demo


def test_pipeline_end_to_end(tmp_path):
    run_demo()

    assert os.path.exists("zones_index.csv")
    assert os.path.exists("reco_results.csv")

    zi = pd.read_csv("zones_index.csv")
    reco = pd.read_csv("reco_results.csv")

    assert len(zi) > 0
    assert len(reco) > 0
