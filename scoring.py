import pandas as pd
import numpy as np

def compute_dl_score(path="dl_embeddings.csv"):
    df = pd.read_csv(path)

    emb_cols = [c for c in df.columns if c.startswith("emb_")]

    df["score_dl"] = np.linalg.norm(df[emb_cols].values, axis=1)

    df.to_csv("dl_scored.csv", index=False)
    print("✅ DL score saved → dl_scored.csv")

    return df
