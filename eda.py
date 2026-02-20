installimport os
print("Working directory:", os.getcwd())

import seaborn as sns
import matplotlib.pyplot as plt

from utils import load_data, get_features

def run_eda():
    # Load data
    df = load_data()

    # Select features
    X = get_features(df)

    # Correlation matrix
    corr = X.corr()

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Pairwise correlations between visual metrics")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_eda()
