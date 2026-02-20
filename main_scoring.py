# main_scoring_DL.py
import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1️⃣ Dataset custom
# --------------------------------------------------
class MetricsDataset(Dataset):
    """Dataset pour charger les images et scores tabulaires."""
    def __init__(self, csv_file, img_dir="streetview_images", transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        if "luminance" not in self.df.columns:
            self.df["luminance"] = 0.5*self.df["brightness"] + 0.5*self.df["sky_ratio"]

        feature_cols = ["greenery_ratio", "luminance", "visual_complexity", "building_regularity"]
        self.X = self.df[feature_cols].values.astype(np.float32)

        if "target_score" not in self.df.columns:
            self.df["target_score"] = 0.4*self.df["greenery_ratio"] + \
                                      0.2*self.df["brightness"] + \
                                      0.2*self.df["sky_ratio"] + \
                                      0.2*self.df["visual_complexity"] - \
                                      0.1*self.df["building_regularity"]
        self.y = self.df["target_score"].values.astype(np.float32)

        # Vérification de la colonne image
        if "image_filename" not in self.df.columns:
            image_files = sorted(os.listdir(img_dir))
            if len(image_files) < len(self.df):
                raise ValueError("❌ Pas assez d'images pour toutes les lignes du CSV")
            self.df["image_filename"] = image_files[:len(self.df)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx]["image_filename"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.y[idx])

# --------------------------------------------------
# 2️⃣ Transformations images
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# --------------------------------------------------
# 3️⃣ CNN simple
# --------------------------------------------------
class CNNScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512,1)

    def forward(self, x):
        emb = self.backbone(x)
        score = self.fc(emb)
        return score.squeeze(1), emb

# --------------------------------------------------
# 4️⃣ GPU / CPU
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------
# 5️⃣ Dataset & DataLoader
# --------------------------------------------------
dataset = MetricsDataset("metrics_output_with_images.csv", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# --------------------------------------------------
# 6️⃣ Modèle, loss, optimizer
# --------------------------------------------------
model = CNNScore().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --------------------------------------------------
# 7️⃣ Entraînement simple
# --------------------------------------------------
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _ = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataset):.6f}")

# --------------------------------------------------
# 8️⃣ Extraire embeddings et scores
# --------------------------------------------------
model.eval()
all_embeddings, all_scores = [], []

with torch.no_grad():
    for imgs, _ in loader:
        imgs = imgs.to(device)
        scores, emb = model(imgs)
        all_embeddings.append(emb.cpu().numpy())
        all_scores.append(scores.cpu().numpy())

embeddings = np.vstack(all_embeddings)
pred_scores = np.concatenate(all_scores)

# --------------------------------------------------
# 9️⃣ Sauvegarde embeddings et DL scores
# --------------------------------------------------
pd.DataFrame(embeddings).to_csv("dl_embeddings.csv", index=False)
pd.DataFrame({"dl_score": pred_scores}).to_csv("dl_scored.csv", index=False)
print("✅ Embeddings et DL scores sauvegardés")

# --------------------------------------------------
# 10️⃣ Clustering sur embeddings
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(embeddings)

# KMeans avec 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_scaled)

df_clusters = dataset.df.copy()
df_clusters["cluster"] = labels

# Map cluster → score
cluster_score_map = {0:80,1:60,2:45,3:75}
df_clusters["score_cluster"] = df_clusters["cluster"].map(cluster_score_map)

# Sauvegarde
df_clusters.to_csv("clusters_scored_DL.csv", index=False)
print("🏷️ clusters_scored_DL.csv sauvegardé")

# Profils moyens par cluster
feature_cols = ["greenery_ratio", "luminance", "visual_complexity", "building_regularity"]
cluster_profiles = df_clusters.groupby("cluster")[feature_cols].mean()
cluster_profiles.to_csv("cluster_profiles_DL.csv")
print("📦 cluster_profiles_DL.csv sauvegardé")
print("\n📊 Profils moyens par cluster :\n", cluster_profiles)

# --------------------------------------------------
# 11️⃣ PCA pour visualiser
# --------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
for cluster in sorted(set(labels)):
    plt.scatter(X_pca[labels==cluster,0], X_pca[labels==cluster,1], label=f"Cluster {cluster}")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Visualisation PCA clusters DL")
plt.legend()
plt.savefig("pca_clusters_DL.png", bbox_inches="tight")
plt.show()
print("📊 pca_clusters_DL.png sauvegardé")
