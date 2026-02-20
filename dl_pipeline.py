# dl_pipeline.py
import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1️⃣ Dataset custom
# --------------------------------------------------
class MetricsDataset(Dataset):
    """
    Dataset pour charger les images et scores tabulaires.
    Chaque image est associée à un score cible (target_score ou synthétique).
    """
    def __init__(self, csv_file, img_dir="images", transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # créer luminance si absent
        if "luminance" not in self.df.columns:
            self.df["luminance"] = 0.5*self.df["brightness"] + 0.5*self.df["sky_ratio"]

        # features tabulaires pour supervision (optionnel)
        feature_cols = ["greenery_ratio", "luminance", "visual_complexity", "building_regularity"]
        self.X = self.df[feature_cols].values.astype(np.float32)

        # target score
        if "target_score" not in self.df.columns:
            self.df["target_score"] = 0.4*self.df["greenery_ratio"] + \
                                      0.2*self.df["brightness"] + \
                                      0.2*self.df["sky_ratio"] + \
                                      0.2*self.df["visual_complexity"] - \
                                      0.1*self.df["building_regularity"]
        self.y = self.df["target_score"].values.astype(np.float32)

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
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# --------------------------------------------------
# 3️⃣ CNN modèle simple
# --------------------------------------------------
class CNNScore(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone simple ou prétrainé
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # on garde les embeddings
        # Couche fully connected pour score
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        emb = self.backbone(x)
        score = self.fc(emb)
        return score.squeeze(1), emb  # renvoie score et embedding

# --------------------------------------------------
# 4️⃣ Setup GPU / CPU
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------
# 5️⃣ Dataset & DataLoader
# --------------------------------------------------
dataset = MetricsDataset("metrics_output_with_images.csv", img_dir="streetview_images", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

df_check = pd.read_csv("metrics_output_with_images.csv")
print("Colonnes disponibles :", df_check.columns)
print("Exemple image_filename :", df_check["image_filename"].head())
print("Images trouvées dans streetview_images :", len(os.listdir("streetview_images")))


# --------------------------------------------------
# 6️⃣ Initialiser modèle, loss, optimizer
# --------------------------------------------------
model = CNNScore().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --------------------------------------------------
# 7️⃣ Entraînement simple (10 epochs)
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
all_embeddings = []
all_scores = []

with torch.no_grad():
    for imgs, targets in loader:
        imgs = imgs.to(device)
        scores, emb = model(imgs)
        all_embeddings.append(emb.cpu().numpy())
        all_scores.append(scores.cpu().numpy())

embeddings = np.vstack(all_embeddings)
pred_scores = np.concatenate(all_scores)

# --------------------------------------------------
# 9️⃣ Sauvegarde
# --------------------------------------------------
pd.DataFrame(embeddings).to_csv("dl_embeddings.csv", index=False)
pd.DataFrame({"dl_score": pred_scores}).to_csv("dl_scored.csv", index=False)
print("✅ Embeddings et DL scores sauvegardés")

# Sauvergarde du modèle DL
torch.save(model.state_dict(), "models/streetview_model.pth")
print("✅ Modèle DL sauvegardé pour l’API")
