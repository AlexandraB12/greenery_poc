# Mesures simples à partir d’une image (sans deep learning)
# # Sert à donner des scores compréhensibles pour le modèle.
# greenery_ratio = proxy “taux de verdure”
# luminance_score = proxy “luminosité”

# calcule greenery + luminance 🌿☀️
import numpy as np

def luminance_score(rgb: np.ndarray) -> float:     # Fonction luminance_score — mesurer la luminosité réelle
    """
    rgb: uint8 array (H, W, 3)     # shape = (hauteur, largeur, 3 couleurs)
    Returns 0-100
    """
    x = rgb.astype(np.float32) / 255.0    # normalisation des pixels
    # Rec. 709 luma
    y = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]   # calcul de la luminance oeil humain
    # luminosité = mélange pondéré des 3 couleurs (Rouge -> 21%, Vert -> 71%, Bleu -> 7%)
    return float(np.clip(y.mean() * 100.0, 0.0, 100.0))    # moyenne de l’image entière (0-100, score luminosité)

def greenery_ratio(rgb: np.ndarray) -> float:     # Fonction greenery_ratio — détecter la verdure simplement
    """
    Very simple greenery proxy: "green channel significantly dominates"
    Returns 0-100
    """
    x = rgb.astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]   # Séparation des 3 couches couleur
    green_mask = (g > r + 15) & (g > b + 15) & (g > 60)  # heuristic
    ratio = green_mask.mean()
    return float(np.clip(ratio * 100.0, 0.0, 100.0))    # pourcentage de verdure visible

