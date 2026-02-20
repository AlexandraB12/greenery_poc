# Traduction IA → humain
# Convertit des différences numériques en phrases :
# “🌿 nettement plus verte (+12) ; ☀️ un peu moins lumineuse (-6) ; ✨ charme proche (+2)”
# crucial pour la confiance investisseur

# qualif_delta — traduire un écart numérique en langage naturel
def qualif_delta(d: float) -> str:   # reçoit d = différence entre deux scores
    ad = abs(d)    # On regarde l’ampleur de la différence, pas le signe (ex : +12.3 de verdure)
    # On classe l’écart
    if ad >= 5:
        return "beaucoup"
    if ad >= 2:
        return "nettement"
    if ad >= 0.5:
        return "un peu"
    return "quasi"

# plus_moins — donner le sens de la différence
def plus_moins(d: float) -> str:
    return "plus" if d > 0 else "moins"   # d > 0 → la zone candidate est meilleure, d < 0 → elle est moins bonne

# explain_zone — assembler une explication complète
def explain_zone(ref_scores: dict, cand_scores: dict) -> str:     # ref_scores → zone de référence (celle de l’utilisateur)
    # cand_scores → zone recommandée
    parts = []
    # Boucle sur les critères expliqués
    for k, label, emoji in [
        ("greenery", "verdure", "🌿"),
        ("luminance", "luminosité", "☀️"),
        ("charm", "charme", "✨"),
    ]:
        # Vérifications de sécurité
        if k in ref_scores and k in cand_scores and ref_scores[k] is not None and cand_scores[k] is not None:
            # Calcul de la différence (Exemple : verdure ref = 18, verdure candidate = 32, d = +14)
            d = cand_scores[k] - ref_scores[k]
            # génération de la phrase
            parts.append(f"{emoji} {qualif_delta(d)} {plus_moins(d)} {label} ({d:+.2f})")
    # assemblage final
    return " ; ".join(parts) if parts else "Profil global proche ✅"  # Cas fallback
    # Si aucune différence significative ou données absentes --> Profil global proche
