import easyocr
import re

def extraire_plaque_maroc_generique(chemin_image):
    # Initialisation de l'OCR (Arabe + Anglais)
    reader = easyocr.Reader(['ar', 'en'], gpu=False)
    
    # Lecture brute
    results = reader.readtext(chemin_image, detail=0)
    
    # 1. Dictionnaire de correspondance (Mapping)
    # On ajoute les lettres que vous avez listées : أ ب د هـ و ط ي
    mapping = {
        'A': 'أ', 'B': 'ب', 'D': 'د', 'J': 'د', 
        'H': 'هـ', 'W': 'و', 'T': 'ط', 'Y': 'ي', 'I': 'ي',
        '٥': '8', 'V': '7' # Corrections de chiffres fréquents
    }

    plaque_propre = []
    
    for bloc in results:
        # Nettoyage des barres verticales souvent lues comme des '1' ou '|'
        bloc = bloc.replace('|', '').strip()
        
        # Si le bloc est une seule lettre (souvent mal lue en Latin)
        if len(bloc) == 1 and not bloc.isdigit():
            # On cherche dans notre dictionnaire de correspondance
            lettre_corrigee = mapping.get(bloc.upper(), bloc)
            plaque_propre.append(lettre_corrigee)
        else:
            # Si c'est un bloc de chiffres, on s'assure qu'ils sont en format latin
            # (Certains OCR mélangent les chiffres 123 et ١٢٣)
            chiffres_corriges = bloc.replace('٥', '8').replace('١', '1').replace('٩', '9')
            plaque_propre.append(chiffres_corriges)

    return " | ".join(plaque_propre)

# --- Test sur n'importe quelle image ---
print(extraire_plaque_maroc_generique('crops/plate_0.png'))