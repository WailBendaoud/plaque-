import easyocr
import cv2
import numpy as np
import os
import re

class MoroccanOCR:
    def __init__(self):
        # Initialisation silencieuse de l'OCR
        self.reader = easyocr.Reader(['ar', 'en'], gpu=False)
        self.lettres_valides = "أبدهوطي"
        # Mapping des confusions OCR les plus fréquentes
        self.mapping = {
            'J': 'د', 'D': 'د', 'ل': 'د', 
            'A': 'أ', 'B': 'ب', 'H': 'هـ', 
            'W': 'و', '9': 'و', 'g': 'و', 
            'T': 'ط', 'b': 'ط', 'Y': 'ي', 
            'I': 'ي', '٥': '8', 'V': '7', 'o': '8'
        }

    def preprocess(self, img):
        # Agrandissement pour séparer les caractères collés (ex: 5001 ou 8720)
        img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Amélioration du contraste pour les plaques sombres (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        # Érosion légère pour affiner les traits et éviter les fusions (0 vs 8)
        kernel = np.ones((2,2), np.uint8)
        return cv2.erode(gray, kernel, iterations=1)

    def extraire(self, chemin):
        if not os.path.exists(chemin): return "Erreur: Chemin invalide"
        
        image = cv2.imread(chemin)
        processed = self.preprocess(image)
        
        # Lecture avec paramètres de sensibilité accrus
        results = self.reader.readtext(processed, detail=1, contrast_ths=0.1)
        
        # Tri de gauche à droite
        results = sorted(results, key=lambda x: x[0][0][0])
        
        final_blocks = []
        for (_, text, _) in results:
            # 1. Nettoyage : on ne garde que l'alphanumérique + lettres arabes cibles
            t = "".join([c for c in text if c.isalnum() or c in self.lettres_valides])
            
            # 2. Correction via mapping
            for err, corr in self.mapping.items():
                t = t.replace(err, corr)
            
            # 3. Filtrage strict (chiffres et lettres autorisées uniquement)
            t_clean = "".join([c for c in t if c.isdigit() or c in self.lettres_valides])
            
            if t_clean:
                final_blocks.append(t_clean)
        
        return self.formater_resultat(final_blocks)

    def formater_resultat(self, blocks):
        """Assure que le résultat ressemble à une plaque marocaine"""
        if not blocks: return "Aucune détection"
        # On rejoint les blocs détectés avec le séparateur standard
        return " | ".join(blocks)

# --- EXECUTION ---
ocr = MoroccanOCR()
# Liste de vos fichiers à tester
fichiers = ['plate.png', 'crops/plate_0.png']

for f in fichiers:
    if os.path.exists(f):
        print(f"Fichier: {f} -> Résultat: {ocr.extraire(f)}")