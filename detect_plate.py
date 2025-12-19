#!/usr/bin/env python3
"""
Détection et lecture de plaques marocaines avec YOLOv8 et PaddleOCR
Format marocain: 12345-ا-67 (chiffres | lettre arabe | chiffres)
"""

import os
import cv2
import torch
import argparse
import re
from ultralytics import YOLO

# === Patch PyTorch 2.6+ pour weights_only=False ===
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

# Lettres arabes valides pour les plaques marocaines
VALID_ARABIC_LETTERS = ['أ', 'ب', 'د', 'هـ', 'و', 'ط', 'ي']
# Variantes possibles de هـ
VALID_ARABIC_LETTERS_VARIANTS = VALID_ARABIC_LETTERS + ['ه', 'ھ']

print(f"[INFO] Lettres arabes autorisées: {' '.join(VALID_ARABIC_LETTERS)}")

# Import PaddleOCR
try:
    from paddleocr import PaddleOCR
    # Initialiser PaddleOCR avec support multilangue (arabe + anglais)
    ocr = PaddleOCR(use_angle_cls=True, lang='arabic', use_gpu=False, show_log=False)
    USE_PADDLE = True
    print("[INFO] PaddleOCR chargé avec succès")
except ImportError:
    print("[WARNING] PaddleOCR non installé. Installez avec: pip install paddlepaddle paddleocr")
    print("[INFO] Fallback sur EasyOCR...")
    import easyocr
    reader = easyocr.Reader(['en', 'ar'], gpu=False)
    USE_PADDLE = False

def preprocess_plate(crop_bgr):
    """Prétraitement pour améliorer la lecture OCR"""
    # Redimensionner si trop petit
    h, w = crop_bgr.shape[:2]
    if h < 100:
        scale = 100 / h
        crop_bgr = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Convertir en gris
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # CLAHE pour améliorer le contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Débruitage léger
    denoised = cv2.fastNlMeansDenoising(enhanced, h=7)
    
    # Binarisation
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary, cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

def ocr_with_paddle(crop_bgr, debug=False):
    """OCR avec PaddleOCR - meilleur pour les plaques structurées"""
    binary, rgb = preprocess_plate(crop_bgr)
    
    all_texts = []
    all_confs = []
    
    # Essayer sur l'image originale
    try:
        result = ocr.ocr(crop_bgr, cls=True)
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                conf = line[1][1]
                all_texts.append(text)
                all_confs.append(conf)
                if debug:
                    print(f"    [DEBUG] PaddleOCR détecté: '{text}' (conf: {conf:.2f})")
    except:
        pass
    
    # Essayer sur l'image prétraitée
    try:
        result2 = ocr.ocr(binary, cls=True)
        if result2 and result2[0]:
            for line in result2[0]:
                text = line[1][0]
                conf = line[1][1]
                all_texts.append(text)
                all_confs.append(conf)
                if debug:
                    print(f"    [DEBUG] PaddleOCR (prétraité) détecté: '{text}' (conf: {conf:.2f})")
    except:
        pass
    
    if all_texts:
        # Prendre le meilleur résultat
        best_idx = all_confs.index(max(all_confs))
        return all_texts[best_idx], all_confs[best_idx]
    
    return "", 0.0

def ocr_with_easyocr(crop_bgr, debug=False):
    """OCR avec EasyOCR - fallback"""
    binary, rgb = preprocess_plate(crop_bgr)
    
    results = []
    
    # Essayer les deux versions
    for img in [rgb, cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)]:
        try:
            detections = reader.readtext(img, detail=1, paragraph=False)
            results.extend(detections)
        except:
            pass
    
    if results:
        # Trier par position horizontale (gauche à droite)
        results = sorted(results, key=lambda x: x[0][0][0])
        
        # Extraire les textes
        texts = []
        confs = []
        for r in results:
            text = r[1].strip()
            conf = r[2]
            if text:
                texts.append(text)
                confs.append(conf)
                if debug:
                    print(f"    [DEBUG] EasyOCR détecté: '{text}' (conf: {conf:.2f})")
        
        if texts:
            full_text = " ".join(texts)
            avg_conf = sum(confs) / len(confs)
            return full_text, avg_conf
    
    return "", 0.0

def validate_arabic_letter(letter):
    """Valide si la lettre arabe est autorisée sur les plaques marocaines"""
    return letter in VALID_ARABIC_LETTERS_VARIANTS

def find_closest_arabic_letter(letter):
    """
    Trouve la lettre arabe valide la plus proche
    Utile pour corriger les erreurs OCR
    """
    # Mapping des erreurs courantes
    corrections = {
        'ت': 'ب',  # Souvent confondu
        'ث': 'ب',
        'ن': 'ب',
        'ح': 'ط',
        'خ': 'ط',
        'ج': 'ط',
        'ذ': 'د',
        'ر': 'و',
        'ز': 'و',
        'س': 'و',
        'ش': 'و',
        'ص': 'و',
        'ض': 'و',
        'ع': 'و',
        'غ': 'و',
        'ف': 'و',
        'ق': 'و',
        'ك': 'ي',
        'ل': 'ي',
        'م': 'ي',
        'ه': 'هـ',
        'ھ': 'هـ',
    }
    
    if letter in corrections:
        return corrections[letter]
    
    # Si la lettre est déjà valide
    if validate_arabic_letter(letter):
        if letter in ['ه', 'ھ']:
            return 'هـ'  # Normaliser vers la forme standard
        return letter
    
    # Par défaut, retourner la première lettre valide
    return VALID_ARABIC_LETTERS[0]

def clean_moroccan_text(text):
    """Nettoie et formate le texte de la plaque"""
    # Supprimer les caractères parasites
    text = re.sub(r'[|\/\\\-_.:,;]+', ' ', text)
    
    # Garder chiffres, lettres arabes et espaces
    text = re.sub(r'[^\d\u0600-\u06FF\s]+', ' ', text)
    
    # Normaliser les espaces
    text = ' '.join(text.split())
    
    return text.strip()

def format_plate_text(text, debug=False):
    """Formate le texte selon le format marocain avec validation des lettres arabes"""
    cleaned = clean_moroccan_text(text)
    
    # Extraire composants
    parts = cleaned.split()
    numbers = []
    arabic = []
    
    for part in parts:
        # Chercher les chiffres
        nums = re.findall(r'\d+', part)
        numbers.extend(nums)
        
        # Chercher les lettres arabes
        arabs = re.findall(r'[\u0600-\u06FF]', part)
        for arab_letter in arabs:
            # Valider et corriger la lettre arabe
            if validate_arabic_letter(arab_letter):
                # Normaliser هـ
                if arab_letter in ['ه', 'ھ']:
                    arabic.append('هـ')
                else:
                    arabic.append(arab_letter)
                if debug:
                    print(f"    [DEBUG] Lettre arabe valide: '{arab_letter}'")
            else:
                # Essayer de corriger
                corrected = find_closest_arabic_letter(arab_letter)
                arabic.append(corrected)
                if debug:
                    print(f"    [DEBUG] Lettre arabe '{arab_letter}' corrigée en '{corrected}'")
    
    # Construire le format
    result_parts = []
    
    if len(numbers) >= 1:
        result_parts.append(numbers[0])
    
    if arabic:
        # Prendre la première lettre arabe valide trouvée
        result_parts.append(arabic[0])
    
    if len(numbers) >= 2:
        result_parts.append(numbers[-1])
    elif len(numbers) == 1 and len(numbers[0]) > 3:
        # Si un seul long nombre, essayer de le diviser
        num = numbers[0]
        if len(num) >= 5:
            # Format probable: 12345 67 -> 12345 et 67
            result_parts[0] = num[:-2]
            result_parts.append(num[-2:])
    
    return " ".join(result_parts) if result_parts else cleaned

def ocr_moroccan_plate(crop_bgr, debug=False):
    """Lit une plaque marocaine"""
    if USE_PADDLE:
        text, conf = ocr_with_paddle(crop_bgr, debug=debug)
    else:
        text, conf = ocr_with_easyocr(crop_bgr, debug=debug)
    
    if text:
        formatted = format_plate_text(text, debug=debug)
        if debug:
            print(f"    [DEBUG] Texte brut: '{text}'")
            print(f"    [DEBUG] Texte formaté: '{formatted}'")
        return formatted, conf
    
    return "", 0.0

def save_crop_and_ocr(crop, out_dir, idx, debug=False):
    """Sauvegarde le crop et effectue l'OCR"""
    os.makedirs(out_dir, exist_ok=True)
    
    # Sauvegarder l'image
    path = os.path.join(out_dir, f"plate_{idx}.png")
    cv2.imwrite(path, crop)
    
    # Sauvegarder version prétraitée en debug
    if debug:
        binary, _ = preprocess_plate(crop)
        path_prep = os.path.join(out_dir, f"plate_{idx}_preprocessed.png")
        cv2.imwrite(path_prep, binary)
    
    # OCR
    text, confidence = ocr_moroccan_plate(crop, debug=debug)
    
    return path, text, confidence

def main(args):
    print(f"[INFO] Chargement du modèle: {args.model}")
    model = YOLO(args.model)
    
    print(f"[INFO] Traitement de la source: {args.source}")
    print(f"[INFO] Moteur OCR: {'PaddleOCR' if USE_PADDLE else 'EasyOCR'}")
    print(f"[INFO] Lettres arabes autorisées: {' '.join(VALID_ARABIC_LETTERS)}")
    
    # Déterminer si c'est une webcam
    is_webcam = args.source.isdigit()
    if is_webcam:
        args.source = int(args.source)
    
    results = model(args.source, imgsz=1280, conf=args.conf_thres, device=args.device, stream=True)
    
    idx = 0
    for res in results:
        frame = res.orig_img.copy()
        boxes = res.boxes
        
        if boxes is None or len(boxes) == 0:
            if args.display:
                cv2.imshow("Detections", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue
        
        print(f"\n[INFO] {len(boxes)} plaque(s) détectée(s)")
        
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(b.conf[0])
            
            # Dessiner le rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Extraire et analyser la plaque
            crop = res.orig_img[y1:y2, x1:x2]
            if crop.size != 0:
                if args.debug:
                    print(f"\n  Analyse de la plaque {idx+1}:")
                
                path, text, ocr_conf = save_crop_and_ocr(crop, args.crop_dir, idx, debug=args.debug)
                print(f"  ✓ Plaque {idx+1}: '{text}' (détection: {conf:.2f}, OCR: {ocr_conf:.2f})")
                idx += 1
                
                # Afficher le texte sur l'image
                label = f"{text}"
                label_conf = f"({conf:.2f})"
                
                # Fond vert pour le texte
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                (cw, ch), _ = cv2.getTextSize(label_conf, font, 0.6, 2)
                
                # Rectangle de fond
                padding = 8
                cv2.rectangle(
                    frame,
                    (x1, y2 + 5),
                    (x1 + max(tw, cw) + padding * 2, y2 + th + ch + padding * 3),
                    (0, 255, 0),
                    -1
                )
                
                # Texte de la plaque
                cv2.putText(
                    frame, label, (x1 + padding, y2 + th + padding),
                    font, font_scale, (0, 0, 0), thickness
                )
                
                # Confiance
                cv2.putText(
                    frame, label_conf, (x1 + padding, y2 + th + ch + padding * 2),
                    font, 0.6, (0, 0, 0), 2
                )
        
        # Affichage en temps réel
        if args.display:
            cv2.imshow("Detections - Plaques Marocaines", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Arrêt demandé par l'utilisateur")
                break
        
        # Sauvegarde pour les images
        if not is_webcam and not str(args.source).endswith((".mp4", ".avi", ".mov", ".mkv")):
            cv2.imwrite(args.out, frame)
            print(f"[INFO] Résultat sauvegardé: {args.out}")
    
    print(f"\n[TERMINÉ] {idx} plaque(s) détectée(s) et sauvegardée(s) dans '{args.crop_dir}'")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Détection et OCR de plaques marocaines avec YOLOv8 + PaddleOCR/EasyOCR"
    )
    parser.add_argument("--source", required=True,
                        help="Source: image (ex: plaque.jpg), vidéo (ex: video.mp4) ou webcam (0)")
    parser.add_argument("--model", default="license_plate_detector.pt",
                        help="Chemin vers le modèle YOLO (.pt)")
    parser.add_argument("--out", default="result.jpg",
                        help="Fichier de sortie pour les images")
    parser.add_argument("--crop-dir", default="crops",
                        help="Dossier pour sauvegarder les plaques détectées")
    parser.add_argument("--conf-thres", type=float, default=0.25,
                        help="Seuil de confiance minimum (0-1)")
    parser.add_argument("--display", action="store_true",
                        help="Afficher les résultats en temps réel")
    parser.add_argument("--device", default="cpu",
                        help="Device de calcul: 'cpu' ou 'cuda:0'")
    parser.add_argument("--debug", action="store_true",
                        help="Afficher les détails de détection OCR")
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n[INFO] Interruption par l'utilisateur")
    except Exception as e:
        print(f"\n[ERREUR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()