#!/usr/bin/env python3
"""
Détection et lecture de plaques avec YOLOv8 et EasyOCR
"""

import os
import cv2
import torch
import argparse
from ultralytics import YOLO
import easyocr

# === Patch PyTorch 2.6+ pour weights_only=False ===
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

# --- OCR helper ---
reader = easyocr.Reader(['en'])  # English letters/numbers

def ocr_easyocr(crop_bgr):
    """Lit le texte d'une plaque avec EasyOCR"""
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    results = reader.readtext(img_rgb)
    if results:
        # Prendre le texte avec la plus haute confiance
        results = sorted(results, key=lambda x: x[2], reverse=True)
        return results[0][1].upper()  # texte en majuscules
    return ""

def save_crop_and_ocr(crop, out_dir, idx):
    """Sauvegarde le crop et effectue l'OCR"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"plate_{idx}.png")
    cv2.imwrite(path, crop)
    text = ocr_easyocr(crop)
    return path, text

def main(args):
    print(f"[INFO] Chargement du modèle: {args.model}")
    model = YOLO(args.model)
    
    print(f"[INFO] Traitement de la source: {args.source}")
    results = model(args.source, imgsz=1280, conf=args.conf_thres, device=args.device)
    
    idx = 0
    for res in results:
        frame = res.orig_img.copy()
        boxes = res.boxes
        
        if boxes is None or len(boxes) == 0:
            print("[INFO] Aucune plaque détectée dans cette frame")
            continue
        
        print(f"[INFO] {len(boxes)} plaque(s) détectée(s)")
        
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(b.conf[0])
            
            # Dessiner le rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Extraire et analyser la plaque
            crop = frame[y1:y2, x1:x2]
            if crop.size != 0:
                path, text = save_crop_and_ocr(crop, args.crop_dir, idx)
                print(f"  - Plaque {idx+1}: '{text}' (confiance: {conf:.2f})")
                idx += 1
                
                # Afficher le texte sur l'image
                label = f"{text} ({conf:.2f})"
                cv2.putText(frame, label, (x1, y2 + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
        
        # Affichage en temps réel
        if args.display:
            cv2.imshow("Detections", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Arrêt demandé par l'utilisateur")
                break
        
        # Sauvegarde pour les images
        if not args.source.endswith((".mp4", ".avi", ".mov", ".mkv")):
            cv2.imwrite(args.out, frame)
            print(f"[INFO] Résultat sauvegardé: {args.out}")
    
    print(f"\n[TERMINÉ] {idx} plaque(s) détectée(s) et sauvegardée(s) dans '{args.crop_dir}'")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Détection et OCR de plaques d'immatriculation avec YOLOv8 + EasyOCR"
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
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n[INFO] Interruption par l'utilisateur")
    except Exception as e:
        print(f"\n[ERREUR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

