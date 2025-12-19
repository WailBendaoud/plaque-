#!/usr/bin/env python3
"""
Détection et lecture de plaques avec YOLOv8 et Tesseract (PyTorch 2.6+ compatible)
Optimisé pour vidéos avec traitement par streaming et frame skipping
"""

import os
import cv2
import torch
import argparse
import pytesseract
from PIL import Image
from ultralytics import YOLO

# Configuration automatique de Tesseract pour Windows
if os.name == 'nt':  # Windows
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Tesseract-OCR\tesseract.exe',
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"[INFO] Tesseract trouvé: {path}")
            break
    else:
        print("[ATTENTION] Tesseract OCR non trouvé !")
        print("Téléchargez-le: https://github.com/UB-Mannheim/tesseract/wiki")

# === CORRECTION PYTORCH 2.6+ ===
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

# --- OCR helper ---
def ocr_tesseract(crop_bgr):
    """Extrait le texte d'une image de plaque avec Tesseract"""
    try:
        img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        text = pytesseract.image_to_string(pil, config='--psm 7')
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        return "[OCR NON DISPONIBLE]"
    except Exception as e:
        return "[ERREUR]"

def save_crop_and_ocr(crop, out_dir, idx, frame_num=None):
    """Sauvegarde le crop et effectue l'OCR"""
    os.makedirs(out_dir, exist_ok=True)
    if frame_num is not None:
        path = os.path.join(out_dir, f"plate_frame{frame_num}_{idx}.png")
    else:
        path = os.path.join(out_dir, f"plate_{idx}.png")
    cv2.imwrite(path, crop)
    text = ocr_tesseract(crop)
    return path, text

def is_video_source(source):
    """Vérifie si la source est une vidéo"""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')
    return isinstance(source, str) and source.lower().endswith(video_extensions)

def main(args):
    print(f"[INFO] Chargement du modèle: {args.model}")
    model = YOLO(args.model)
    
    print(f"[INFO] Traitement de la source: {args.source}")
    
    # Traitement optimisé pour vidéos
    if is_video_source(args.source):
        print(f"[INFO] Mode vidéo activé - Traitement toutes les {args.frame_skip} frames")
        print(f"[INFO] Utilisation du streaming pour éviter la saturation RAM")
        process_video(model, args)
    else:
        process_image(model, args)

def process_video(model, args):
    """Traitement optimisé pour vidéos avec streaming"""
    idx = 0
    frame_count = 0
    total_detections = 0
    
    # IMPORTANT: stream=True pour éviter la saturation RAM
    results = model(args.source, imgsz=1280, conf=args.conf_thres, 
                   device=args.device, stream=True)
    
    for res in results:
        frame_count += 1
        
        # Frame skipping pour accélérer le traitement
        if frame_count % args.frame_skip != 0:
            continue
        
        frame = res.orig_img.copy()
        boxes = res.boxes
        
        if boxes is None or len(boxes) == 0:
            continue
        
        num_plates = len(boxes)
        total_detections += num_plates
        print(f"[Frame {frame_count}] {num_plates} plaque(s) détectée(s)")
        
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(b.conf[0])
            
            # Dessiner le rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Extraire et analyser la plaque
            crop = frame[y1:y2, x1:x2]
            if crop.size != 0:
                path, text = save_crop_and_ocr(crop, args.crop_dir, idx, frame_count)
                print(f"  → Plaque {idx+1}: '{text}' (confiance: {conf:.2f})")
                idx += 1
                
                # Afficher le texte sur l'image
                label = f"{text} ({conf:.2f})"
                cv2.putText(frame, label, (x1, y2 + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
        
        # Affichage en temps réel
        if args.display:
            # Ajouter info frame
            info = f"Frame: {frame_count} | Detections: {total_detections}"
            cv2.putText(frame, info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            cv2.imshow("Detections", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Arrêt demandé par l'utilisateur")
                break
    
    print(f"\n[TERMINÉ] Vidéo traitée:")
    print(f"  - Frames analysées: {frame_count // args.frame_skip}")
    print(f"  - Total plaques détectées: {idx}")
    print(f"  - Crops sauvegardés dans: '{args.crop_dir}'")
    cv2.destroyAllWindows()

def process_image(model, args):
    """Traitement pour images"""
    results = model(args.source, imgsz=1280, conf=args.conf_thres, 
                   device=args.device, stream=False)
    
    idx = 0
    for res in results:
        frame = res.orig_img.copy()
        boxes = res.boxes
        
        if boxes is None or len(boxes) == 0:
            print("[INFO] Aucune plaque détectée")
            continue
        
        print(f"[INFO] {len(boxes)} plaque(s) détectée(s)")
        
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(b.conf[0])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size != 0:
                path, text = save_crop_and_ocr(crop, args.crop_dir, idx)
                print(f"  → Plaque {idx+1}: '{text}' (confiance: {conf:.2f})")
                idx += 1
                
                label = f"{text} ({conf:.2f})"
                cv2.putText(frame, label, (x1, y2 + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
        
        if args.display:
            cv2.imshow("Detections", frame)
            cv2.waitKey(0)
        
        cv2.imwrite(args.out, frame)
        print(f"[INFO] Résultat sauvegardé: {args.out}")
    
    print(f"\n[TERMINÉ] {idx} plaque(s) sauvegardée(s) dans '{args.crop_dir}'")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Détection et OCR de plaques d'immatriculation avec YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Image
  python detect_plate.py --source image.jpg --display
  
  # Vidéo (traite 1 frame sur 5)
  python detect_plate.py --source video.mp4 --display --frame-skip 5
  
  # Vidéo avec GPU
  python detect_plate.py --source video.mp4 --device cuda:0 --frame-skip 10
  
  # Webcam
  python detect_plate.py --source 0 --display --frame-skip 2
        """
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
    parser.add_argument("--frame-skip", type=int, default=5,
                       help="Traiter 1 frame sur N (vidéo uniquement, défaut: 5)")
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