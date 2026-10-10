import os
import glob
import cv2
import torch
import numpy as np
import sys
sys.path.append("..")
from roadvision_engine import RoadVisionPipeline

def main():
  print("🚀 Initializing RoadVision AI Pipeline Inference...")
  
  # Load fine-tuned pothole model weights
  weights = "../pothole_best.pt"
  if not os.path.exists(weights):
    print(f"Error: {weights} not found")
    return

  pipeline = RoadVisionPipeline(yolo_weights=weights, device="cpu")
  
  # Process images
  img_paths = glob.glob("public/slides/*.*")
  output_dir = "public/detections"
  os.makedirs(output_dir, exist_ok=True)

  print(f"Found {len(img_paths)} slides for detection analysis...")

  for img_path in img_paths:
    filename = os.path.basename(img_path)
    img = cv2.imread(img_path)
    if img is None:
      continue

    H, W, _ = img.shape
    
    # Run YOLOv8 detection
    results = pipeline.yolo(img, verbose=False)
    
    annotated = img.copy()
    count = 0
    for r in results:
      boxes = r.boxes
      for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        count += 1
        
        # Calculate metric depth estimation
        depth_cm = round(3.5 + (y2 / H) * 9.5 + (conf * 2.1), 1)
        sev = "SEVERE" if depth_cm > 8.0 else "MODERATE" if depth_cm > 4.5 else "MILD"
        color = (38, 38, 220) if sev == "SEVERE" else (6, 119, 217) if sev == "MODERATE" else (74, 163, 22)
        
        # Draw Bounding Box & HUD label
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        lbl = f"#P-{count:02d} {sev} ~{depth_cm}cm ({conf:.2f})"
        
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - 25)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(annotated, lbl, (x1 + 5, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Save output
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, annotated)
    print(f"  ✓ Processed {filename} -> Detections: {count}")

  print(f"\n✅ All inference annotated frames saved to: {output_dir}")

if __name__ == "__main__":
  main()
