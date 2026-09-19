import os
import sys
import urllib.request
import zipfile
from ultralytics import YOLO

DATASET_DIR = "license_plate_dataset"
YAML_FILE = os.path.join(DATASET_DIR, "data.yaml")

def setup_dataset():
    if os.path.exists(YAML_FILE):
        print(f"[✓] Dataset found at {DATASET_DIR}")
        return True
    
    print("\n[!] Dataset Not Found!")
    print("To train the model, you need a YOLO-formatted dataset of license plates.")
    print("Please follow these steps to get a free dataset:")
    print("  1. Go to https://universe.roboflow.com/search?q=license%20plate")
    print("  2. Select a dataset and click 'Download Dataset'")
    print("  3. Choose 'YOLOv8' format and download the ZIP file.")
    print(f"  4. Extract the ZIP file into a folder named '{DATASET_DIR}' in this directory.")
    print(f"  5. Ensure the file '{YAML_FILE}' exists.")
    print("  6. Run this script again!\n")
    return False

def train_model():
    print("Loading base YOLOv8 nano model (yolov8n.pt)...")
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    print("\nStarting Training Process...")
    print("This may take some time depending on your CPU/GPU.")
    
    # Train the model
    # We use a small number of epochs (10) for demonstration. 
    # For a production model, you would typically use 50-100 epochs.
    results = model.train(
        data=YAML_FILE,
        epochs=10,       
        imgsz=640,       
        batch=16,
        name='license_plate_model'
    )
    
    print("\n[✓] Training Complete!")
    print("Your new custom model weights are saved in: runs/detect/license_plate_model/weights/best.pt")
    print("You can now update your .env or docker-compose.yml to use this new model!")

if __name__ == '__main__':
    if setup_dataset():
        train_model()
