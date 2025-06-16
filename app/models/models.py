import gdown
import os
from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO
from typing import List, Dict

# Path where models will be saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Function to download models from Google Drive
def download_model(model_id: str, model_filename: str):
    url = f"https://drive.google.com/uc?export=download&id={model_id}"
    output_path = os.path.join(BASE_DIR, model_filename)
    gdown.download(url, output_path, quiet=False)
    return output_path

# Initialize Roboflow client
roboflow_client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_MODEL_API_KEY")  # Ensure this is set correctly
)

# Model file IDs from Google Drive
model_id_v8s_pretrained = "1Lg2rokWI6975zidBiI262IHs-PN4NQ2s"  # yolov8s pretrained model ID
model_filename_v8s_pretrained = "yolov8s-pretrained.pt"

model_id_v8s_finetuned = "1X5wVhkpQE6kz0h97tV20sM9hyK7Kw5kF"  # yolov8s fine-tuned model ID
model_filename_v8s_finetuned = "yolov8s-finetuned.pt"

# Download models if they don't exist already
if not os.path.exists(os.path.join(BASE_DIR, model_filename_v8s_pretrained)):
    download_model(model_id_v8s_pretrained, model_filename_v8s_pretrained)

if not os.path.exists(os.path.join(BASE_DIR, model_filename_v8s_finetuned)):
    download_model(model_id_v8s_finetuned, model_filename_v8s_finetuned)

# Load YOLO models using the downloaded files
yolo_model_v8s_pretrained = YOLO(os.path.join(BASE_DIR, model_filename_v8s_pretrained))
yolo_model_v8s_finetuned = YOLO(os.path.join(BASE_DIR, model_filename_v8s_finetuned))

# List of classes to exclude (Vietnamese food categories)
excluded_classes = [
    "Banh_mi", "Banh_trang_tron", "Banh_xeo", "Bun_bo_Hue", "Bun_dau", "Com_tam",
    "Goi_cuon", "Pho", "Hu_tieu", "Xoi", "Pudding"
]

def filter_detections(detections):
    # Filter detections to exclude unwanted classes
    return [d for d in detections if d["label"] not in excluded_classes]


def detect_food_labels(image_path: str, confidence_threshold: float = 0.65) -> List[Dict]:
    detections = []
    model_used = "None"

    # Step 1: Try detection using Roboflow
    roboflow_result = roboflow_client.run_workflow(
        workspace_name="machinelearning-4tyun",
        workflow_id="detect-and-classify",
        images={"image": image_path},
        use_cache=True
    )

    # Deteksi makanan menggunakan Roboflow
    for item in roboflow_result:
        predictions = item.get('detection_predictions', {}).get('predictions', [])
        
        for prediction in predictions:
            label = prediction['class']
            confidence = prediction['confidence']
            
            if confidence >= confidence_threshold:
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "model": "Roboflow",  # Menambahkan informasi model yang digunakan
                })

    # Step 2: If Roboflow fails, try YOLOv8s pretrained model
    if not detections:
        print("Roboflow did not detect any objects with sufficient confidence. Using YOLOv8s pretrained as fallback.")
        model_used = "YOLOv8s Pretrained"
        
        # Deteksi menggunakan YOLOv8s pretrained model
        yolo_results = yolo_model_v8s_pretrained(image_path)
        
        for result in yolo_results:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                label = yolo_model_v8s_pretrained.names[cls_id]
                confidence = float(box.conf[0].item())
                
                if confidence >= confidence_threshold:
                    detections.append({
                        "label": label,
                        "confidence": confidence,
                        "model": "YOLOv8s Pretrained",  # Menambahkan informasi model yang digunakan
                    })

    # Step 3: If YOLOv8s pretrained fails, try the fine-tuned YOLOv8s model
    if not detections:
        print("YOLOv8s Pretrained did not detect any objects with sufficient confidence. Using YOLOv8s Fine-tuned as fallback.")
        model_used = "YOLOv8s Fine-tuned"
        
        # Deteksi menggunakan YOLOv8s fine-tuned model
        yolo_results_finetuned = yolo_model_v8s_finetuned(image_path)
        
        for result in yolo_results_finetuned:
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                label = yolo_model_v8s_finetuned.names[cls_id]
                confidence = float(box.conf[0].item())
                
                if confidence >= confidence_threshold:
                    detections.append({
                        "label": label,
                        "confidence": confidence,
                        "model": "YOLOv8s Fine-tuned",  # Menambahkan informasi model yang digunakan
                    })

    # Apply filter to remove unwanted detections
    detections = filter_detections(detections)

    return detections