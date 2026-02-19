from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from bson import ObjectId
import gridfs
import torch
import os
import io
import numpy as np
import cv2
import base64
from PIL import Image
import segmentation_models_pytorch as smp
import torchvision.transforms as T

# ------------------ APP INIT ------------------
app = FastAPI()

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ DATABASE ------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["UroScan_DB"]
fs = gridfs.GridFS(db)

# ------------------ LOAD MODEL ------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best_stone_model_final.pth")

if not os.path.exists(model_path):
    raise RuntimeError(f"Model file not found at {model_path}")

model = smp.Unet(
    encoder_name="resnet34",
    in_channels=3,
    classes=3
)

model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ------------------ PREPROCESS ------------------
preprocess = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------ LOGIN ------------------
@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    user = db.users.find_one({"username": username, "password": password})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid Credentials")

    return {
        "status": "success",
        "username": user["username"],
        "role": user["role"]
    }

# ------------------ UPLOAD + AI ------------------
@app.post("/upload_scan")
async def upload_scan(
    patient_id: str = Form(...),
    doctor_id: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        file_bytes = await file.read()

        # Save original to GridFS
        file_id = fs.put(file_bytes, filename=file.filename, patient_id=patient_id)

        # Convert original to base64 for doctor UI
        original_base64 = base64.b64encode(file_bytes).decode("utf-8")

        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            mask = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
            stone_conf_map = probs[0, 2].cpu().numpy()

        # -------- Metrics --------
        stone_pixels = int(np.sum(mask == 2))
        stone_detected = stone_pixels > 50
        confidence = round(float(stone_conf_map.max()) * 100, 2)

        if stone_pixels > 500:
            severity = "High"
        elif stone_pixels > 200:
            severity = "Moderate"
        elif stone_pixels > 0:
            severity = "Mild"
        else:
            severity = "None"

        # -------- Segmentation Image --------
        mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        mask_rgb[mask == 1] = [0, 120, 255]
        mask_rgb[mask == 2] = [255, 0, 0]

        mask_resized = cv2.resize(
            mask_rgb,
            (image.width, image.height),
            interpolation=cv2.INTER_NEAREST
        )

        _, buffer = cv2.imencode('.png', mask_resized)
        segmentation_base64 = base64.b64encode(buffer).decode("utf-8")

        # -------- Heatmap --------
        heatmap_norm = stone_conf_map / stone_conf_map.max() if stone_conf_map.max() > 0 else stone_conf_map

        heatmap_resized = cv2.resize(
            heatmap_norm,
            (image.width, image.height)
        )

        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        _, buffer2 = cv2.imencode('.png', heatmap_colored)
        heatmap_base64 = base64.b64encode(buffer2).decode("utf-8")

        # -------- Save to DB --------
        scan_data = {
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "file_id": file_id,
            "diagnosis": "Stone Detected" if stone_detected else "No Stone Detected",
            "stone_pixels": stone_pixels,
            "confidence": confidence,
            "severity": severity,
            "original_image": original_base64,
            "segmentation_image": segmentation_base64,
            "heatmap_image": heatmap_base64,
            "status": "Pending Review"
        }

        db.scans.insert_one(scan_data)

        return {
            "message": "Scan processed successfully",
            "result": scan_data["diagnosis"],
            "stone_pixels": stone_pixels,
            "confidence": confidence,
            "severity": severity,
            "segmentation_image": segmentation_base64,
            "heatmap_image": heatmap_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ GET DOCTOR SCANS ------------------
@app.get("/doctor_scans/{doctor_id}")
def get_doctor_scans(doctor_id: str):
    scans = list(db.scans.find({"doctor_id": doctor_id}))
    for scan in scans:
        scan["_id"] = str(scan["_id"])
        scan["file_id"] = str(scan["file_id"])
    return scans

# ------------------ UPDATE SCAN ------------------
@app.put("/update_scan/{scan_id}")
def update_scan(scan_id: str, status: str = Form(...), notes: str = Form("")):
    result = db.scans.update_one(
        {"_id": ObjectId(scan_id)},
        {"$set": {"status": status, "doctor_notes": notes}}
    )

    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Scan not found")

    return {"message": "Scan updated successfully"}
