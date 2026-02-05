from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import gridfs
import torch
import os
import io
from PIL import Image
import segmentation_models_pytorch as smp
import torchvision.transforms as T

# ------------------ APP INIT ------------------
app = FastAPI()

# ------------------ CORS (Angular) ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ DATABASE ------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["UroScan_DB"]
fs = gridfs.GridFS(db)

# ------------------ LOAD AI MODEL ------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best_stone_model.pth")

model = smp.Unet(
    encoder_name="resnet34",
    in_channels=3,
    classes=3
)

model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ------------------ IMAGE PREPROCESS ------------------
preprocess = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------ ROUTES ------------------

@app.post("/login")
async def login(
    username: str = Form(...),
    password: str = Form(...)
):
    user = db.users.find_one({
        "username": username,
        "password": password
    })

    if not user:
        raise HTTPException(status_code=401, detail="Invalid Credentials")

    return {
        "status": "success",
        "username": user["username"],
        "role": user["role"]
    }


@app.post("/upload_scan")
async def upload_scan(
    patient_id: str = Form(...),
    doctor_id: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # ---------- Save Image to GridFS ----------
        file_bytes = await file.read()
        file_id = fs.put(
            file_bytes,
            filename=file.filename,
            patient_id=patient_id
        )

        # ---------- AI Prediction ----------
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            preds = torch.sigmoid(output)

            # Class index 2 assumed as "stone"
            stone_detected = preds[0, 2].max().item() > 0.5

        # ---------- Save Result ----------
        scan_data = {
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "file_id": file_id,
            "diagnosis": "Stone Detected" if stone_detected else "No Stone Detected",
            "stone_detected": stone_detected,
            "status": "Pending Review"
        }

        result = db.scans.insert_one(scan_data)

        return {
            "message": "Scan processed successfully",
            "scan_id": str(result.inserted_id),
            "result": scan_data["diagnosis"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
