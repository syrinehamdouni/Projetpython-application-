import mlflow.pyfunc
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# -------------------
# CONFIG
# -------------------

CLASSES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash"
]

MODEL_ID = "m-206d43e6afde4c4f99d00add2c720980"
model_uri = f"models:/{MODEL_ID}"

# -------------------
# LOAD MODEL
# -------------------

model = mlflow.pyfunc.load_model(model_uri)
print("Model loaded from MLflow")

# -------------------
# FASTAPI INIT
# -------------------

app = FastAPI(title="Waste Classification API")

# -------------------
# CORS (important for Android / Web)
# -------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# IMAGE TRANSFORM
# -------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------
# ROOT
# -------------------

@app.get("/")
def home():
    return {"message": "Waste Classification API is running"}

# -------------------
# PREDICT
# -------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        img_tensor = transform(image)
        img_tensor = img_tensor.unsqueeze(0)

        preds = model.predict(img_tensor.numpy())

        preds_tensor = torch.tensor(preds)

        probs = torch.softmax(preds_tensor, dim=1)

        confidence, predicted_class = torch.max(probs, 1)

        result = {
            "predicted_class": CLASSES[predicted_class.item()],
            "confidence": float(confidence.item())
        }

        return JSONResponse(result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )