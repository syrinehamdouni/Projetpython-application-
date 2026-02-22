import mlflow.pyfunc
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# -------------------
# CONFIG
# -------------------

CLASSES = [
    'battery', 'biological', 'cardboard', 'clothes',
    'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash'
]


MODEL_ID = "m-206d43e6afde4c4f99d00add2c720980"
model_uri = f"models:/{MODEL_ID}"

# -------------------
# LOAD MODEL FROM MLFLOW
# -------------------

model = mlflow.pyfunc.load_model(model_uri)
print(" Model loaded from MLflow using Model ID")

# -------------------
# FASTAPI
# -------------------

app = FastAPI(title="Waste Classification API")

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


@app.get("/")
def home():
    return {"message": "API is running "}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)  # batch dimension

   
    preds = model.predict(img_tensor.numpy())

    preds_tensor = torch.tensor(preds)
    probs = torch.softmax(preds_tensor, dim=1)

    confidence, predicted_class = torch.max(probs, 1)

    return JSONResponse({
        "predicted_class": CLASSES[predicted_class.item()],
        "confidence": float(confidence.item())
    })