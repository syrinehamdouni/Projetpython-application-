from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
classes = ['battery','biological','cardboard','clothes',
           'glass','metal','paper','plastic','shoes','trash']

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("models/trash_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

THRESHOLD = 0.7  # أقل probability يعتبر uncertain

@app.get("/")
def home():
    return {"message": "Trash Classification API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # قراءة الصورة
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        
        # Top-3 predictions
        top_probs, top_idx = torch.topk(probs, 3)
        top3 = [{"class": classes[i], "probability": p.item()} 
                for i, p in zip(top_idx[0], top_probs[0])]

    result = {"top3": top3}

    # Warning لو أقل من THRESHOLD
    if top3[0]["probability"] < THRESHOLD:
        result["warning"] = "uncertain prediction"

    return JSONResponse(result)
