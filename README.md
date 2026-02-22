# ♻️ Cleanup App

AI-powered smart waste management system combining **Image Classification** and **Text Analysis** for intelligent recycling and complaint monitoring.

---

## Overview

Cleanup App is an AI-based solution designed to:

- Classify waste images into 10 categories  
- Analyze citizen complaints using NLP  
- Support smart recycling and environmental monitoring  

### Waste Categories
`battery • biological • cardboard • clothes • glass • metal • paper • plastic • shoes • trash`

---

## Image Classification Module

**Architecture:** Transfer Learning (EfficientNet backbone)  
**Framework:** PyTorch  
**Validation Accuracy:** 83.92%  
**Weighted F1-Score:** 0.83  

### Training Setup
- 15,806 training images  
- 3,956 validation images  
- Weighted CrossEntropy Loss (class imbalance handling)  
- MLflow for experiment tracking & model logging  

---

##  NLP Waste Report Module

Multi-task text classification for:

- Issue Type Detection  
- Priority Classification  
- Risk Assessment  

**Models Used:**
- TF-IDF + Logistic Regression (baseline)  
- LSTM (multi-output)  
- BERT (experimental transformer model)  

**Data Sources:** NYC 311, Kaggle datasets, synthetic reports  

---

##  Tech Stack

- PyTorch  
- Scikit-learn  
- MLflow  
- FastAPI  
- Uvicorn  
- NLTK / SpaCy  

---

##  API (FastAPI)

### Endpoints

- `GET /` – API status  
- `GET /health` – Health check  
- `POST /predict` – Image classification  

### Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload

Swagger UI:
http://127.0.0.1:8000/docs

Future Improvements

Full fine-tuning of backbone

Hybrid Image + Text fusion model

Real-time dashboard integration

Cloud & container deployment

##  Author

Smart Environmental AI Project – Intelligent Waste Management 🌱


