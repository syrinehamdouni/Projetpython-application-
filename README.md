# ♻️ Cleanup App

AI-powered smart waste management system combining **Image Classification** and **Text Analysis** for intelligent recycling and complaint monitoring.

---

##  Overview

- Classifies waste images into 10 categories:
battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash
- Analyzes text complaints to detect:
- Category (Collection, Recycling, Hazardous, General)
- Priority (High, Medium, Low)
- Risk (Health, Environmental, Low)
- Supports hybrid decision-making for smart city applications.

---

##  Image Classification

**Model:** CNN with Transfer Learning (pretrained backbone)  
**Dataset:** Custom waste images  
**Performance:**  
- Accuracy: 83.92%  
- Weighted F1-Score: 0.83  

**Pipeline:**
- Image preprocessing & resizing  
- Data augmentation  
- Train/Validation split  
- Weighted loss for class imbalance  
- Evaluation: Accuracy, Confusion Matrix, F1-score  

---

##  NLP Waste Report Module

**Tasks:** Multi-task text classification  
- Issue Classification  
- Priority Detection  
- Risk Assessment  

**Models:**  
- Baseline: TF-IDF + Logistic Regression  
- Advanced: LSTM Multi-output  
- Experimental: BERT (Transformer)  

**Dataset Sources:**  
- NYC 311 Service Requests  
- Kaggle service complaint datasets  
- Custom synthetic text data  

---


---

##  Tech Stack

- Python  
- PyTorch, Torchvision  
- TensorFlow / Keras (optional NLP)  
- Scikit-learn, NLTK, SpaCy  
- NumPy, Pandas, Matplotlib  

---

##  Future Work

- Full backbone fine-tuning  
- Hybrid image + text prediction system  
- Real-time dashboard & GIS integration  
- Web/Mobile deployment  
- Advanced multi-task transformer models  

---

##  Author

Smart Environmental AI Project – Intelligent Waste Management 🌱


