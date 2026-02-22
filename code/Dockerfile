# -------------------
# Base image with PyTorch preinstalled
# -------------------
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy only requirements and app
COPY requirements.txt app.py ./

# Install remaining Python dependencies (lighter ones)
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]