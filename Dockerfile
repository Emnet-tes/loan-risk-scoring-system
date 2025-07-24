FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and MLflow model artifacts
COPY src ./src
COPY exported_model ./exported_model

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]