FROM python:3.13-slim

WORKDIR /app

COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY artifacts/exp03/ ./artifacts/exp03/

EXPOSE 8000

CMD ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
