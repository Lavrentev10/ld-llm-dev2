FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --progress-bar on -r requirements.txt

COPY . .

# RUN mkdir -p /app/models/Qwen-7B && chmod -R 777 /app/models

CMD ["python", "app/main.py"]