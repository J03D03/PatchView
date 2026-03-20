FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --timeout 120 --retries 3 -r requirements.txt

COPY . .

RUN useradd -u 1000 -m appuser
USER appuser
