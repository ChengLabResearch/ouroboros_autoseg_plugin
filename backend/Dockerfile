FROM python:3.10.19-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Mitigate pip CVEs (including CVE-2023-5752 and CVE-2025-8869).
RUN python -m pip install --no-cache-dir --upgrade "pip>=25.3"

COPY . .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir "wheel>=0.46.2"
RUN pip install --no-cache-dir .

EXPOSE 8686

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8686"]
