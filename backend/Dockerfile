# ========================
# Base: Python 3.11 + CUDA 12.1
# Necesario para llama.cpp con GPU
# ========================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Evita prompts interactivos durante apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Instala Python 3.11 y dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Alias python → python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Instala dependencias primero (mejor cache de Docker)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia el código
COPY . .

# La carpeta de modelos existe pero vacía (se monta como volumen)
RUN mkdir -p models

EXPOSE 8000

# Arranca FastAPI con recarga automática en desarrollo
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]