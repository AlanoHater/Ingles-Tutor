# 🇰🇷 Korean Tutor — App de aprendizaje de coreano

Stack local de AI para aprender coreano con español como idioma base.

## Modelos usados

| Modelo              | Tarea                  | VRAM   |
| ------------------- | ---------------------- | ------ |
| Qwen3.5-2B (Q4_K_M) | LLM tutor              | ~2.5GB |
| Qwen3-ASR-0.6B      | Transcripción de voz   | ~1GB   |
| Kokoro-82M          | Text-to-Speech coreano | CPU    |

## Requisitos

- Docker + Docker Compose
- NVIDIA GPU con 6GB+ VRAM (RTX 4050 Laptop ✅)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Node.js 20+ (para desarrollo frontend sin Docker)

## Setup inicial

### 1. Clona e instala

```bash
git clone <tu-repo>
cd korean-tutor
```

### 2. Variables de entorno

```bash
cp backend/.env.example backend/.env
cp frontend/.env.local.example frontend/.env.local
```

### 3. Descarga el modelo LLM

```bash
# Crea la carpeta si no existe
mkdir -p backend/models

# Descarga desde Hugging Face (GGUF cuantizado Q4_K_M)
huggingface-cli download \
  bartowski/Qwen3.5-2B-GGUF \
  Qwen3.5-2B-Q4_K_M.gguf \
  --local-dir backend/models
```

> Los modelos ASR y TTS se descargan automáticamente la primera vez que inicias el backend.

### 4. Instala llama-cpp-python con soporte CUDA

> Esto se hace dentro del contenedor automáticamente, pero si instalas local:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall
```

### 5. Levanta todo

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Docs API: http://localhost:8000/docs

## Desarrollo frontend sin Docker

```bash
cd frontend
npm install
npm run dev
```

## Deploy en producción

- **Frontend** → Vercel (conecta el repo, agrega `BACKEND_URL` en env vars)
- **Backend** → corre en tu 4060 expuesto via Cloudflare Tunnel
  ```bash
  cloudflared tunnel --url http://localhost:8000
  ```

## Endpoints del backend

| Método | Ruta                   | Descripción                   |
| ------ | ---------------------- | ----------------------------- |
| GET    | `/health`              | Health check                  |
| POST   | `/v1/chat/completions` | Chat con el tutor (streaming) |
| POST   | `/tts`                 | Texto → audio coreano         |
| POST   | `/asr`                 | Audio → texto transcrito      |
