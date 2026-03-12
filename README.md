# 🇺🇸 English Tutor — App de aprendizaje de inglés

Stack local de AI para aprender inglés con español como idioma base.

## Modelos usados

| Modelo              | Tarea                  | VRAM   |
| ------------------- | ---------------------- | ------ |
| Qwen3.5-2B (Q4_K_M) | LLM tutor              | ~2.5GB |
| Kokoro-82M          | Text-to-Speech inglés  | CPU    |

## Requisitos

- Docker + Docker Compose
- NVIDIA GPU con 6GB+ VRAM (RTX 4050 Laptop ✅)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Node.js 20+ (opcional, solo para desarrollo frontend sin Docker)

## Setup inicial

### 1. Clona e instala

```bash
git clone <tu-repo>
cd english-tutor
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

### 4. Levanta todo

```bash
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Docs API: http://localhost:8000/docs

> ⏱️ El primer build tarda ~10-15 min porque compila `llama-cpp-python` con soporte CUDA dentro del contenedor, aunque las dependencias base bajan rapidísimo gracias al cache de `uv`.

## Optimizaciones GPU

El backend aplica 4 optimizaciones automáticamente para sacarle el máximo a tu GPU:

| Optimización           | Qué hace                                                         | Impacto                     |
| ---------------------- | ---------------------------------------------------------------- | --------------------------- |
| **KV Cache Quant**     | Cuantiza el cache de atención de f16 → q8_0                      | ~400MB VRAM liberados       |
| **Flash Attention**    | Cálculo de atención optimizado                                   | ~20% más rápido             |
| **Prompt Caching**     | Cachea el eval del system prompt entre requests                   | Latencia mínima del 2do msg en adelante |
| **Continuous Batching**| Procesa tokens en batches de 512                                 | Mejor throughput            |

Resultado esperado: **25-40 tokens/segundo** en una RTX 4050 Laptop.

Todas son configurables via variables de entorno:

```bash
LLM_KV_TYPE_K=q8_0       # Tipo KV cache para Keys (default: q8_0)
LLM_KV_TYPE_V=q8_0       # Tipo KV cache para Values (default: q8_0)
LLM_FLASH_ATTN=true      # Flash Attention activado (default: true)
LLM_N_BATCH=512           # Tokens por batch (default: 512)
LLM_CONTEXT_SIZE=4096     # Tamaño de contexto (default: 4096)
LLM_GPU_LAYERS=-1         # Capas en GPU, -1 = todas (default: -1)
```

## Endpoints del backend

| Método | Ruta                   | Descripción                   |
| ------ | ---------------------- | ----------------------------- |
| GET    | `/health`              | Health check + estado de modelos |
| POST   | `/v1/chat/completions` | Chat con el tutor (streaming SSE) |
| POST   | `/tts`                 | Texto en inglés → audio WAV   |
| POST   | `/asr`                 | Audio → texto transcrito      |

## Desarrollo frontend sin Docker

```bash
cd frontend
npm install
npm run dev
```

## CI/CD

El proyecto usa **GitHub Actions** con pipelines para frontend y backend:

- **Backend** (`.github/workflows/ci-backend.yml`): lint con `ruff`, tests con `pytest`, verificación de Docker build
- **Frontend** (`.github/workflows/ci-frontend.yml`): lint, type check, build, y deploy automático a Vercel en `main`

Para el deploy a Vercel, agrega estos **secrets** en tu repo de GitHub:
- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`

## Deploy en producción

- **Frontend** → Vercel (conecta el repo, agrega `BACKEND_URL` en env vars)
- **Backend** → corre en tu GPU expuesto via Cloudflare Tunnel
  ```bash
  cloudflared tunnel --url http://localhost:8000
  ```

## Arquitectura

```
english-tutor/
├── backend/
│   ├── main.py              # FastAPI app, CORS, health, lifespan
│   ├── routers/
│   │   ├── chat.py          # LLM streaming (Qwen3.5-2B)
│   │   ├── tts.py           # Text-to-Speech (Kokoro)
│   │   └── asr.py           # Speech-to-Text (Qwen3-ASR)
│   ├── models/              # GGUFs (no en git)
│   ├── Dockerfile           # CUDA 12.1 + Python 3.11 (optimizado por uv)
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── layout.tsx       # Root layout, SEO
│   │   ├── page.tsx         # Chat principal
│   │   └── api/chat/route.ts # Proxy SSE al backend
│   ├── components/
│   │   ├── ChatBox.tsx      # Lista de mensajes
│   │   ├── MicButton.tsx    # Grabación de audio
│   │   └── AudioPlayer.tsx  # Reproducir TTS
│   ├── Dockerfile           # Node 20 Alpine
│   └── package.json
├── .github/workflows/       # CI/CD pipelines
├── docker-compose.yml       # GPU passthrough + networking
└── README.md
```
