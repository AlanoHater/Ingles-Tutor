# 🇺🇸 English Tutor — App de aprendizaje de inglés

Stack local de AI para aprender inglés con español como idioma base.

## Modelos usados

| Modelo              | Tarea                  | VRAM (Est.) |
| ------------------- | ---------------------- | ----------- |
| Qwen3.5-2B (Q4_K_M) | LLM tutor              | ~3.2GB      |
| Kokoro-82M          | Text-to-Speech inglés  | ~1.2GB      |

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

> El modelo TTS (Kokoro) se descarga automáticamente la primera vez que inicias el backend. El dictado (ASR) utiliza la API nativa del navegador para ahorrar VRAM.

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
| **KV Cache Quant**     | Cuantiza el cache de atención de f16 → q4_0                      | ~900MB VRAM liberados       |
| **Flash Attention**    | Cálculo de atención optimizado                                   | ~20% más rápido             |
| **Prompt Caching**     | Cachea el eval del system prompt (anclado via FIFO)              | Latencia <100ms tras el 1er msg |
| **8K Context**         | Doble de memoria que el estándar (4096)                         | Charlas mucho más largas    |

Resultado esperado: **30-50 tokens/segundo** en una RTX 4050 Laptop.

Todas son configurables via variables de entorno:

```bash
LLM_KV_TYPE_K=q4_0       # Tipo KV cache para Keys (default: q4_0)
LLM_KV_TYPE_V=q4_0       # Tipo KV cache para Values (default: q4_0)
LLM_FLASH_ATTN=true      # Flash Attention activado (default: true)
LLM_N_BATCH=512           # Tokens por batch (default: 512)
LLM_CONTEXT_SIZE=8192     # Tamaño de contexto (default: 8192)
LLM_GPU_LAYERS=-1         # Capas en GPU, -1 = todas (default: -1)
```

## Endpoints del backend

| Método | Ruta                   | Descripción                   |
| ------ | ---------------------- | ----------------------------- |
| GET    | `/health`              | Health check + estado de modelos |
| POST   | `/v1/chat/completions` | Chat con el tutor (streaming SSE) |
| POST   | `/tts`                 | Texto en inglés → audio WAV   |

## Desarrollo frontend sin Docker

```bash
cd frontend
npm install
npm run dev
```

## Deploy y CI/CD

> [!NOTE]
> Por ahora, el proyecto está enfocado en una **versión mínima funcional local**. 
> Las integraciones de CI/CD avanzadas y el despliegue automático (vía Vercel u otros) se implementarán en el futuro.

Para despliegues locales, se recomienda seguir usando `docker compose`.

## Arquitectura

```
english-tutor/
├── backend/
│   ├── main.py              # FastAPI app, CORS, health, lifespan
│   ├── routers/
│   │   ├── chat.py          # LLM streaming (Qwen3.5-2B) + Context Shifting
│   │   └── tts.py           # Text-to-Speech (Kokoro)
│   ├── data/                # Persistencia de sesiones (JSON)
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
