"""
Korean Tutor — Backend API
FastAPI app con modelos de AI para aprender coreano.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import chat, tts, asr

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan — carga / descarga de modelos
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa modelos al arrancar y los libera al apagar."""
    logger.info("🚀 Arrancando Korean Tutor backend...")

    # Los modelos se cargan lazy (al primer request) para agilizar el arranque.
    # Si quieres precargarlos, descomenta las líneas de abajo:
    # chat.get_llm()
    # tts.get_tts_pipeline()
    # asr.get_asr_pipeline()

    yield

    logger.info("🛑 Apagando Korean Tutor backend...")
    # Limpieza de modelos si es necesario
    chat.cleanup()
    tts.cleanup()
    asr.cleanup()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Korean Tutor API",
    description="API para el tutor de coreano con IA local",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — permite frontend en dev y en producción (Vercel)
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# En producción, añade tu dominio de Vercel
VERCEL_URL = os.getenv("VERCEL_URL")
if VERCEL_URL:
    ALLOWED_ORIGINS.append(f"https://{VERCEL_URL}")

FRONTEND_URL = os.getenv("FRONTEND_URL")
if FRONTEND_URL:
    ALLOWED_ORIGINS.append(FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(chat.router)
app.include_router(tts.router)
app.include_router(asr.router)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health check endpoint."""
    gpu_info = "N/A"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return {
        "status": "ok",
        "gpu": gpu_info,
        "models": {
            "llm": chat.is_loaded(),
            "tts": tts.is_loaded(),
            "asr": asr.is_loaded(),
        },
    }
