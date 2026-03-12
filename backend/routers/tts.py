"""
Router: /tts
Text-to-Speech usando exclusivamente Kokoro para texto en inglés.
Corre localmente en la GPU usando una voz americana por defecto.
"""

import io
import logging
import torch
import soundfile as sf
import numpy as np

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

# Variables globales
kokoro_pipeline = None

def is_loaded() -> bool:
    return kokoro_pipeline is not None

def load_models():
    """Carga Kokoro en RAM/VRAM en el primer uso (lazy loading)."""
    global kokoro_pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if kokoro_pipeline is None:
        logger.info(f"Cargando pipeline Kokoro (Inglés Americano) en {device}...")
        from kokoro import KPipeline
        # lang_code='a' es Inglés Americano en Kokoro
        kokoro_pipeline = KPipeline(lang_code='a', device=device)
        logger.info("✅ Kokoro cargado exitosamente.")

def cleanup():
    """Libera la memoria de Kokoro."""
    global kokoro_pipeline
    kokoro_pipeline = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("🧹 TTS liberado de memoria")

class TTSRequest(BaseModel):
    text: str
    voice: str | None = "af_heart"  # Voz femenina en inglés por defecto

@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convierte texto a audio WAV usando Kokoro (Inglés).
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="El campo 'text' no puede estar vacío")

    try:
        load_models()
        buffer = io.BytesIO()
        
        # Generar audio con la voz seleccionada (ej. af_heart, am_adam)
        generator = kokoro_pipeline(request.text, voice=request.voice, speed=1)
        
        audio_chunks = []
        sample_rate = 24000
        
        for graphemes, phonemes, audio in generator:
            audio_chunks.append(audio)
            
        if not audio_chunks:
            raise ValueError("Kokoro no produjo audio.")
            
        # Concatenar si hay múltiples fragmentos generados
        full_audio = np.concatenate(audio_chunks)
        
        sf.write(buffer, full_audio, sample_rate, format='WAV')
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=tts_output.wav"},
        )
            
    except Exception as e:
        logger.error(f"❌ Error generando TTS con Kokoro: {e}")
        raise HTTPException(status_code=500, detail=f"Error generando audio: {e}")
