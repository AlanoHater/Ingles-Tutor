"""
Router: /tts
Text-to-Speech con Kokoro para audio en coreano.
"""

import io
import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Modelo global (lazy-loaded)
# ---------------------------------------------------------------------------
_tts_pipeline = None

TTS_VOICE = os.getenv("TTS_VOICE", "kf_bella")


def get_tts_pipeline():
    """Carga Kokoro TTS de forma lazy."""
    global _tts_pipeline
    if _tts_pipeline is None:
        try:
            from kokoro import KPipeline

            logger.info(f"🔊 Cargando Kokoro TTS (voz: {TTS_VOICE})...")
            _tts_pipeline = KPipeline(lang_code="k")
            logger.info("✅ TTS cargado correctamente")
        except Exception as e:
            logger.error(f"❌ Error cargando TTS: {e}")
            raise
    return _tts_pipeline


def is_loaded() -> bool:
    return _tts_pipeline is not None


def cleanup():
    global _tts_pipeline
    if _tts_pipeline is not None:
        del _tts_pipeline
        _tts_pipeline = None
        logger.info("🧹 TTS liberado de memoria")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TTSRequest(BaseModel):
    text: str
    voice: str | None = None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Convierte texto coreano a audio WAV."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="El campo 'text' no puede estar vacío")

    try:
        pipeline = get_tts_pipeline()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"TTS no disponible: {e}")

    voice = request.voice or TTS_VOICE

    try:
        import soundfile as sf
        import numpy as np

        # Kokoro genera audio en segmentos
        audio_segments = []
        for _, _, audio in pipeline(request.text, voice=voice):
            if audio is not None:
                audio_segments.append(audio)

        if not audio_segments:
            raise HTTPException(status_code=500, detail="No se generó audio")

        # Concatenar segmentos
        full_audio = np.concatenate(audio_segments)

        # Escribir a buffer WAV
        buffer = io.BytesIO()
        sf.write(buffer, full_audio, samplerate=24000, format="WAV")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=tts_output.wav",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generando TTS: {e}")
        raise HTTPException(status_code=500, detail=f"Error generando audio: {e}")
