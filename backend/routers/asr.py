"""
Router: /asr
Automatic Speech Recognition con Qwen3-ASR para transcribir audio coreano.
"""

import io
import os
import logging
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Modelo global (lazy-loaded)
# ---------------------------------------------------------------------------
_asr_model = None
_asr_processor = None

ASR_MODEL_NAME = os.getenv("ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")


def get_asr_pipeline():
    """Carga Qwen3-ASR de forma lazy."""
    global _asr_model, _asr_processor
    if _asr_model is None:
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if device == "cuda" else torch.float32

            logger.info(f"🎙️ Cargando ASR ({ASR_MODEL_NAME}) en {device}...")

            _asr_processor = AutoProcessor.from_pretrained(
                ASR_MODEL_NAME,
                trust_remote_code=True,
            )
            _asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                ASR_MODEL_NAME,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            ).to(device)

            logger.info("✅ ASR cargado correctamente")
        except Exception as e:
            logger.error(f"❌ Error cargando ASR: {e}")
            raise
    return _asr_model, _asr_processor


def is_loaded() -> bool:
    return _asr_model is not None


def cleanup():
    global _asr_model, _asr_processor
    if _asr_model is not None:
        del _asr_model
        del _asr_processor
        _asr_model = None
        _asr_processor = None
        logger.info("🧹 ASR liberado de memoria")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ASRResponse(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@router.post("/asr", response_model=ASRResponse)
async def speech_to_text(audio: UploadFile = File(...)):
    """Transcribe audio en coreano a texto."""

    # Validar tipo de archivo
    allowed_types = [
        "audio/webm", "audio/wav", "audio/wave", "audio/x-wav",
        "audio/mp3", "audio/mpeg", "audio/ogg", "audio/flac",
        "application/octet-stream",  # browsers a veces mandan esto
    ]
    if audio.content_type and audio.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Formato de audio no soportado: {audio.content_type}. "
                   f"Usa: webm, wav, mp3, ogg, flac",
        )

    try:
        model, processor = get_asr_pipeline()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ASR no disponible: {e}")

    try:
        import torch
        import torchaudio

        # Leer audio del upload
        audio_bytes = await audio.read()

        # Guardar temporalmente para que torchaudio lo pueda leer
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            # Cargar y resamplear a 16kHz
            waveform, sample_rate = torchaudio.load(tmp_path)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # Convertir a mono si es estéreo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform = waveform.squeeze()

            # Procesar con el modelo
            device = next(model.parameters()).device
            inputs = processor(
                waveform.numpy(),
                sampling_rate=16000,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                predicted_ids = model.generate(**inputs)

            transcription = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            return ASRResponse(text=transcription.strip())

        finally:
            os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en transcripción: {e}")
        raise HTTPException(status_code=500, detail=f"Error transcribiendo audio: {e}")
