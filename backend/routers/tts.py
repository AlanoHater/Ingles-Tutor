"""
Router: /tts
Text-to-Speech usando:
- facebook/mms-tts-kor para texto en coreano.
- Kokoro (lang_code='e') para texto en español.
Ambos corren localmente en la GPU.
"""

import io
import re
import logging
import torch
import soundfile as sf
import warnings

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

# Variables globales para los modelos
mms_tokenizer = None
mms_model = None
kokoro_pipeline = None

def is_korean(text: str) -> bool:
    """Verifica si el texto contiene caracteres Hangul (coreano)."""
    return bool(re.search(r'[\u3131-\uD79D]', text))

def is_loaded() -> bool:
    return mms_model is not None and kokoro_pipeline is not None

def load_models():
    """Carga los modelos TTS en RAM/VRAM en el primer uso (lazy loading)."""
    global mms_tokenizer, mms_model, kokoro_pipeline
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if mms_model is None:
        logger.info("Cargando modelo MMS-TTS-KOR...")
        from transformers import VitsModel, AutoTokenizer
        # Ignore warning for weights
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.utils.weight_norm")
        mms_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")
        mms_model = VitsModel.from_pretrained("facebook/mms-tts-kor").to(device)
        logger.info("MMS-TTS-KOR cargado exitosamente.")
        
    if kokoro_pipeline is None:
        logger.info("Cargando pipeline Kokoro (Español)...")
        from kokoro import KPipeline
        # lang_code='e' es Español en Kokoro
        kokoro_pipeline = KPipeline(lang_code='e', device=device)
        logger.info("Kokoro cargado exitosamente.")

def cleanup():
    """Libera la memoria de los modelos."""
    global mms_tokenizer, mms_model, kokoro_pipeline
    mms_tokenizer = None
    mms_model = None
    kokoro_pipeline = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class TTSRequest(BaseModel):
    text: str
    voice: str | None = None

@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convierte texto a audio WAV. 
    Usa MMS para coreano y Kokoro para español.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="El campo 'text' no puede estar vacío")

    try:
        load_models()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        buffer = io.BytesIO()
        
        if is_korean(request.text):
            # Usar facebook/mms-tts-kor
            inputs = mms_tokenizer(request.text, return_tensors="pt").to(device)
            with torch.no_grad():
                output = mms_model(**inputs).waveform
            
            audio_data = output.cpu().numpy().squeeze()
            sample_rate = mms_model.config.sampling_rate
            
            sf.write(buffer, audio_data, sample_rate, format='WAV')
            buffer.seek(0)
            
            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={"Content-Disposition": "inline; filename=tts_output.wav"},
            )
        else:
            # Usar Kokoro en español
            # Voz por defecto para español en Kokoro (e.g. 'e_male_1' o 'e_female_1' si existen)
            # 'p' can use pre-packaged voices if specified, or default
            generator = kokoro_pipeline(request.text, voice='e_f_isabella', speed=1)
            
            # KPipeline devuelve un generador que produce (graphemes, phonemes, audio)
            audio_chunks = []
            sample_rate = 24000
            
            for graphemes, phonemes, audio in generator:
                audio_chunks.append(audio)
                
            if not audio_chunks:
                raise ValueError("Kokoro no produjo audio.")
                
            # Concatenar si hay múltiples chunks
            import numpy as np
            full_audio = np.concatenate(audio_chunks)
            
            sf.write(buffer, full_audio, sample_rate, format='WAV')
            buffer.seek(0)

            return StreamingResponse(
                buffer,
                media_type="audio/wav",
                headers={"Content-Disposition": "inline; filename=tts_output.wav"},
            )
            
    except Exception as e:
        logger.error(f"Error generando TTS: {e}")
        raise HTTPException(status_code=500, detail=f"Error generando audio: {e}")
