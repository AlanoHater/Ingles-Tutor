"""
Router: /v1/chat/completions
LLM chat con Qwen3.5-2B via llama-cpp-python (streaming SSE).
"""

import os
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Modelo global (lazy-loaded)
# ---------------------------------------------------------------------------
_llm = None

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/qwen3.5-2b-q4_k_m.gguf")
CONTEXT_SIZE = int(os.getenv("LLM_CONTEXT_SIZE", "4096"))
GPU_LAYERS = int(os.getenv("LLM_GPU_LAYERS", "-1"))  # -1 = todas las capas en GPU

# --- Optimizaciones GPU ---
# KV Cache Quantization: q8_0 ahorra ~400MB VRAM sin pérdida perceptible
KV_CACHE_TYPE_K = os.getenv("LLM_KV_TYPE_K", "q8_0")
KV_CACHE_TYPE_V = os.getenv("LLM_KV_TYPE_V", "q8_0")
# Flash Attention: ~20% más rápido, menos memoria
FLASH_ATTN = os.getenv("LLM_FLASH_ATTN", "true").lower() == "true"
# Batch size: tokens procesados por iteración (512 es buen balance)
N_BATCH = int(os.getenv("LLM_N_BATCH", "512"))

SYSTEM_PROMPT = """Eres un tutor de coreano amigable y paciente. Tu estudiante habla español.

Reglas:
1. Responde siempre en español, excepto cuando muestres palabras o frases en coreano.
2. Cuando escribas en coreano, siempre incluye: hangul, romanización y traducción al español.
3. Usa ejemplos prácticos y cotidianos.
4. Si el estudiante comete errores, corrígelos amablemente y explica por qué.
5. Adapta tu nivel al del estudiante.
6. Sé conciso — respuestas de máximo 3-4 párrafos.
7. Usa emojis ocasionalmente para hacer la conversación más amigable 🇰🇷"""


def get_llm():
    """Carga el modelo LLM de forma lazy."""
    global _llm
    if _llm is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Modelo GGUF no encontrado en {MODEL_PATH}. "
                f"Descárgalo con: huggingface-cli download bartowski/Qwen3.5-2B-GGUF "
                f"Qwen3.5-2B-Q4_K_M.gguf --local-dir backend/models"
            )
        try:
            from llama_cpp import Llama

            logger.info(f"📚 Cargando LLM desde {MODEL_PATH}...")
            logger.info(f"   ⚡ Flash Attention: {FLASH_ATTN}")
            logger.info(f"   🗜️  KV Cache: K={KV_CACHE_TYPE_K}, V={KV_CACHE_TYPE_V}")
            logger.info(f"   📦 Batch size: {N_BATCH}")

            _llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=CONTEXT_SIZE,
                n_gpu_layers=GPU_LAYERS,
                # --- Optimización 1: KV Cache Quantization ---
                # Reduce VRAM del cache de atención de f16 a q8_0
                # Ahorra ~400MB sin pérdida perceptible de calidad
                type_k=KV_CACHE_TYPE_K,
                type_v=KV_CACHE_TYPE_V,
                # --- Optimización 2: Flash Attention ---
                # Cálculo de atención optimizado, ~20% más rápido
                flash_attn=FLASH_ATTN,
                # --- Optimización 4: Continuous Batching ---
                # Más tokens por iteración = mejor throughput
                n_batch=N_BATCH,
                verbose=False,
            )
            # --- Optimización 3: Prompt Caching ---
            # El modelo se carga UNA vez y se reutiliza entre requests.
            # llama-cpp-python cachea internamente el eval del system prompt:
            # la primera vez procesa los ~200 tokens del prompt, y en requests
            # subsecuentes reutiliza ese estado — reduciendo latencia brutalmente.
            logger.info("✅ LLM cargado correctamente — optimizado para RTX 4050")
        except Exception as e:
            logger.error(f"❌ Error cargando LLM: {e}")
            raise
    return _llm


def is_loaded() -> bool:
    """Retorna True si el modelo está cargado."""
    return _llm is not None


def cleanup():
    """Libera el modelo de memoria."""
    global _llm
    if _llm is not None:
        del _llm
        _llm = None
        logger.info("🧹 LLM liberado de memoria")


# ---------------------------------------------------------------------------
# Schemas (compatible con formato OpenAI)
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = True


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@router.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat con el tutor coreano. Soporta streaming SSE."""
    try:
        llm = get_llm()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando modelo: {e}")

    # Prepend system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend([{"role": m.role, "content": m.content} for m in request.messages])

    if request.stream:
        return StreamingResponse(
            _stream_response(llm, messages, request.temperature, request.max_tokens),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming response
        response = llm.create_chat_completion(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False,
        )
        return response


async def _stream_response(llm, messages: list, temperature: float, max_tokens: int):
    """Genera chunks SSE compatibles con el formato de OpenAI."""
    try:
        stream = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            chunk_data = {
                "choices": [
                    {
                        "delta": chunk["choices"][0].get("delta", {}),
                        "index": 0,
                        "finish_reason": chunk["choices"][0].get("finish_reason"),
                    }
                ]
            }
            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Error en streaming: {e}")
        error_data = {"error": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"
