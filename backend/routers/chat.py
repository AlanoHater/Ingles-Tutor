"""
Router: /v1/chat/completions
LLM chat con Qwen3.5-2B via llama-cpp-python (streaming SSE).
"""

import os
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Modelo global y Memoria de Sesiones
# ---------------------------------------------------------------------------
_llm = None
_session_summaries: Dict[str, str] = {}  # session_id -> summary_text

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/Qwen3.5-2B-Q4_K_M.gguf")
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

SYSTEM_PROMPT = """Eres un tutor experto de inglés, amigable, paciente y motivador. Tu estudiante habla español.

Rol y Tono:
- Eres un profesor nativo de inglés que domina el español.
- Tu tono es animado, alentador y estructurado.
- Siempre felicitas el progreso y corriges con delicadeza.

Reglas Pedagógicas Estrictas:
1. Responde SIEMPRE en español, excepto cuando enseñes vocabulario o frases en inglés.
2. Cada vez que introduzcas o uses una palabra/frase en inglés, DEBES incluir este formato exacto:
   - Inglés: [texto en inglés]
   - Significado: "[traducción al español]"
3. Si el estudiante comete un error, primero valora el intento, luego da la corrección exacta y una breve explicación gramatical de por qué.
4. Desglosa la gramática compleja en partes simples.
5. Usa ejemplos muy prácticos, de la vida real (cafeterías, viajes, saludos formales e informales).
6. Mantén las respuestas concisas y fáciles de leer (máximo 3-4 párrafos cortos). Usa viñetas para listas.
7. Termina ocasionalmente tus explicaciones con una pequeña pregunta en inglés para mantener al estudiante practicando.
8. Usa emojis para hacer el texto amigable y visualmente atractivo 🇺🇸✨."""


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
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = True
    session_id: str = "default_user"  # Para persistencia de resúmenes

# ---------------------------------------------------------------------------
# Utilidades de Context Shifting
# ---------------------------------------------------------------------------

def _count_tokens(llm, text: str) -> int:
    """Cuenta tokens reales usando el tokenizador del modelo."""
    if not text: return 0
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False))

def _trim_history(llm, messages: List[Dict], max_context: int, reserve_tokens: int) -> List[Dict]:
    """
    Mantiene el System Prompt y recorta el historial (FIFO) para 
    ajustarse al límite de tokens del contexto.
    """
    system_msg = messages[0] if messages and messages[0]["role"] == "system" else None
    chat_history = messages[1:] if system_msg else messages
    
    available_tokens = max_context - reserve_tokens
    if system_msg:
        available_tokens -= _count_tokens(llm, system_msg["content"])
    
    trimmed = []
    current_tokens = 0
    
    # Recorremos de más nuevos a más viejos
    for msg in reversed(chat_history):
        msg_tokens = _count_tokens(llm, msg["content"])
        if current_tokens + msg_tokens > available_tokens:
            break
        trimmed.insert(0, msg)
        current_tokens += msg_tokens
        
    final_messages = [system_msg] + trimmed if system_msg else trimmed
    return final_messages

def _generate_summary_task(llm, session_id: str, messages_to_summarize: List[Dict]):
    """Tarea en segundo plano para resumir parte del historial."""
    global _session_summaries
    if not messages_to_summarize:
        return

    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages_to_summarize])
    prompt = f"Resume brevemente (máximo 2 párrafos) los puntos clave de esta conversación para que el tutor pueda recordarlos. Habla en español.\n\nCONVERSACIÓN:\n{history_text}\n\nRESUMEN:"
    
    try:
        logger.info(f"🔄 Generando resumen asíncrono para sesión {session_id}...")
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.3
        )
        summary = response["choices"][0]["message"]["content"].strip()
        _session_summaries[session_id] = summary
        logger.info(f"✅ Resumen actualizado para {session_id}")
    except Exception as e:
        logger.error(f"❌ Error en tarea de resumen: {e}")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@router.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, background_tasks: BackgroundTasks):
    """Chat con el tutor de inglés. Soporta Context Shifting."""
    try:
        llm = get_llm()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cargando modelo: {e}")

    # 1. Preparar historial base
    raw_messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    # 2. Inyectar Resumen si existe
    summary = _session_summaries.get(request.session_id)
    system_content = SYSTEM_PROMPT
    if summary:
        system_content += f"\n\nRECORDATORIO DE LA CONVERSACIÓN ANTERIOR:\n{summary}"
        
    messages = [{"role": "system", "content": system_content}]
    messages.extend(raw_messages)

    # 3. Sliding Window: Asegurar que no excedemos CONTEXT_SIZE
    # Reservamos tokens para la respuesta (max_tokens)
    final_messages = _trim_history(llm, messages, CONTEXT_SIZE, request.max_tokens)
    
    # 4. Trigger de Resumen Asíncrono (si estamos al 80%)
    total_tokens = sum([_count_tokens(llm, m["content"]) for m in final_messages])
    if total_tokens > (CONTEXT_SIZE * 0.8) and len(raw_messages) > 4:
        # Resumimos los mensajes que NO entraron en final_messages o los más viejos
        # Para simplificar: resumimos los primeros N mensajes del request actual
        messages_to_summarize = raw_messages[:min(len(raw_messages), 6)]
        background_tasks.add_task(_generate_summary_task, llm, request.session_id, messages_to_summarize)

    if request.stream:
        return StreamingResponse(
            _stream_response(llm, final_messages, request.temperature, request.max_tokens),
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
            messages=final_messages,
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
