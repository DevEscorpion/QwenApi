#!/usr/bin/env python3
"""
Servidor API compatible con OpenAI para Qwen con soporte para pensamiento.
OPTIMIZADO PARA BAJA LATENCIA Y ALTA CONCURRENCIA.

- I/O totalmente asíncrono para no bloquear el servidor.
- Pool de sesiones automático para reducir latencia.
- Sistema avanzado de gestión de sesiones.
- Soporte para cookies en base64 y token de autenticación.

Levanta un servidor local en http://localhost:5001 que traduce las peticiones
de la API de OpenAI al protocolo de Qwen.

Modelos disponibles:
- qwen-standard: Modelo estándar sin razonamiento (solo respuesta final)
- qwen-standard-thinking: Modelo estándar con proceso de razonamiento completo
- qwen-coder: Modelo especializado en código (solo respuesta final)
- qwen-coder-flash: Modelo especializado en código rápido (solo respuesta final)
- qwen-max: Modelo premium máximo (solo respuesta final)
- qwen-max-thinking: Modelo premium máximo con proceso de razonamiento

Requisitos:
- pip install fastapi uvicorn httpx orjson python-dotenv uvloop httptools

Ejecución:
1. Configura las variables QWEN_AUTH_TOKEN or QWEN_COOKIES_JSON_B64 en el archivo .env
2. Guarda el archivo como 'main_proxy.py'
3. Ejecuta desde la terminal: uvicorn main_proxy:app --host 0.0.0.0 --port 5001 --loop uvloop --http httptools

Configuración en clientes (Genie, CodeGPT, etc.):
- API Endpoint / Base URL: http://localhost:5001/v1
- API Key: Cualquier cosa (ej: "sk-12345")
- Models: qwen-standard, qwen-standard-thinking, qwen-coder, qwen-coder-flash, qwen-max, qwen-max-thinking
"""
import json
import logging
import time
import uuid
import asyncio
import hashlib
import re
import os
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Intenta usar orjson para un parseo JSON más rápido ---
try:
    import orjson
    def orjson_dumps(v, *, default=None):
        return orjson.dumps(v, default=default).decode()
    JSON_SERIALIZER = orjson_dumps
    JSON_DESERIALIZER = orjson.loads
    print("Usando 'orjson' para un rendimiento JSON mejorado.")
except ImportError:
    JSON_SERIALIZER = json.dumps
    JSON_DESERIALIZER = json.loads
    print("Usando 'json' estándar. Instala 'orjson' para mejorar el rendimiento.")

# --- CONFIGURACIÓN DE CREDENCIALES ---
load_dotenv()
QWEN_AUTH_TOKEN_FALLBACK = os.getenv("QWEN_AUTH_TOKEN", "")
QWEN_COOKIES_JSON_B64 = os.getenv("QWEN_COOKIES_JSON_B64", "")

# ---------- COOKIES / TOKEN ----------
QWEN_AUTH_TOKEN: str = ""
QWEN_COOKIE_STRING: str = ""

def process_cookies_and_extract_token(b64_string: str | None):
    global QWEN_AUTH_TOKEN, QWEN_COOKIE_STRING
    if not b64_string:
        print("[WARN] Cookies var not found → using fallback token")
        QWEN_AUTH_TOKEN = QWEN_AUTH_TOKEN_FALLBACK
        QWEN_COOKIE_STRING = ""
        return
    try:
        cookies_list = JSON_DESERIALIZER(base64.b64decode(b64_string))
        QWEN_COOKIE_STRING = "; ".join(f"{c['name']}={c['value']}" for c in cookies_list)
        token_value = next((c.get("value", "") for c in cookies_list if c.get("name") == "token"), "")
        QWEN_AUTH_TOKEN = f"Bearer {token_value}" if token_value else QWEN_AUTH_TOKEN_FALLBACK
        print("✅ Cookies & token processed OK")
    except Exception as e:
        print(f"[ERROR] Cookie parse: {e}")
        QWEN_AUTH_TOKEN = QWEN_AUTH_TOKEN_FALLBACK
        QWEN_COOKIE_STRING = ""

process_cookies_and_extract_token(QWEN_COOKIES_JSON_B64)

# --- CONSTANTES DE LA API QWEN ---
QWEN_API_BASE_URL = "https://chat.qwen.ai/api/v2"

# Modelos disponibles con sus configuraciones internas
MODEL_CONFIGS = {
    "qwen-standard": {"internal_model_id": "qwen3-235b-a22b", "filter_phase": True, "display_name": "Qwen3 Standard"},
    "qwen-standard-thinking": {"internal_model_id": "qwen3-235b-a22b", "filter_phase": False, "display_name": "Qwen3 Standard (con pensamiento)"},
    "qwen-coder": {"internal_model_id": "qwen3-coder-plus", "filter_phase": True, "display_name": "Qwen3 Coder"},
    "qwen-coder-flash": {"internal_model_id": "qwen3-coder-30b-a3b-instruct", "filter_phase": True, "display_name": "Qwen3 Coder Flash"},
    "qwen-max": {"internal_model_id": "qwen3-max-preview", "filter_phase": True, "display_name": "Qwen3 Max"},
    "qwen-max-thinking": {"internal_model_id": "qwen3-max-preview", "filter_phase": False, "display_name": "Qwen3 Max (con pensamiento)"},
}

# Alias para compatibilidad
MODEL_QWEN_STANDARD = "qwen-standard"
MODEL_QWEN_STANDARD_THINKING = "qwen-standard-thinking"
MODEL_QWEN_CODER = "qwen-coder"
MODEL_QWEN_CODER_FLASH = "qwen-coder-flash"
MODEL_QWEN_MAX = "qwen-max"
MODEL_QWEN_MAX_THINKING = "qwen-max-thinking"

QWEN_HEADERS = {
    "Accept": "application/json", "Accept-Language": "es-AR,es;q=0.7", "Authorization": QWEN_AUTH_TOKEN,
    "bx-v": "2.5.31", "Content-Type": "application/json; charset=UTF-8", 
    "Origin": "https://chat.qwen.ai", "Referer": "https://chat.qwen.ai/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
    "source": "web", "x-accel-buffering": "no",
}

# Añadir cookie string si está disponible
if QWEN_COOKIE_STRING:
    QWEN_HEADERS["Cookie"] = QWEN_COOKIE_STRING

MIN_CHAT_ID_POOL_SIZE = 2
MAX_CHAT_ID_POOL_SIZE = 5

# --- INICIALIZACIÓN DE LA APP Y LOGGING ---
app = FastAPI(title="Qwen Web API Proxy (Ultra Low Latency)", version="2.5.0")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QwenProxy")

# --- MODELOS DE DATOS (Pydantic) ---
class OpenAIMessage(BaseModel):
    role: str; content: str
class OpenAIChatCompletionRequest(BaseModel):
    model: str; messages: List[OpenAIMessage]; stream: bool = False
class OpenAIChunkDelta(BaseModel):
    content: str | None = None
class OpenAIChunkChoice(BaseModel):
    index: int = 0; delta: OpenAIChunkDelta; finish_reason: str | None = None
class OpenAICompletionChunk(BaseModel):
    id: str; object: str = "chat.completion.chunk"; created: int; model: str; choices: List[OpenAIChunkChoice]

# --- SISTEMA AVANZADO DE GESTIÓN DE SESIONES CON POOL ---
class AdvancedSessionManager:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._session_pool: Dict[str, Dict[str, Any]] = {}
        self._client_sessions: Dict[str, Dict[str, str]] = {}
        self._available_sessions: Dict[str, asyncio.Queue] = {
            model: asyncio.Queue(maxsize=MAX_CHAT_ID_POOL_SIZE) 
            for model in MODEL_CONFIGS.keys()
        }
        self._stats = {"sessions_created": 0, "sessions_reused": 0, "clients_served": 0, "pool_hits": 0, "pool_misses": 0}

    def _generate_unique_client_id(self, request_info: dict) -> str:
        unique_string = f"{time.time()}_{uuid.uuid4().hex[:8]}_{request_info.get('user_agent', 'unknown')}_{request_info.get('model', 'default')}"
        return f"client_{hashlib.md5(unique_string.encode()).hexdigest()[:12]}"

    def _detect_returning_client(self, request_info: dict) -> Optional[str]:
        headers = request_info.get("headers", {})
        if "x-client-session" in headers:
            return headers["x-client-session"]
        if "authorization" in headers:
            auth_hash = hashlib.md5(headers["authorization"].encode()).hexdigest()[:8]
            return f"auth_{auth_hash}"
        return None

    def _create_session_info(self, session_id: str, model: str, client_id: str) -> dict:
        return {"id": session_id, "model": model, "client_id": client_id, "created": datetime.now(), "last_used": datetime.now(), "expiry": datetime.now() + timedelta(hours=1), "usage_count": 0, "status": "active"}

    async def get_or_create_session(self, request_info: dict, client: httpx.AsyncClient) -> Tuple[str, Optional[str]]:
        async with self._lock:
            model = request_info.get("model", MODEL_QWEN_STANDARD)
            client_id = self._detect_returning_client(request_info)
            if not client_id:
                client_id = self._generate_unique_client_id(request_info)
                self._stats["clients_served"] += 1
            
            client_model_sessions = self._client_sessions.get(client_id, {})
            existing_session_id = client_model_sessions.get(model)

            if existing_session_id:
                if (existing_session_id in self._session_pool and datetime.now() < self._session_pool[existing_session_id]["expiry"]):
                    session_info = self._session_pool[existing_session_id]
                    session_info["last_used"] = datetime.now()
                    session_info["usage_count"] += 1
                    self._stats["sessions_reused"] += 1
                    return client_id, existing_session_id
                else:
                    if model in client_model_sessions:
                        del client_model_sessions[model]
                    if existing_session_id in self._session_pool:
                        del self._session_pool[existing_session_id]
            
            # Intentar obtener del pool
            try:
                session_id = self._available_sessions[model].get_nowait()
                if session_id in self._session_pool and datetime.now() < self._session_pool[session_id]["expiry"]:
                    self._stats["pool_hits"] += 1
                    
                    self._client_sessions.setdefault(client_id, {})[model] = session_id
                    session_info = self._session_pool[session_id]
                    session_info["client_id":] = client_id
                    session_info["last_used"] = datetime.now()
                    session_info["usage_count"] += 1
                    
                    return client_id, session_id
            except asyncio.QueueEmpty:
                pass
            
            self._stats["pool_misses"] += 1
            return client_id, None

    async def register_new_session(self, client_id: str, session_id: str, model: str):
        async with self._lock:
            session_info = self._create_session_info(session_id, model, client_id)
            self._session_pool[session_id] = session_info
            self._client_sessions.setdefault(client_id, {})[model] = session_id
            self._stats["sessions_created"] += 1

    async def return_session_to_pool(self, session_id: str, model: str):
        """Devuelve una sesión al pool para reutilización"""
        try:
            if session_id in self._session_pool:
                self._available_sessions[model].put_nowait(session_id)
        except asyncio.QueueFull:
            pass  # Silenciosamente descartar si el pool está lleno

    async def cleanup_expired_sessions(self):
        async with self._lock:
            now = datetime.now()
            expired_ids = {sid for sid, info in self._session_pool.items() if now >= info["expiry"]}
            if not expired_ids: return

            self._session_pool = {sid: info for sid, info in self._session_pool.items() if sid not in expired_ids}
            
            clients_to_clean = []
            for cid, model_sessions in self._client_sessions.items():
                sessions_to_remove = {model for model, sid in model_sessions.items() if sid in expired_ids}
                for model in sessions_to_remove:
                    del model_sessions[model]
                if not model_sessions:
                    clients_to_clean.append(cid)
            
            for cid in clients_to_clean:
                del self._client_sessions[cid]
            
            # Limpiar pools
            for model, queue in self._available_sessions.items():
                new_queue = asyncio.Queue(maxsize=MAX_CHAT_ID_POOL_SIZE)
                while not queue.empty():
                    try:
                        session_id = queue.get_nowait()
                        if session_id not in expired_ids:
                            new_queue.put_nowait(session_id)
                    except asyncio.QueueEmpty:
                        break
                self._available_sessions[model] = new_queue
    
    async def get_detailed_stats(self) -> dict:
        async with self._lock:
            active_client_sessions = sum(len(sessions) for sessions in self._client_sessions.values())
            
            pool_sizes = {}
            for model, queue in self._available_sessions.items():
                pool_sizes[model] = queue.qsize()

            return {
                "pool_stats": {
                    "total_sessions": len(self._session_pool),
                    "available_sessions": pool_sizes,
                    "active_clients": len(self._client_sessions),
                    "active_client_sessions": active_client_sessions
                },
                "usage_stats": self._stats
            }

# --- LÓGICA DE COMUNICACIÓN CON QWEN ---
async def create_qwen_chat(client: httpx.AsyncClient, model: str) -> str | None:
    """Crea una nueva sesión de chat en Qwen y devuelve su ID. Usado por el pool."""
    url = f"{QWEN_API_BASE_URL}/chats/new"
    internal_model_id = MODEL_CONFIGS[model]["internal_model_id"]
    payload = {"title": "Proxy Pool Chat", "models": [internal_model_id], "chat_mode": "normal", "chat_type": "t2t", "timestamp": int(time.time() * 1000)}
    try:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        
        response_text = response.text
        data = JSON_DESERIALIZER(response_text)

        if data.get("success") and (chat_id := data.get("data", {}).get("id")):
            return chat_id
        
        return None
    except Exception:
        return None

def _build_qwen_completion_payload(chat_id: str, message: OpenAIMessage, model: str) -> Dict[str, Any]:
    current_timestamp = int(time.time())
    internal_model_id = MODEL_CONFIGS[model]["internal_model_id"]
    
    return {
        "stream": True, "incremental_output": True, "chat_id": chat_id, "chat_mode": "normal",
        "model": internal_model_id, "parent_id": None,
        "messages": [{
            "fid": str(uuid.uuid4()), "parentId": None, "role": message.role, "content": message.content,
            "user_action": "chat", "files": [], "timestamp": current_timestamp, "models": [internal_model_id],
            "chat_type": "t2t", "feature_config": {"thinking_enabled": True, "output_schema": "phase", "thinking_budget": 81920},
            "extra": {"meta": {"subChatType": "t2t"}}, "sub_chat_type": "t2t",
        }],
        "timestamp": current_timestamp,
    }

def _format_sse_chunk(data: BaseModel) -> str:
    json_str = JSON_SERIALIZER(data.model_dump(exclude_unset=True), default=str)
    return f"data: {json_str}\n\n"

# Pre-compilar regex para parsing más rápido
DATA_PREFIX = re.compile(r'^data:\s*')
DONE_MARKER = re.compile(r'^data:\s*\[DONE\]')

# --- GESTIÓN DEL CICLO DE VIDA DE LA APLICACIÓN ---
app_state: Dict[str, Any] = {}
session_manager = AdvancedSessionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación"""
    # Configuración optimizada para baja latencia
    limits = httpx.Limits(
        max_connections=200,
        max_keepalive_connections=50,
        keepalive_expiry=5.0
    )
    
    timeout = httpx.Timeout(
        connect=3.0,
        read=10.0,
        write=10.0,
        pool=3.0
    )
    
    client = httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        http2=True,
        headers=QWEN_HEADERS
    )
    
    app_state.update({"http_client": client})
    
    # Iniciar gestor del pool optimizado
    pool_task = asyncio.create_task(low_latency_pool_manager(client, session_manager))
    app_state["pool_task"] = pool_task
    
    yield
    
    # Apagado
    pool_task.cancel()
    try:
        await pool_task
    except asyncio.CancelledError: 
        pass
    await client.aclose()

app.router.lifespan_context = lifespan

def get_http_client() -> httpx.AsyncClient: 
    return app_state["http_client"]

async def low_latency_pool_manager(client: httpx.AsyncClient, session_manager: AdvancedSessionManager):
    """Gestor optimizado para baja latencia"""
    # Pre-calentar conexiones HTTP
    warmup_tasks = []
    for _ in range(3):
        warmup_tasks.append(client.get(f"{QWEN_API_BASE_URL}/chats/new"))
    
    try:
        await asyncio.gather(*warmup_tasks, return_exceptions=True)
    except Exception:
        pass  # Ignorar errores de warm-up
    
    while True:
        try:
            for model in MODEL_CONFIGS.keys():
                queue = session_manager._available_sessions[model]
                current_size = queue.qsize()
                
                # Mantener el pool siempre lleno para respuesta inmediata
                if current_size < MAX_CHAT_ID_POOL_SIZE:
                    num_to_create = MAX_CHAT_ID_POOL_SIZE - current_size
                    created = 0
                    
                    # Crear sesiones en paralelo para mayor velocidad
                    create_tasks = [create_qwen_chat(client, model) for _ in range(num_to_create)]
                    results = await asyncio.gather(*create_tasks, return_exceptions=True)
                    
                    for chat_id in results:
                        if isinstance(chat_id, str) and chat_id:
                            await session_manager.register_new_session(f"pool_{model}", chat_id, model)
                            await session_manager.return_session_to_pool(chat_id, model)
                            created += 1
            
            # Intervalo más corto para respuesta más rápida
            await asyncio.sleep(2)
            
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(5)

# --- MIDDLEWARE DE LATENCIA ---
@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    
    # Solo registrar latencias altas para no sobrecargar
    if process_time > 0.5:  # Más de 500ms
        logger.warning(f"LATENCY: {request.method} {request.url.path} took {process_time:.3f}s")
    
    # Header de latencia para debugging
    response.headers["X-Process-Time"] = f"{process_time:.3f}"
    return response

# --- ENDPOINTS DE LA API ---
@app.get("/")
async def root():
    stats = await session_manager.get_detailed_stats()
    return JSONResponse(content={
        "service": "Qwen Proxy Server - Ultra Low Latency",
        "version": "2.5.0", "status": "running",
        "features": ["Async I/O", "Session Pooling", "Optimized Streaming", "Low Latency", "Cookie Auth Support", "Multi-Model Support"],
        "endpoints": {"models": "/v1/models", "chat": "/v1/chat/completions", "stats": "/stats"},
        "session_stats": stats,
        "auth_method": "cookies" if QWEN_COOKIE_STRING else "token",
        "available_models": list(MODEL_CONFIGS.keys())
    })

@app.get("/stats")
async def get_stats():
    return JSONResponse(content=await session_manager.get_detailed_stats())

@app.get("/v1/models")
def list_models():
    model_data = [
        {
            "id": model_id, 
            "object": "model", 
            "created": int(time.time()), 
            "owned_by": "qwen", 
            "description": f"{config['display_name']} - {'Solo respuesta final' if config.get('filter_phase', True) else 'Con proceso de razonamiento'}"
        }
        for model_id, config in MODEL_CONFIGS.items()
    ]
    return JSONResponse(content={"object": "list", "data": model_data})

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    retry_count = 0
    max_retries = 2
    session_id = None
    model = MODEL_QWEN_STANDARD
    client_id = None
    
    while retry_count <= max_retries:
        try:
            # Limpieza periódica
            if time.time() % 300 < 1:
                await session_manager.cleanup_expired_sessions()

            # Leer y parsear el cuerpo de forma optimizada
            body = await request.body()
            req_data = JSON_DESERIALIZER(body)
            
            model = req_data.get("model", MODEL_QWEN_STANDARD)
            
            # Validar que el modelo sea soportado
            if model not in MODEL_CONFIGS:
                raise HTTPException(status_code=400, detail=f"Modelo '{model}' no soportado. Modelos disponibles: {list(MODEL_CONFIGS.keys())}")
            
            request_info = {"headers": dict(request.headers), "model": model, "user_agent": request.headers.get("user-agent", "unknown")}
            
            client = get_http_client()
            client_id, session_id = await session_manager.get_or_create_session(request_info, client)
            
            if session_id is None:
                session_id = await create_qwen_chat(client, model)
                if not session_id:
                    if retry_count < max_retries:
                        retry_count += 1
                        logger.warning("⚠️ Fallo al crear sesión Qwen, reintentando...")
                        continue
                    raise HTTPException(status_code=500, detail="No se pudo crear sesión en Qwen.")
                await session_manager.register_new_session(client_id, session_id, model)

            logger.info(f"💬 Petición: Cliente={client_id}, Modelo={model}, Sesión={session_id[:8]}..., Thinking={'No' if MODEL_CONFIGS[model]['filter_phase'] else 'Sí'}")
            
            if not req_data.get("messages"):
                raise HTTPException(status_code=400, detail="El campo 'messages' no puede estar vacío.")
            
            # Convertir el último mensaje
            last_message_dict = req_data["messages"][-1]
            last_message = OpenAIMessage(**last_message_dict)
            
            if req_data.get("stream", False):
                async def stream_generator():
                    try:
                        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                        created_time = int(time.time())
                        
                        url = f"{QWEN_API_BASE_URL}/chat/completions?chat_id={session_id}"
                        payload = _build_qwen_completion_payload(session_id, last_message, model)
                        request_headers = {**QWEN_HEADERS, "Referer": f"https://chat.qwen.ai/c/{session_id}"}
                        
                        async with client.stream("POST", url, json=payload, headers=request_headers) as response:
                            response.raise_for_status()
                            
                            buffer = ""
                            async for chunk in response.aiter_text():
                                buffer += chunk
                                lines = buffer.split('\n')
                                buffer = lines.pop() if lines else ""
                                
                                for line in lines:
                                    line = line.strip()
                                    if not line or DONE_MARKER.match(line):
                                        continue
                                    
                                    if line.startswith('data:'):
                                        json_data = DATA_PREFIX.sub('', line, count=1).strip()
                                        if not json_data:
                                            continue
                                            
                                        try:
                                            qwen_chunk = JSON_DESERIALIZER(json_data)
                                            delta = qwen_chunk.get("choices", [{}])[0].get("delta", {})
                                            
                                            # Filtrado por fase según configuración del modelo
                                            model_config = MODEL_CONFIGS.get(model, {})
                                            filter_phase = model_config.get("filter_phase", True)
                                            
                                            if filter_phase and delta.get("phase") != "answer":
                                                continue
                                                
                                            if content_chunk := delta.get("content"):
                                                openai_chunk = {
                                                    "id": completion_id,
                                                    "object": "chat.completion.chunk",
                                                    "created": created_time,
                                                    "model": model,
                                                    "choices": [{
                                                        "index": 0,
                                                        "delta": {"content": content_chunk},
                                                        "finish_reason": None
                                                    }]
                                                }
                                                yield f"data: {JSON_SERIALIZER(openai_chunk, default=str)}\n\n"
                                                
                                        except (json.JSONDecodeError, KeyError, IndexError):
                                            continue
                        
                        # Chunk final
                        final_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {JSON_SERIALIZER(final_chunk, default=str)}\n\n"
                        yield "data: [DONE]\n\n"
                        logger.info(f"✅ Respuesta completada para sesión {session_id[:8]}...")
                        
                    except Exception as e:
                        error_chunk = {"error": {"message": f"Proxy error: {str(e)}", "type": "proxy_error"}}
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                    finally:
                        # Devolver sesión al pool después de completar el streaming
                        await session_manager.return_session_to_pool(session_id, model)
                
                # Headers para baja latencia en streaming
                headers = {
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Client-Session": client_id
                }
                
                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream",
                    headers=headers
                )
            else:
                # Modo no-streaming
                full_content = []
                try:
                    url = f"{QWEN_API_BASE_URL}/chat/completions?chat_id={session_id}"
                    payload = _build_qwen_completion_payload(session_id, last_message, model)
                    request_headers = {**QWEN_HEADERS, "Referer": f"https://chat.qwen.ai/c/{session_id}"}
                    
                    async with client.stream("POST", url, json=payload, headers=request_headers) as response:
                        response.raise_for_status()
                        
                        buffer = ""
                        async for chunk in response.aiter_text():
                            buffer += chunk
                            lines = buffer.split('\n')
                            buffer = lines.pop() if lines else ""
                            
                            for line in lines:
                                line = line.strip()
                                if not line or DONE_MARKER.match(line):
                                    continue
                                
                                if line.startswith('data:'):
                                    json_data = DATA_PREFIX.sub('', line, count=1).strip()
                                    if not json_data:
                                        continue
                                        
                                    try:
                                        qwen_chunk = JSON_DESERIALIZER(json_data)
                                        delta = qwen_chunk.get("choices", [{}])[0].get("delta", {})
                                        
                                        # Filtrado por fase según configuración del modelo
                                        model_config = MODEL_CONFIGS.get(model, {})
                                        filter_phase = model_config.get("filter_phase", True)
                                        
                                        if filter_phase and delta.get("phase") != "answer":
                                            continue
                                            
                                        if content_chunk := delta.get("content"):
                                            full_content.append(content_chunk)
                                            
                                    except (json.JSONDecodeError, KeyError, IndexError):
                                        continue
                
                finally:
                    await session_manager.return_session_to_pool(session_id, model)
                
                final_response = "".join(full_content)
                logger.info(f"✅ Respuesta no-streaming: {len(final_response)} caracteres")
                
                # Calcular tokens aproximados
                prompt_tokens = sum(len(msg.get('content', '').split()) for msg in req_data.get("messages", []))
                completion_tokens = len(final_response.split())
                
                return JSONResponse(content={
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": final_response
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                })
            
            break
            
        except HTTPException as e:
            # Devolver sesión al pool en caso de error
            if session_id:
                await session_manager.return_session_to_pool(session_id, model)
            raise e
        except Exception as e:
            # Devolver sesión al pool en caso de error
            if session_id:
                await session_manager.return_session_to_pool(session_id, model)
                
            if retry_count < max_retries:
                retry_count += 1
                logger.warning(f"⚠️ Error en intento {retry_count}: {e}. Reintentando...")
                await asyncio.sleep(1)
                continue
                
            logger.error(f"❌ Error fatal después de {max_retries+1} intentos.", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("🚀 Qwen Proxy Server v2.5.0 - OPTIMIZADO PARA ULTRA BAJA LATENCIA")
    print("="*80)
    print("\n✅ OPTIMIZACIONES ACTIVAS:")
    print("  ⚡ I/O Totalmente Asíncrono con uvloop")
    print("  📦 Pool de Sesiones con Pre-warming")
    print("  🔗 Conexiones HTTP/2 Persistentes")
    print("  ⚡ Parsing Optimizado con Regex Pre-compilado")
    print("  📊 Middleware de Monitoreo de Latencia")
    print("  🍪 Soporte para autenticación por cookies")
    print("\n🔐 MÉTODO DE AUTENTICACIÓN:")
    print(f"  • Token: {'✅ Disponible' if QWEN_AUTH_TOKEN else '❌ No disponible'}")
    print(f"  • Cookies: {'✅ Disponible' if QWEN_COOKIE_STRING else '❌ No disponible'}")
    print("\n🤖 MODELOS DISPONIBLES:")
    for model_id, config in MODEL_CONFIGS.items():
        thinking_status = "🚫 Sin pensamiento" if config["filter_phase"] else "💭 Con pensamiento"
        print(f"  • {model_id}: {config['display_name']} ({thinking_status})")
    print("\n🔗 ENDPOINTS:")
    print("  • GET  /                       - Información y estadísticas del servidor")
    print("  • GET  /stats                  - Estadísticas detalladas de sesiones")
    print("  • GET  /v1/models              - Lista de modelos disponibles")
    print("  • POST /v1/chat/completions     - Endpoint principal de chat")
    print("\n⚙️  CONFIGURACIÓN PARA CLIENTES:")
    print(f"  • URL base: http://localhost:5001/v1")
    print(f"  • API Key: Cualquier valor (ej: sk-12345)")
    print("\n" + "="*80 + "\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=5001,
        loop="uvloop",
        http="httptools",
        timeout_keep_alive=5,
        timeout_graceful_shutdown=1.0,
        limit_concurrency=100,
        backlog=2048,
    )
