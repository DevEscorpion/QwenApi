import json
import logging
import time
import uuid
import asyncio
import hashlib
import re
import os
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
    def orjson_dumps(v, *, default):
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
QWEN_AUTH_TOKEN = os.getenv("QWEN_AUTH_TOKEN")
QWEN_COOKIE = os.getenv("QWEN_COOKIE")

if not QWEN_AUTH_TOKEN or not QWEN_COOKIE:
    raise RuntimeError("Las variables de entorno QWEN_AUTH_TOKEN y QWEN_COOKIE deben estar definidas en un archivo .env.")

# --- CONSTANTES DE LA API QWEN ---
QWEN_API_BASE_URL = "https://chat.qwen.ai/api/v2"
QWEN_INTERNAL_MODEL = "qwen3-235b-a22b"
MODEL_QWEN_FINAL = "qwen-final"
MODEL_QWEN_THINKING = "qwen-thinking"

QWEN_HEADERS = {
    "Accept": "application/json", "Accept-Language": "es-AR,es;q=0.7", "Authorization": QWEN_AUTH_TOKEN,
    "bx-v": "2.5.31", "Content-Type": "application/json; charset=UTF-8", "Cookie": QWEN_COOKIE,
    "Origin": "https://chat.qwen.ai", "Referer": "https://chat.qwen.ai/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
    "source": "web", "x-accel-buffering": "no",
}

MIN_CHAT_ID_POOL_SIZE = 2
MAX_CHAT_ID_POOL_SIZE = 5

# --- INICIALIZACIÓN DE LA APP Y LOGGING ---
app = FastAPI(title="Qwen Web API Proxy (Ultra Low Latency)", version="2.3.0")
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
            MODEL_QWEN_FINAL: asyncio.Queue(maxsize=MAX_CHAT_ID_POOL_SIZE),
            MODEL_QWEN_THINKING: asyncio.Queue(maxsize=MAX_CHAT_ID_POOL_SIZE)
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
            model = request_info.get("model", MODEL_QWEN_FINAL)
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
                    session_info["client_id"] = client_id
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
async def create_qwen_chat(client: httpx.AsyncClient) -> str | None:
    """Crea una nueva sesión de chat en Qwen y devuelve su ID. Usado por el pool."""
    url = f"{QWEN_API_BASE_URL}/chats/new"
    payload = {"title": "Proxy Pool Chat", "models": [QWEN_INTERNAL_MODEL], "chat_mode": "normal", "chat_type": "t2t", "timestamp": int(time.time() * 1000)}
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

def _build_qwen_completion_payload(chat_id: str, message: OpenAIMessage) -> Dict[str, Any]:
    current_timestamp = int(time.time())
    return {
        "stream": True, "incremental_output": True, "chat_id": chat_id, "chat_mode": "normal",
        "model": QWEN_INTERNAL_MODEL, "parent_id": None,
        "messages": [{
            "fid": str(uuid.uuid4()), "parentId": None, "role": message.role, "content": message.content,
            "user_action": "chat", "files": [], "timestamp": current_timestamp, "models": [QWEN_INTERNAL_MODEL],
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

async def optimized_stream_parser(
    client: httpx.AsyncClient, chat_id: str, message: OpenAIMessage, requested_model: str
) -> AsyncGenerator[str, None]:
    """Parser optimizado para streaming de baja latencia"""
    url = f"{QWEN_API_BASE_URL}/chat/completions?chat_id={chat_id}"
    payload = _build_qwen_completion_payload(chat_id, message)
    request_headers = {**QWEN_HEADERS, "Referer": f"https://chat.qwen.ai/c/{chat_id}"}
    
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created_timestamp = int(time.time())
    
    try:
        async with client.stream("POST", url, json=payload, headers=request_headers) as response:
            response.raise_for_status()
            
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                lines = buffer.split('\n')
                buffer = lines.pop() if lines else ""  # Guardar línea incompleta
                
                for line in lines:
                    line = line.strip()
                    if not line or DONE_MARKER.match(line):
                        continue
                    
                    # Eliminar prefijo "data: " más eficientemente
                    if line.startswith('data:'):
                        json_data = DATA_PREFIX.sub('', line, count=1).strip()
                        if not json_data:
                            continue
                            
                        try:
                            # Parseo directo sin validaciones adicionales
                            qwen_chunk = JSON_DESERIALIZER(json_data)
                            delta = qwen_chunk.get("choices", [{}])[0].get("delta", {})
                            
                            # Filtrado rápido por fase
                            if requested_model == MODEL_QWEN_FINAL and delta.get("phase") != "answer":
                                continue
                                
                            if content_chunk := delta.get("content"):
                                openai_chunk = OpenAICompletionChunk(
                                    id=completion_id, created=created_timestamp, model=requested_model,
                                    choices=[OpenAIChunkChoice(delta=OpenAIChunkDelta(content=content_chunk))],
                                )
                                yield _format_sse_chunk(openai_chunk)
                                
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
    
    except Exception as e:
        error_chunk = {"error": {"message": f"Proxy error: {str(e)}", "type": "proxy_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"
        return
    
    # Chunk final
    final_chunk = OpenAICompletionChunk(
        id=completion_id, created=created_timestamp, model=requested_model,
        choices=[OpenAIChunkChoice(delta=OpenAIChunkDelta(), finish_reason="stop")],
    )
    yield _format_sse_chunk(final_chunk)
    yield "data: [DONE]\n\n"

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
            for model in [MODEL_QWEN_FINAL, MODEL_QWEN_THINKING]:
                queue = session_manager._available_sessions[model]
                current_size = queue.qsize()
                
                # Mantener el pool siempre lleno para respuesta inmediata
                if current_size < MAX_CHAT_ID_POOL_SIZE:
                    num_to_create = MAX_CHAT_ID_POOL_SIZE - current_size
                    created = 0
                    
                    # Crear sesiones en paralelo para mayor velocidad
                    create_tasks = [create_qwen_chat(client) for _ in range(num_to_create)]
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
        "version": "2.3.0", "status": "running",
        "features": ["Async I/O", "Session Pooling", "Optimized Streaming", "Low Latency"],
        "endpoints": {"models": "/v1/models", "chat": "/v1/chat/completions", "stats": "/stats"},
        "session_stats": stats
    })

@app.get("/stats")
async def get_stats():
    return JSONResponse(content=await session_manager.get_detailed_stats())

@app.get("/v1/models")
def list_models():
    model_data = [
        {"id": MODEL_QWEN_FINAL, "object": "model", "created": int(time.time()), "owned_by": "qwen", "description": "Respuesta final."},
        {"id": MODEL_QWEN_THINKING, "object": "model", "created": int(time.time()), "owned_by": "qwen", "description": "Proceso de razonamiento completo."},
    ]
    return JSONResponse(content={"object": "list", "data": model_data})

@app.post("/v1/chat/completions")
async def low_latency_chat_completions(request: Request):
    try:
        # Limpieza periódica de sesiones expiradas
        if time.time() % 300 < 1:
            await session_manager.cleanup_expired_sessions()

        # Leer y parsear el cuerpo lo más rápido posible
        body = await request.body()
        req_data = JSON_DESERIALIZER(body)
        
        model = req_data.get("model", MODEL_QWEN_FINAL)
        request_info = {"headers": dict(request.headers), "model": model}
        
        client = get_http_client()
        client_id, session_id = await session_manager.get_or_create_session(request_info, client)
        
        if session_id is None:
            session_id = await create_qwen_chat(client)
            if not session_id:
                raise HTTPException(status_code=500, detail="No se pudo crear sesión en Qwen.")
            await session_manager.register_new_session(client_id, session_id, model)
        
        if not req_data.get("messages"):
            raise HTTPException(status_code=400, detail="El campo 'messages' no puede estar vacío.")
        
        last_message = req_data["messages"][-1]
        
        if req_data.get("stream", False):
            generator = optimized_stream_parser(
                client=client, chat_id=session_id, message=last_message, requested_model=model
            )
            
            # Headers para baja latencia en streaming
            headers = {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
            
            return StreamingResponse(
                generator, 
                media_type="text/event-stream",
                headers=headers
            )
        else:
            # Modo no-streaming (para compatibilidad)
            full_content = []
            async for chunk in optimized_stream_parser(client, session_id, last_message, model):
                if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                    try:
                        chunk_data = JSON_DESERIALIZER(chunk[6:].strip())
                        if "choices" in chunk_data and chunk_data["choices"][0]["delta"].get("content"):
                            full_content.append(chunk_data["choices"][0]["delta"]["content"])
                    except:
                        pass
            
            await session_manager.return_session_to_pool(session_id, model)
            
            return JSONResponse(content={
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion",
                "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(full_content)}, "finish_reason": "stop"}],
            })
            
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error en el endpoint de chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("🚀 Qwen Proxy Server v2.3.0 - OPTIMIZADO PARA ULTRA BAJA LATENCIA")
    print("="*80)
    print("\n✅ OPTIMIZACIONES ACTIVAS:")
    print("  ⚡ I/O Totalmente Asíncrono con uvloop")
    print("  📦 Pool de Sesiones con Pre-warming")
    print("  🔗 Conexiones HTTP/2 Persistentes")
    print("  ⚡ Parsing Optimizado con Regex Pre-compilado")
    print("  📊 Middleware de Monitoreo de Latencia")
    print("\n🤖 MODELOS DISPONIBLES:")
    print(f"  • {MODEL_QWEN_FINAL}: Modelo estándar (solo respuesta final)")
    print(f"  • {MODEL_QWEN_THINKING}: Modelo con proceso de razonamiento completo")
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
