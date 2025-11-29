# FastAPI Migration Guide

This guide explains how to migrate from the Flask-based chat API to the new FastAPI implementation.

## Overview

The FastAPI version provides:
- **Type Safety**: Pydantic models for request/response validation
- **Automatic Documentation**: OpenAPI/Swagger docs at `/docs`
- **Better Error Handling**: Consistent error responses
- **Async Support**: Ready for async operations
- **Code Reusability**: Clear separation of concerns

## Key Changes

### 1. Request/Response Models

**Flask (Old)**:
```python
@chat_bp.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    preset = data.get('preset', 'happy_mode')
    message = data.get('message', '')
    # ... manual validation
```

**FastAPI (New)**:
```python
class ChatRequest(BaseModel):
    preset: str = Field("happy_mode", description="Preset steering vector mode")
    message: str = Field(..., description="User message")
    # ... fields with type hints and validation

@chat_router.post("/chat", response_model=ChatResponse)
async def chat(request_data: ChatRequest) -> ChatResponse:
    # Automatic validation and type hints
```

### 2. Parameter Validation

**Before**: Manual validation with `.get()` defaults
```python
temperature = float(data.get('temperature', 0.8))
max_tokens = int(data.get('max_tokens', 512))
```

**After**: Pydantic handles validation and conversion
```python
temperature: float = Field(0.8, ge=0.0, le=2.0)
max_tokens: int = Field(512, ge=1, le=4096)
```

### 3. Error Handling

**Before**:
```python
return jsonify({'success': False, 'error': str(e)}), 500
```

**After**:
```python
raise HTTPException(
    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    detail=f"Generation error: {str(e)}"
)
# Automatic JSON response with proper status code
```

## File Structure

```
frontend/
├── main_fastapi.py              # Main FastAPI application
├── chat_fastapi.py              # Chat router with Pydantic models
├── inference_fastapi.py         # (To be created) Inference router
├── extraction_fastapi.py        # (To be created) Extraction router
├── FASTAPI_MIGRATION_GUIDE.md   # This file
└── (old Flask files)
    ├── app.py                   # Keep for reference
    ├── chat_api.py              # Keep for reference
    ├── inference_api.py         # Keep for reference
    └── extraction_api.py        # Keep for reference
```

## Pydantic Models

### ChatRequest
All parameters with proper validation:
```python
preset: str = "happy_mode"           # Available presets: happy_mode, chinese, reject_mode, cat_mode
message: str                          # User message (required)
history: List[ChatMessage]           # Chat history
steered_history: List[ChatMessage]   # Steered chat history
gpu_devices: str = "0"               # GPU device IDs
temperature: float = 0.8             # 0.0 to 2.0
max_tokens: int = 512                # 1 to 4096
repetition_penalty: float = 1.1      # 1.0 to 2.0
```

### ChatMessage
Represents a single message:
```python
role: str                            # "user" or "assistant"
content: str                         # Message content
```

### ChatResponse
Response from the chat endpoint:
```python
success: bool                        # True if successful
normal_response: str                 # Response without steering
steered_response: str                # Response with steering
preset: str                          # Preset used
error: Optional[str] = None          # Error message if unsuccessful
```

## Running the Application

### Install Dependencies
```bash
pip install fastapi uvicorn pydantic
```

### Development (with auto-reload)
```bash
uvicorn main_fastapi:app --host 0.0.0.0 --port 5000 --reload
```

### Production
```bash
# Using gunicorn with uvicorn workers
gunicorn main_fastapi:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:5000
```

## API Endpoints

### POST /api/chat
Generate text with optional steering vectors.

**Request Body**:
```json
{
    "preset": "happy_mode",
    "message": "Hello, how are you?",
    "history": [],
    "steered_history": [],
    "gpu_devices": "0",
    "temperature": 0.8,
    "max_tokens": 512,
    "repetition_penalty": 1.1
}
```

**Response**:
```json
{
    "success": true,
    "normal_response": "I'm doing well, thank you for asking!",
    "steered_response": "I'm absolutely fantastic, thank you so much for asking! 😊",
    "preset": "happy_mode"
}
```

### GET /api/health/chat
Check chat service health.

**Response**:
```json
{
    "status": "healthy",
    "service": "chat",
    "loaded_presets": 4
}
```

### GET /api/presets
List available presets.

**Response**:
```json
{
    "presets": {
        "happy_mode": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "scale": 2.0,
            "algorithm": "direct"
        },
        ...
    }
}
```

### GET /health
Global health check.

**Response**:
```json
{
    "status": "healthy",
    "service": "EasySteer API"
}
```

### GET /
Root endpoint with API info.

## OpenAPI/Swagger Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:5000/docs
- **ReDoc**: http://localhost:5000/redoc
- **OpenAPI JSON**: http://localhost:5000/openapi.json

## Migration Checklist

- [ ] Install FastAPI and Uvicorn
- [ ] Test chat_fastapi.py endpoints
- [ ] Update frontend client to use new endpoint (no changes needed, same URL)
- [ ] Test inference endpoint (create inference_fastapi.py if needed)
- [ ] Test extraction endpoint (create extraction_fastapi.py if needed)
- [ ] Update deployment configuration to use uvicorn instead of flask
- [ ] Test with actual LLM models
- [ ] Monitor logs for errors
- [ ] Archive old Flask files

## Client Code Changes

No changes needed! The API endpoints remain the same:

```javascript
// Before (Flask) and After (FastAPI) - same URL
const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        preset: 'happy_mode',
        message: 'Hello!',
        history: [],
        steered_history: []
    })
});
```

## Benefits of FastAPI

1. **Type Hints**: Catch errors at validation time, not runtime
2. **Auto Documentation**: OpenAPI docs generated automatically
3. **Validation**: Pydantic validates all inputs
4. **Performance**: Faster than Flask by design
5. **Async**: Ready for async operations (models can be async in future)
6. **Standards**: Based on industry standards (OpenAPI, JSON Schema)

## Common Issues and Solutions

### Issue: "No module named 'fastapi'"
**Solution**: `pip install fastapi uvicorn`

### Issue: Port 5000 already in use
**Solution**: `uvicorn main_fastapi:app --port 8000`

### Issue: CORS errors in frontend
**Solution**: Already handled with CORSMiddleware in main_fastapi.py

### Issue: Presets not loading
**Solution**: Ensure config files exist in `configs/chat/` directory

## Testing the API

```bash
# Using curl
curl -X POST "http://localhost:5000/api/chat" \
    -H "Content-Type: application/json" \
    -d '{
        "preset": "happy_mode",
        "message": "Hello!",
        "history": []
    }'

# Using Python
import requests
response = requests.post(
    "http://localhost:5000/api/chat",
    json={
        "preset": "happy_mode",
        "message": "Hello!",
        "history": []
    }
)
print(response.json())
```

## Next Steps

1. Create `inference_fastapi.py` for the inference endpoint
2. Create `extraction_fastapi.py` for the extraction endpoint
3. Update frontend to point to FastAPI server
4. Add authentication if needed (JWT, API keys, etc.)
5. Add rate limiting
6. Add caching for LLM instances
7. Add monitoring and metrics

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
