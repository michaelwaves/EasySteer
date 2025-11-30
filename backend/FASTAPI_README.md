# FastAPI Chat Implementation

This directory contains a refactored FastAPI implementation of the EasySteer chat service, replacing the previous Flask blueprint approach.

## Files Created

### 1. **chat_fastapi.py** - Chat Router with Pydantic Models
The core chat functionality as a FastAPI router.

**Key Components**:
- `ChatMessage` - Pydantic model for a single message
- `ChatRequest` - Request schema with full validation
- `ChatResponse` - Response schema with type hints
- `chat()` - Main endpoint handler (POST /api/chat)
- `health_check()` - Service health check
- `list_presets()` - List available presets

**Features**:
- Automatic request/response validation
- Type hints for IDE support
- Comprehensive docstrings
- Error handling with HTTPException
- Helper functions for LLM management

### 2. **main_fastapi.py** - FastAPI Application
The main FastAPI application that integrates all routers.

**Key Components**:
- FastAPI app initialization
- CORS middleware configuration
- Lifecycle event handlers (startup/shutdown)
- Root endpoints
- Router inclusion
- Global exception handlers

**Features**:
- Clean app structure
- Lifecycle management
- OpenAPI documentation configuration
- Error handling
- Health check endpoints

### 3. **requirements_fastapi.txt** - Dependencies
List of all Python packages needed for the FastAPI implementation.

**Includes**:
- FastAPI and Uvicorn
- Pydantic for validation
- vLLM for LLM inference
- Development tools (pytest, mypy, black)

### 4. **FASTAPI_MIGRATION_GUIDE.md** - Comprehensive Migration Guide
Detailed guide for migrating from Flask to FastAPI.

**Covers**:
- Overview of FastAPI benefits
- Side-by-side comparison of key changes
- File structure and organization
- Pydantic model definitions
- Running the application
- API endpoint documentation
- Migration checklist
- Client code changes (none needed!)
- Testing strategies
- Troubleshooting

### 5. **FASTAPI_COMPARISON.md** - Detailed Feature Comparison
In-depth comparison between Flask and FastAPI implementations.

**Sections**:
1. Request Validation (Flask manual vs FastAPI automatic)
2. Error Handling (manual returns vs HTTPException)
3. Response Models (untyped vs typed Pydantic)
4. Complex Types (nested objects)
5. Documentation (manual vs automatic)
6. Routing (blueprints vs routers)
7. Type Safety (optional vs enforced)
8. Performance metrics
9. Testing approaches
10. Configuration & deployment

### 6. **QUICKSTART_FASTAPI.md** - Quick Start Guide
Get up and running in 5 minutes.

**Includes**:
- Installation steps
- Starting the server
- Using the API (curl, Python, JavaScript)
- Available presets
- Health checks
- Configuration
- Common requests
- Troubleshooting
- Performance tips

### 7. **FASTAPI_README.md** - This File
Overview and documentation of all FastAPI implementation files.

## Architecture Overview

```
main_fastapi.py
├── CORS Middleware
├── Lifecycle Events (startup/shutdown)
├── Root Routes
│   ├── GET /
│   ├── GET /health
│   └── GET /docs (Swagger)
└── Router Inclusion
    └── chat_router (from chat_fastapi.py)
        ├── POST /api/chat
        ├── GET /api/health/chat
        └── GET /api/presets
```

## Key Features

### 1. Automatic Validation
```python
class ChatRequest(BaseModel):
    message: str  # Required
    temperature: float = Field(0.8, ge=0.0, le=2.0)  # With constraints
    history: List[ChatMessage] = Field(default_factory=list)  # Nested objects
```

### 2. Type Safety
```python
@chat_router.post("/chat", response_model=ChatResponse)
async def chat(request_data: ChatRequest) -> ChatResponse:
    # Type hints prevent errors
    # IDE provides autocompletion
    # Return value validated against ChatResponse schema
```

### 3. Automatic Documentation
- Swagger UI at `/docs`
- ReDoc at `/redoc`
- OpenAPI schema at `/openapi.json`
- All generated from code

### 4. Error Handling
```python
raise HTTPException(
    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    detail="Error message"
)
# Automatic JSON response with correct status code
```

### 5. Modular Design
- Router can be included in any FastAPI app
- Helper functions are reusable
- Pydantic models can be imported elsewhere
- Clean separation of concerns

## Comparison with Flask Version

| Aspect | Flask | FastAPI |
|--------|-------|---------|
| Validation | Manual | Automatic (Pydantic) |
| Type Hints | Optional | Required |
| Documentation | Manual | Automatic |
| Error Handling | Manual try/except | HTTPException |
| Response Models | Untyped dict | Typed Pydantic |
| Testing | Manual setup | Built-in TestClient |
| IDE Support | Basic | Excellent |
| Performance | Good | Excellent |

## Installation & Running

### Install
```bash
pip install -r requirements_fastapi.txt
```

### Development
```bash
uvicorn main_fastapi:app --reload
```

### Production
```bash
gunicorn main_fastapi:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

## API Endpoints

### Chat
- **POST** `/api/chat` - Generate chat response with optional steering

Request:
```json
{
    "preset": "happy_mode",
    "message": "Hello!",
    "history": [],
    "temperature": 0.8,
    "max_tokens": 512
}
```

Response:
```json
{
    "success": true,
    "normal_response": "Hi there!",
    "steered_response": "Hi there! I'm doing great! 😊",
    "preset": "happy_mode"
}
```

### Health Checks
- **GET** `/health` - Global health check
- **GET** `/api/health/chat` - Chat service health

### Information
- **GET** `/` - API info
- **GET** `/api/presets` - Available presets
- **GET** `/docs` - Swagger UI
- **GET** `/redoc` - ReDoc documentation

## Pydantic Models

### ChatMessage
```python
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str  # Message content
```

### ChatRequest
```python
class ChatRequest(BaseModel):
    preset: str = "happy_mode"
    message: str
    history: List[ChatMessage] = []
    steered_history: List[ChatMessage] = []
    gpu_devices: str = "0"
    temperature: float = 0.8  # 0.0-2.0
    max_tokens: int = 512  # 1-4096
    repetition_penalty: float = 1.1  # 1.0-2.0
```

### ChatResponse
```python
class ChatResponse(BaseModel):
    success: bool
    normal_response: str
    steered_response: str
    preset: str
    error: Optional[str] = None
```

## Usage Examples

### Python
```python
import requests

response = requests.post(
    "http://localhost:5000/api/chat",
    json={"preset": "happy_mode", "message": "Hi!"}
)
print(response.json())
```

### JavaScript
```javascript
const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        preset: 'happy_mode',
        message: 'Hi!'
    })
});
const data = await response.json();
```

### cURL
```bash
curl -X POST "http://localhost:5000/api/chat" \
    -H "Content-Type: application/json" \
    -d '{"preset":"happy_mode","message":"Hi!"}'
```

## Project Structure

```
frontend/
├── main_fastapi.py                    # Main FastAPI app
├── chat_fastapi.py                    # Chat router implementation
├── requirements_fastapi.txt           # Python dependencies
├── FASTAPI_README.md                  # This file
├── FASTAPI_MIGRATION_GUIDE.md         # Migration guide
├── FASTAPI_COMPARISON.md              # Detailed comparison
├── QUICKSTART_FASTAPI.md              # Quick start guide
│
├── configs/                           # Configuration files
│   └── chat/
│       ├── happy_mode.json
│       ├── chinese_mode.json
│       ├── reject_mode.json
│       └── cat_mode.json
│
└── (old Flask files - kept for reference)
    ├── app.py
    ├── chat_api.py
    ├── inference_api.py
    └── extraction_api.py
```

## Migration Status

- [x] Chat endpoint (complete)
- [ ] Inference endpoint (planned)
- [ ] Extraction endpoint (planned)
- [ ] Authentication (planned)
- [ ] Rate limiting (planned)
- [ ] Metrics/monitoring (planned)

## Benefits of This Implementation

1. **Type Safety**: Catch errors at validation time
2. **Documentation**: Auto-generated OpenAPI docs
3. **Validation**: Pydantic handles all input validation
4. **Error Handling**: Consistent error responses
5. **Testing**: Built-in TestClient for testing
6. **Performance**: FastAPI is optimized for performance
7. **Maintainability**: Clear code structure
8. **IDE Support**: Better autocompletion and hints

## Next Steps

1. **Test**: Run the server and test endpoints
2. **Deploy**: Use the production configuration
3. **Extend**: Add more endpoints (inference, extraction)
4. **Monitor**: Set up logging and metrics
5. **Scale**: Deploy with Docker/Kubernetes if needed

## Support & Documentation

- **Interactive Docs**: http://localhost:5000/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Pydantic Docs**: https://docs.pydantic.dev/
- **Uvicorn Docs**: https://www.uvicorn.org/

## FAQ

**Q: Do I need to change my frontend code?**
A: No! The API endpoints remain the same. Your frontend can use the FastAPI version without any changes.

**Q: Why FastAPI over Flask?**
A: FastAPI provides automatic validation, documentation, and better type safety while being faster than Flask.

**Q: Can I run both Flask and FastAPI?**
A: Yes, run them on different ports during migration. Flask on 5000, FastAPI on 8000, for example.

**Q: How do I add more endpoints?**
A: Create a new router similar to `chat_fastapi.py` and include it in `main_fastapi.py`.

**Q: Is there authentication?**
A: Not yet, but can be added easily with FastAPI dependencies.

## Contributing

To extend this implementation:

1. Create a new router file (e.g., `inference_fastapi.py`)
2. Define Pydantic models for request/response
3. Implement route handlers
4. Include router in `main_fastapi.py`
5. Document in this README

## License

Same as EasySteer project
