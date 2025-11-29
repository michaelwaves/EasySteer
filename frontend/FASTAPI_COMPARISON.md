# FastAPI vs Flask Comparison

This document shows the key differences between the old Flask implementation and the new FastAPI implementation.

## 1. Request Validation

### Flask Approach
```python
@chat_bp.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json

        # Manual extraction with defaults
        preset = data.get('preset', 'happy_mode')
        message = data.get('message', '')
        history = data.get('history', [])
        temperature = float(data.get('temperature', 0.8))
        max_tokens = int(data.get('max_tokens', 512))

        # Manual validation
        if not message:
            return jsonify({'error': 'Missing required field: message'}), 400
        if temperature < 0 or temperature > 2.0:
            return jsonify({'error': 'Temperature must be between 0 and 2'}), 400

        # ... rest of implementation
```

**Problems**:
- Lots of boilerplate code
- Validation errors return plain JSON
- Type conversions can fail with cryptic errors
- No documentation of expected parameters
- Error handling is manual

### FastAPI Approach
```python
class ChatRequest(BaseModel):
    preset: str = Field("happy_mode", description="Preset steering vector mode")
    message: str = Field(..., description="User message")
    history: List[ChatMessage] = Field(default_factory=list)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=1, le=4096)

@chat_router.post("/chat", response_model=ChatResponse)
async def chat(request_data: ChatRequest) -> ChatResponse:
    # request_data is already validated and converted
    # No manual validation needed
    # ... implementation
```

**Advantages**:
- Automatic validation with clear error messages
- Type hints prevent errors
- Parameters documented in the model
- Consistent error format
- OpenAPI documentation generated automatically

---

## 2. Error Handling

### Flask Pattern
```python
try:
    llm = get_or_create_llm(model_path, gpu_devices)
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    return jsonify({
        'success': False,
        'error': f"Failed to load model: {str(e)}"
    }), 500
```

### FastAPI Pattern
```python
try:
    llm = get_or_create_llm(model_path, gpu_devices)
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Failed to load model: {str(e)}"
    )
```

**FastAPI Benefits**:
- Automatic HTTP status code handling
- Consistent response format
- Global exception handlers for all errors
- Can use HTTP status constants for clarity

---

## 3. Response Model

### Flask (Untyped)
```python
response = {
    'success': True,
    'normal_response': normal_response,
    'steered_response': steered_response,
    'preset': preset
}
return jsonify(response)
```

### FastAPI (Typed with Pydantic)
```python
class ChatResponse(BaseModel):
    success: bool
    normal_response: str
    steered_response: str
    preset: str
    error: Optional[str] = None

@chat_router.post("/chat", response_model=ChatResponse)
async def chat(...) -> ChatResponse:
    return ChatResponse(
        success=True,
        normal_response=normal_response,
        steered_response=steered_response,
        preset=preset
    )
```

**FastAPI Benefits**:
- Type hints validate response
- IDE autocomplete support
- OpenAPI documentation includes response schema
- Catch response errors at development time

---

## 4. Request Bodies with Complex Types

### Flask (Manual handling)
```python
# Parse history manually
history = data.get('history', [])
for msg in history:
    role = msg.get('role')
    content = msg.get('content')
    if not role or not content:
        return jsonify({'error': 'Invalid history format'}), 400
    # Process message
```

### FastAPI (Automatic)
```python
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    history: List[ChatMessage] = Field(default_factory=list)

# In route handler:
@chat_router.post("/chat")
async def chat(request_data: ChatRequest):
    for msg in request_data.history:
        role = msg.role  # Type hint: str
        content = msg.content  # Type hint: str
        # Process message - validation already done
```

**FastAPI Benefits**:
- Nested object validation
- Type hints for nested objects
- Clear schema documentation
- Easier to test

---

## 5. Documentation

### Flask
```python
@chat_bp.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat API endpoint - processes a chat request
    """
    # No automated documentation
    # Must create separate API docs manually
```

### FastAPI
```python
@chat_router.post("/chat", response_model=ChatResponse)
async def chat(request_data: ChatRequest) -> ChatResponse:
    """
    Chat API endpoint - processes a chat request and returns both normal and steered responses.

    The endpoint:
    1. Validates the preset configuration exists
    2. Loads the LLM model if needed
    3. Generates a baseline (non-steered) response
    4. Generates a steered response using the specified preset
    5. Returns both responses for comparison
    """
    # Automatic OpenAPI documentation at /docs
    # Interactive API explorer
    # Schema validation documented
```

**Automatic Endpoints**:
- `/docs` - Swagger UI
- `/redoc` - ReDoc documentation
- `/openapi.json` - OpenAPI schema

---

## 6. Routing

### Flask
```python
# app.py
from flask import Flask
app = Flask(__name__)
app.register_blueprint(chat_bp, url_prefix='')

# chat_api.py
chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/api/chat', methods=['POST'])
def chat():
    pass
```

### FastAPI
```python
# main_fastapi.py
from fastapi import FastAPI
app = FastAPI()
app.include_router(chat_router)

# chat_fastapi.py
chat_router = APIRouter(prefix="/api", tags=["chat"])

@chat_router.post("/chat")
async def chat(...):
    pass
```

**FastAPI Benefits**:
- Cleaner organization
- Router parameters (prefix, tags, dependencies)
- Automatic grouping in documentation

---

## 7. Type Safety Comparison

### Flask
```python
# No type hints - prone to errors
def get_or_create_llm(model_path, gpu_devices):
    key = f"{model_path}_{gpu_devices}"
    # Could pass wrong types, IDE doesn't help
    chat_llm_instances[key] = LLM(...)
```

### FastAPI
```python
# Type hints - IDE and linters help catch errors
def get_or_create_llm(model_path: str, gpu_devices: str = "0") -> LLM:
    key = f"{model_path}_{gpu_devices}"
    chat_llm_instances[key] = LLM(...)
    return chat_llm_instances[key]
```

**FastAPI Benefits**:
- IDE autocompletion
- Type checking with mypy
- Self-documenting code
- Catch errors before runtime

---

## 8. Performance Comparison

| Metric | Flask | FastAPI |
|--------|-------|---------|
| Framework Overhead | ~5ms per request | ~2ms per request |
| Parsing JSON | Manual | Automatic (Pydantic) |
| Validation | Manual loops | Automatic |
| Type Hints | Optional | Enforced |
| Async Support | Limited | Native |
| Auto Documentation | No | Yes (/docs) |
| Validation Performance | Slower | Faster (optimized) |

**Note**: For this application, difference is negligible since LLM inference dominates. FastAPI shines with high-concurrency scenarios.

---

## 9. Testing Comparison

### Flask Testing
```python
import json

def test_chat():
    response = client.post('/api/chat',
        data=json.dumps({
            'preset': 'happy_mode',
            'message': 'Hello'
        }),
        content_type='application/json'
    )
    data = json.loads(response.data)
    assert data['success'] == True
```

### FastAPI Testing
```python
from fastapi.testclient import TestClient

def test_chat():
    response = client.post("/api/chat",
        json={
            "preset": "happy_mode",
            "message": "Hello"
        }
    )
    assert response.status_code == 200
    assert response.json()['success'] == True
    # Automatic JSON parsing, status codes, etc.
```

**FastAPI Benefits**:
- Built-in TestClient
- Automatic JSON handling
- Status code assertions
- Response model validation

---

## 10. Configuration & Deployment

### Flask Requirements
```
Flask>=2.0.0
Werkzeug>=2.0.0
```

### FastAPI Requirements
```
fastapi>=0.95.0
uvicorn>=0.20.0
pydantic>=2.0.0
```

### Deployment

**Flask**:
```bash
python app.py  # Development
gunicorn app:app  # Production
```

**FastAPI**:
```bash
uvicorn main_fastapi:app --reload  # Development
gunicorn main_fastapi:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker  # Production
```

---

## Summary Table

| Feature | Flask | FastAPI |
|---------|-------|---------|
| Type Hints | Optional | Built-in |
| Request Validation | Manual | Automatic (Pydantic) |
| Response Validation | Manual | Automatic |
| Documentation | Manual | Automatic (/docs) |
| Error Handling | Manual try/except | HTTPException |
| Async Support | Limited | Native |
| Performance | Good | Excellent |
| Learning Curve | Shallow | Moderate |
| Code Verbosity | More | Less |
| IDE Support | Basic | Excellent |
| Testing | Manual | TestClient |

---

## Migration Path

1. **Phase 1**: Run FastAPI alongside Flask
   - Deploy `main_fastapi.py` on different port
   - Test with same requests as Flask version
   - Monitor for differences

2. **Phase 2**: Redirect traffic
   - Update frontend to use FastAPI endpoints
   - Monitor error rates and performance
   - Keep Flask running as fallback

3. **Phase 3**: Full migration
   - Decommission Flask
   - Archive Flask code
   - Monitor production metrics

---

## Recommendations

**Use FastAPI if**:
- You need automatic API documentation
- You want better error handling
- You value type safety
- You're building a production API
- You want async support
- You need validation

**Keep Flask if**:
- You have existing Flask infrastructure
- You prefer simplicity over features
- You don't need API documentation
- You have team expertise in Flask
