# Flexible Steering Vector API - Setup & Deployment

Complete guide to setting up and deploying the flexible steering vector API.

## Files

### Core Implementation
- **`chat_fastapi_flexible.py`** - Main router with Pydantic models
- **`main_fastapi.py`** - Main FastAPI application (can be reused)

### Documentation
- **`FLEXIBLE_API_GUIDE.md`** - Complete API reference
- **`FLEXIBLE_API_EXAMPLES.md`** - Usage examples
- **`API_COMPARISON.md`** - Preset vs Flexible comparison
- **`FLEXIBLE_API_SETUP.md`** - This file

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_fastapi.txt
```

### 2. Create Flexible Main App

Create `main_flexible.py`:

```python
"""
FastAPI application for flexible steering vector control.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import the flexible router
from chat_fastapi_flexible import chat_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="EasySteer Flexible API",
    description="Flexible API for text generation with steering vectors",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(chat_router)

@app.get("/")
async def root():
    return {
        "service": "EasySteer Flexible API",
        "version": "1.0.0",
        "docs": "/docs",
        "guide": "/api/schema"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
```

### 3. Run the Server

**Development**:
```bash
uvicorn main_flexible:app --reload
```

**Production**:
```bash
gunicorn main_flexible:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

### 4. Test the API

```bash
curl -X POST "http://localhost:5000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello!",
    "model": {"path": "Qwen/Qwen2.5-7B-Instruct"},
    "steering_vector": {
      "path": "vectors/happiness.pt",
      "scale": 2.0,
      "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    }
  }'
```

## Advanced Setup

### Option 1: Run Both APIs (Recommended)

Run preset-based and flexible APIs on different ports:

```bash
# Terminal 1: Preset-based API
uvicorn main_fastapi:app --port 5000 --reload

# Terminal 2: Flexible API
uvicorn main_flexible:app --port 5001 --reload
```

Update frontend to use appropriate endpoint:

```javascript
// Production code uses presets
fetch('http://localhost:5000/api/chat', {...})

// Research/experimentation uses flexible
fetch('http://localhost:5001/api/chat', {...})
```

### Option 2: Single API with Both Routes

Combine both routers in one app:

```python
from fastapi import FastAPI
from chat_fastapi import chat_router as preset_router
from chat_fastapi_flexible import chat_router as flexible_router

app = FastAPI()

# Include both routers with different prefixes
app.include_router(preset_router, prefix="/api/v1")
app.include_router(flexible_router, prefix="/api/v2")
```

Access:
- Preset API: `/api/v1/chat`
- Flexible API: `/api/v2/chat`

### Option 3: Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements_fastapi.txt .
RUN pip install --no-cache-dir -r requirements_fastapi.txt

# Copy application code
COPY . .

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV VLLM_USE_V1=1

# Run the application
CMD ["uvicorn", "main_flexible:app", "--host", "0.0.0.0", "--port", "5000"]
```

Build and run:

```bash
# Build
docker build -t easysteer-api .

# Run
docker run --gpus all -p 5000:5000 easysteer-api
```

### Option 4: Kubernetes Deployment

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: easysteer-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: easysteer-api
  template:
    metadata:
      labels:
        app: easysteer-api
    spec:
      containers:
      - name: api
        image: easysteer-api:latest
        ports:
        - containerPort: 5000
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: easysteer-api
spec:
  type: LoadBalancer
  selector:
    app: easysteer-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
```

Deploy:

```bash
kubectl apply -f k8s-deployment.yaml
```

## Configuration

### Environment Variables

```bash
# GPU configuration
export CUDA_VISIBLE_DEVICES=0

# vLLM configuration
export VLLM_USE_V1=1

# Logging
export LOG_LEVEL=INFO
```

### Model Paths

Update default model path in requests:

```json
{
  "model": {
    "path": "path/to/your/model"
  }
}
```

### Vector Paths

Use absolute paths for vectors:

```json
{
  "steering_vector": {
    "path": "/absolute/path/to/vectors/happiness.pt"
  }
}
```

## Monitoring & Logging

### Basic Logging

Server logs show requests and errors:

```
2025-01-15 10:30:45 INFO     Chat request received: model=Qwen/Qwen2.5-7B-Instruct, steering=yes
2025-01-15 10:30:47 INFO     Chat request completed with steering: scale=2.0, layers=18
```

### Health Monitoring

Check service health:

```bash
curl http://localhost:5000/health
```

Response:
```json
{"status": "healthy"}
```

### Structured Logging

Enable structured logging:

```python
import json
import logging

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage()
        })

handler = logging.StreamHandler()
handler.setFormatter(StructuredFormatter())
logger.addHandler(handler)
```

## Performance Tuning

### GPU Memory

Reduce memory usage:

```python
# In chat_fastapi_flexible.py
llm = LLM(
    model=model_path,
    enable_steer_vector=True,
    enforce_eager=True,
    tensor_parallel_size=gpu_count,
    max_model_len=8192,  # Reduce from 16384
    gpu_memory_utilization=0.7  # Use 70% of GPU memory
)
```

### Batch Processing

Process multiple requests:

```python
import asyncio

async def batch_chat(requests):
    """Process multiple chat requests concurrently"""
    tasks = [chat(req) for req in requests]
    return await asyncio.gather(*tasks)
```

### Caching

Cache LLM instances:

```python
# Already implemented in chat_fastapi_flexible.py
llm_instances = {}  # Reuses loaded models
```

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce max_model_len
llm = LLM(..., max_model_len=8192)

# Reduce gpu_memory_utilization
llm = LLM(..., gpu_memory_utilization=0.5)
```

### Slow Inference

```python
# Reduce max_tokens
# Check GPU utilization
nvidia-smi

# Check CPU usage
top
```

### Vector File Not Found

Use absolute paths:

```json
{
  "steering_vector": {
    "path": "/home/user/vectors/happiness.pt"
  }
}
```

### Model Loading Timeout

Increase timeout:

```bash
# In gunicorn
gunicorn main_flexible:app --timeout 300
```

## Testing

### Unit Tests

```python
from fastapi.testclient import TestClient
from main_flexible import app

client = TestClient(app)

def test_chat_no_steering():
    response = client.post("/api/chat", json={
        "message": "Hello!",
        "model": {"path": "Qwen/Qwen2.5-7B-Instruct"}
    })
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert response.json()["steered_response"] is None

def test_chat_with_steering():
    response = client.post("/api/chat", json={
        "message": "Hello!",
        "model": {"path": "Qwen/Qwen2.5-7B-Instruct"},
        "steering_vector": {
            "path": "vectors/happiness.pt",
            "scale": 2.0,
            "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        }
    })
    assert response.status_code == 200
    assert response.json()["steered_response"] is not None
```

### Load Testing

```bash
# Using Apache Bench
ab -n 100 -c 10 http://localhost:5000/api/chat

# Using wrk
wrk -t4 -c100 -d30s http://localhost:5000/api/chat
```

## Security Considerations

### 1. Input Validation

- Pydantic validates all inputs
- Vector paths are checked for existence
- Model paths are validated

### 2. Rate Limiting

Add rate limiting (optional):

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter

@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat(...):
    ...
```

### 3. Authentication

Add API key authentication (optional):

```python
from fastapi import Header, HTTPException

async def verify_api_key(x_token: str = Header(...)):
    if x_token != "your-secret-key":
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_token

@app.post("/api/chat")
async def chat(..., token: str = Depends(verify_api_key)):
    ...
```

## Production Checklist

- [ ] Install all dependencies
- [ ] Test with GPU available
- [ ] Configure environment variables
- [ ] Set up logging
- [ ] Test health endpoint
- [ ] Test with sample vectors
- [ ] Run load tests
- [ ] Set up monitoring
- [ ] Configure auto-restart
- [ ] Document deployment
- [ ] Set up backups
- [ ] Plan for scaling

## Support

For issues or questions:
1. Check `/api/schema` for request format
2. Review `FLEXIBLE_API_GUIDE.md` for full documentation
3. Check `FLEXIBLE_API_EXAMPLES.md` for usage examples
4. Review server logs for errors

## Next Steps

1. **Development**: Test the API locally
2. **Testing**: Run unit and load tests
3. **Staging**: Deploy to staging environment
4. **Production**: Deploy to production
5. **Monitoring**: Set up monitoring and alerting
6. **Scaling**: Scale horizontally if needed
