# FastAPI Chat API - Quick Start Guide

Get the EasySteer chat API up and running with FastAPI in 5 minutes.

## Installation

### 1. Install Dependencies
```bash
cd frontend
pip install -r requirements_fastapi.txt
```

### 2. Start the Server

**Development (with auto-reload)**:
```bash
uvicorn main_fastapi:app --reload --host 0.0.0.0 --port 5000
```

**With specific GPU**:
```bash
CUDA_VISIBLE_DEVICES=0 uvicorn main_fastapi:app --reload
```

**Production (with multiple workers)**:
```bash
gunicorn main_fastapi:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:5000 \
    --log-level info
```

## Using the API

### 1. Interactive Documentation

Once the server is running, open:
- **Swagger UI**: http://localhost:5000/docs
- **ReDoc**: http://localhost:5000/redoc

Try out endpoints directly from the browser!

### 2. Basic Chat Request

```bash
curl -X POST "http://localhost:5000/api/chat" \
    -H "Content-Type: application/json" \
    -d '{
        "preset": "happy_mode",
        "message": "Hello, how are you?",
        "history": [],
        "gpu_devices": "0"
    }'
```

### 3. Python Client

```python
import requests

# Chat with steering
response = requests.post(
    "http://localhost:5000/api/chat",
    json={
        "preset": "happy_mode",
        "message": "Hello, how are you?",
        "history": [],
        "temperature": 0.8,
        "max_tokens": 512
    }
)

data = response.json()
print("Normal response:", data['normal_response'])
print("Steered response:", data['steered_response'])
```

### 4. JavaScript/Frontend Client

```javascript
// Fetch chat response
async function getChat() {
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

    const data = await response.json();
    console.log('Normal:', data.normal_response);
    console.log('Steered:', data.steered_response);
}

getChat();
```

## Available Presets

Get the list of available presets:

```bash
curl "http://localhost:5000/api/presets"
```

Response:
```json
{
    "presets": {
        "happy_mode": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "scale": 2.0,
            "algorithm": "direct"
        },
        "chinese": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "scale": 1.5,
            "algorithm": "direct"
        },
        "reject_mode": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "scale": 2.0,
            "algorithm": "direct"
        },
        "cat_mode": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "scale": 1.8,
            "algorithm": "direct"
        }
    }
}
```

## Health Check

Check if the API is running:

```bash
curl "http://localhost:5000/health"
```

Response:
```json
{
    "status": "healthy",
    "service": "EasySteer API"
}
```

Check chat service specifically:

```bash
curl "http://localhost:5000/api/health/chat"
```

Response:
```json
{
    "status": "healthy",
    "service": "chat",
    "loaded_presets": 4
}
```

## Configuration

### Model Configuration

Edit the preset config files in `configs/chat/`:
- `happy_mode.json`
- `chinese_mode.json`
- `reject_mode.json`
- `cat_mode.json`

Example config structure:
```json
{
    "model": {
        "path": "Qwen/Qwen2.5-7B-Instruct"
    },
    "vector": {
        "path": "vectors/persona_vectors/Qwen2.5-7B-Instruct/happiness_response_avg_diff.pt",
        "scale": 2.0,
        "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        "algorithm": "direct",
        "prefill_trigger_token_ids": [-1],
        "generate_trigger_token_ids": [-1],
        "normalize": false
    }
}
```

## Common Requests

### 1. Simple Chat
```bash
curl -X POST "http://localhost:5000/api/chat" \
    -H "Content-Type: application/json" \
    -d '{
        "preset": "happy_mode",
        "message": "Tell me a joke",
        "history": []
    }'
```

### 2. Chat with History
```bash
curl -X POST "http://localhost:5000/api/chat" \
    -H "Content-Type: application/json" \
    -d '{
        "preset": "happy_mode",
        "message": "What is your favorite color?",
        "history": [
            {
                "role": "user",
                "content": "Hi, what is your name?"
            },
            {
                "role": "assistant",
                "content": "I am a helpful assistant!"
            }
        ]
    }'
```

### 3. Different Presets
```bash
# Chinese mode
curl -X POST "http://localhost:5000/api/chat" \
    -H "Content-Type: application/json" \
    -d '{
        "preset": "chinese",
        "message": "你好，你叫什么名字？"
    }'

# Cat mode
curl -X POST "http://localhost:5000/api/chat" \
    -H "Content-Type: application/json" \
    -d '{
        "preset": "cat_mode",
        "message": "What do you like to do?"
    }'
```

### 4. Custom Parameters
```bash
curl -X POST "http://localhost:5000/api/chat" \
    -H "Content-Type: application/json" \
    -d '{
        "preset": "happy_mode",
        "message": "Explain quantum physics",
        "temperature": 0.5,
        "max_tokens": 1024,
        "repetition_penalty": 1.2,
        "gpu_devices": "0"
    }'
```

## Troubleshooting

### Issue: "No module named 'fastapi'"
```bash
pip install fastapi uvicorn
```

### Issue: Port 5000 already in use
```bash
# Use different port
uvicorn main_fastapi:app --port 8000

# Or kill the process using port 5000
lsof -ti:5000 | xargs kill -9  # macOS/Linux
```

### Issue: Preset configs not found
```bash
# Ensure config files exist in correct location
ls -la frontend/configs/chat/

# Should show:
# - happy_mode.json
# - chinese_mode.json
# - reject_mode.json
# - cat_mode.json
```

### Issue: Model loading fails
```bash
# Check GPU availability
nvidia-smi

# Ensure model can be loaded
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"

# Check CUDA visibility
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Slow response times
- Increase `max_tokens` gradually to test
- Monitor GPU memory with `nvidia-smi`
- Check CPU and RAM usage
- Consider reducing batch size

## Performance Tips

1. **GPU Selection**: Use `gpu_devices` parameter
   ```json
   {"gpu_devices": "0,1"}  // Use multiple GPUs
   ```

2. **Model Caching**: Models are cached after first load
   - First request: ~5-10s (model loading)
   - Subsequent requests: ~2-3s (inference only)

3. **Batch Processing**: Not yet implemented, but planned for future

4. **Monitoring**: Check service health
   ```bash
   while true; do curl http://localhost:5000/health && sleep 5; done
   ```

## Next Steps

1. **Frontend Integration**: Update your web frontend to use the API
2. **Authentication**: Add API key validation
3. **Rate Limiting**: Prevent abuse with rate limits
4. **Monitoring**: Set up logging and metrics
5. **Deployment**: Deploy to production with gunicorn/docker

## More Information

- See `FASTAPI_MIGRATION_GUIDE.md` for detailed migration info
- See `FASTAPI_COMPARISON.md` for Flask vs FastAPI comparison
- Check `main_fastapi.py` for the main app code
- Check `chat_fastapi.py` for the chat router implementation

## Getting Help

1. Check the interactive docs at `/docs`
2. Review error messages in server logs
3. Consult Pydantic documentation for validation issues
4. Check vLLM documentation for model-related issues
