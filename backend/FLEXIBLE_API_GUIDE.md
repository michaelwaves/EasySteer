# Flexible Steering Vector API Guide

This guide explains how to use the flexible steering vector API that allows full control over steering parameters without relying on presets.

## Overview

Instead of using predefined presets (happy_mode, chinese, etc.), the flexible API allows you to:
- Specify any model path
- Provide complete steering vector configuration
- Control all generation parameters
- Mix and match different configurations dynamically

## API Endpoint

```
POST /api/chat
```

## Request Structure

### Complete Request Example

```json
{
  "message": "Hello, how are you?",
  "history": [],
  "model": {
    "path": "Qwen/Qwen2.5-7B-Instruct"
  },
  "steering_vector": {
    "path": "/path/to/vectors/happiness_response_avg_diff.pt",
    "scale": 2.0,
    "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    "algorithm": "direct",
    "prefill_trigger_token_ids": [-1],
    "generate_trigger_token_ids": [-1],
    "normalize": false
  },
  "temperature": 0.8,
  "max_tokens": 512,
  "repetition_penalty": 1.1,
  "gpu_devices": "0",
  "debug": false
}
```

### Minimal Request (No Steering)

Generate response without steering:

```json
{
  "message": "Hello!",
  "model": {
    "path": "Qwen/Qwen2.5-7B-Instruct"
  }
}
```

## Request Parameters

### Required Fields

#### `message` (string)
The user message to respond to.

```json
"message": "Tell me a joke"
```

#### `model` (object)
Language model configuration.

```json
"model": {
  "path": "Qwen/Qwen2.5-7B-Instruct"
}
```

**Supported model paths**:
- HuggingFace model IDs: `"Qwen/Qwen2.5-7B-Instruct"`
- Local paths: `"/home/user/models/Qwen2.5-7B-Instruct"`

### Optional Fields

#### `steering_vector` (object)
Steering vector configuration. If omitted, generates only baseline response.

```json
"steering_vector": {
  "path": "/vectors/happiness_response_avg_diff.pt",
  "scale": 2.0,
  "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
  "algorithm": "direct",
  "prefill_trigger_token_ids": [-1],
  "generate_trigger_token_ids": [-1],
  "normalize": false
}
```

**Steering Vector Fields**:

- **`path`** (string, required): Path to vector file (.pt or .gguf)
  - Relative paths start from working directory
  - Use absolute paths for clarity

- **`scale`** (float, default: 1.0): Scale factor
  - Range: any positive number
  - Higher values = stronger steering effect
  - 0.0 = no steering

- **`target_layers`** (list of integers, default: [8-25]): Layers to apply steering
  - Layer indices depend on model architecture
  - Qwen2.5-7B: 0-31 layers available
  - Use lower layers for semantic changes
  - Use higher layers for style changes

- **`algorithm`** (string, default: "direct"): Steering algorithm
  - Options: `"direct"`, `"loreft"`
  - `"direct"`: Simple vector addition
  - `"loreft"`: Low-rank factorization

- **`prefill_trigger_token_ids`** (list of integers, default: [-1])
  - Token IDs to trigger steering during prefill phase
  - `-1` means apply to all tokens in prefill
  - Specific token IDs apply to those tokens only

- **`generate_trigger_token_ids`** (list of integers, default: [-1])
  - Token IDs to trigger steering during generation phase
  - `-1` means apply to all tokens in generation
  - Specific token IDs apply to those tokens only

- **`normalize`** (boolean, default: false): Normalize vector
  - Preserves hidden state norm
  - Useful for sensitive models

#### `history` (list of objects, default: [])
Chat history for normal (non-steered) response.

```json
"history": [
  {
    "role": "user",
    "content": "What is your name?"
  },
  {
    "role": "assistant",
    "content": "I'm an AI assistant."
  }
]
```

#### `steered_history` (list of objects, default: uses history)
Chat history for steered response. If not provided, uses `history`.

```json
"steered_history": [
  {
    "role": "user",
    "content": "What is your name?"
  },
  {
    "role": "assistant",
    "content": "I'm a super happy AI assistant! 😊"
  }
]
```

#### `temperature` (float, default: 0.8)
- Range: 0.0 to 2.0
- Lower = more deterministic
- Higher = more random

```json
"temperature": 0.5
```

#### `max_tokens` (integer, default: 512)
- Range: 1 to 4096
- Maximum tokens to generate

```json
"max_tokens": 1024
```

#### `repetition_penalty` (float, default: 1.1)
- Range: 1.0 to 2.0
- Prevents repeating tokens
- Higher = stronger penalty

```json
"repetition_penalty": 1.2
```

#### `gpu_devices` (string, default: "0")
GPU device IDs.

```json
"gpu_devices": "0"        // Single GPU
"gpu_devices": "0,1"      // Multiple GPUs
"gpu_devices": "0,1,2,3"  // Four GPUs
```

#### `debug` (boolean, default: false)
Enable debug output in server logs.

```json
"debug": true
```

## Response Structure

### Success Response

```json
{
  "success": true,
  "normal_response": "I'm doing well, thank you for asking!",
  "steered_response": "I'm absolutely fantastic! Thanks for asking! 😊",
  "config": {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "temperature": 0.8,
    "max_tokens": 512,
    "repetition_penalty": 1.1,
    "steering": {
      "path": "/vectors/happiness_response_avg_diff.pt",
      "scale": 2.0,
      "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
      "algorithm": "direct",
      "normalize": false
    }
  }
}
```

### Response Fields

- **`success`** (boolean): Whether request succeeded
- **`normal_response`** (string): Response without steering (always generated)
- **`steered_response`** (string or null): Response with steering (null if no steering config)
- **`config`** (object): Echo of configuration used

### Error Response

```json
{
  "success": false,
  "normal_response": "",
  "steered_response": null,
  "config": {},
  "error": "Failed to load model: CUDA out of memory"
}
```

## Usage Examples

### Example 1: Basic Chat (No Steering)

```bash
curl -X POST "http://localhost:5000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is machine learning?",
    "model": {
      "path": "Qwen/Qwen2.5-7B-Instruct"
    }
  }'
```

### Example 2: Chat with Happiness Steering

```bash
curl -X POST "http://localhost:5000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How are you feeling today?",
    "model": {
      "path": "Qwen/Qwen2.5-7B-Instruct"
    },
    "steering_vector": {
      "path": "vectors/persona_vectors/Qwen2.5-7B-Instruct/happiness_response_avg_diff.pt",
      "scale": 2.0,
      "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
      "algorithm": "direct",
      "prefill_trigger_token_ids": [-1],
      "generate_trigger_token_ids": [-1],
      "normalize": false
    }
  }'
```

### Example 3: Chat with Different Scaling

Experiment with different scales:

```bash
# Weak steering (scale=0.5)
curl -X POST "http://localhost:5000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello!",
    "model": {"path": "Qwen/Qwen2.5-7B-Instruct"},
    "steering_vector": {
      "path": "vectors/happiness.pt",
      "scale": 0.5,
      "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    }
  }'

# Strong steering (scale=3.0)
curl -X POST "http://localhost:5000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello!",
    "model": {"path": "Qwen/Qwen2.5-7B-Instruct"},
    "steering_vector": {
      "path": "vectors/happiness.pt",
      "scale": 3.0,
      "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    }
  }'
```

### Example 4: Python Client

```python
import requests
import json

def chat_with_steering(
    message,
    model_path,
    vector_path,
    scale=2.0,
    target_layers=None
):
    """Chat with steering vector"""

    if target_layers is None:
        target_layers = list(range(8, 26))

    response = requests.post(
        "http://localhost:5000/api/chat",
        json={
            "message": message,
            "model": {"path": model_path},
            "steering_vector": {
                "path": vector_path,
                "scale": scale,
                "target_layers": target_layers,
                "algorithm": "direct"
            }
        }
    )

    data = response.json()
    print("Normal:", data['normal_response'])
    print("Steered:", data['steered_response'])
    return data

# Use it
chat_with_steering(
    message="How are you?",
    model_path="Qwen/Qwen2.5-7B-Instruct",
    vector_path="vectors/happiness.pt",
    scale=2.0
)
```

### Example 5: JavaScript Client

```javascript
async function chatWithSteering(config) {
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            message: config.message,
            model: {
                path: config.modelPath
            },
            steering_vector: {
                path: config.vectorPath,
                scale: config.scale || 2.0,
                target_layers: config.targetLayers || [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
                algorithm: "direct"
            }
        })
    });

    return await response.json();
}

// Usage
const result = await chatWithSteering({
    message: "Tell me a joke",
    modelPath: "Qwen/Qwen2.5-7B-Instruct",
    vectorPath: "vectors/happiness.pt",
    scale: 2.0
});

console.log("Normal:", result.normal_response);
console.log("Steered:", result.steered_response);
```

## Advanced Usage

### Experiment with Different Layers

Test which layers are most effective:

```python
import requests

def test_layers(message, vector_path, scales=[0.5, 1.0, 2.0]):
    """Test steering at different layer ranges"""

    configs = {
        "early_layers": list(range(5, 10)),
        "middle_layers": list(range(10, 20)),
        "late_layers": list(range(20, 30)),
        "all_layers": list(range(0, 32)),
    }

    for name, layers in configs.items():
        for scale in scales:
            response = requests.post(
                "http://localhost:5000/api/chat",
                json={
                    "message": message,
                    "model": {"path": "Qwen/Qwen2.5-7B-Instruct"},
                    "steering_vector": {
                        "path": vector_path,
                        "scale": scale,
                        "target_layers": layers,
                        "algorithm": "direct"
                    }
                }
            )
            data = response.json()
            print(f"{name} (scale={scale}): {data['steered_response'][:50]}...")
```

### Compare Different Vectors

Test different steering vectors:

```python
vectors = [
    "vectors/happiness.pt",
    "vectors/sadness.pt",
    "vectors/curiosity.pt"
]

for vector_path in vectors:
    response = requests.post(
        "http://localhost:5000/api/chat",
        json={
            "message": "How do you feel?",
            "model": {"path": "Qwen/Qwen2.5-7B-Instruct"},
            "steering_vector": {
                "path": vector_path,
                "scale": 2.0,
                "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
            }
        }
    )
    data = response.json()
    print(f"Vector: {vector_path}")
    print(f"Response: {data['steered_response']}\n")
```

## Benefits of This Approach

1. **Flexibility**: No preset limitations
2. **Control**: Full control over all parameters
3. **Experimentation**: Easy to test different configurations
4. **Dynamic**: Change configuration per request
5. **Transparent**: All parameters visible in request/response
6. **Scalable**: Add new vectors without modifying API

## Schema Endpoint

Get the full JSON schema for validation:

```bash
curl -X POST "http://localhost:5000/api/schema"
```

Returns:
```json
{
  "steering_vector_schema": { ... },
  "model_schema": { ... },
  "chat_request_schema": { ... }
}
```

## Tips for Best Results

1. **Start with scale=2.0** and adjust based on results
2. **Use layers 8-25** for most Qwen models (provides good balance)
3. **Test with different vectors** to find what works for your use case
4. **Use debug=true** to see server logs during experimentation
5. **Compare normal_response with steered_response** to measure steering effect
6. **Document your findings** for reproducibility

## Troubleshooting

### "Failed to load model"
- Check model path is correct
- Ensure model is available on HuggingFace or locally
- Check GPU memory is sufficient

### "File not found" for vector
- Verify vector path is correct
- Use absolute paths if relative paths don't work
- Ensure vector file exists

### Response is unchanged
- Try increasing scale (e.g., 3.0 or 5.0)
- Try different target_layers
- Verify vector file is correct format

### Slow response time
- Reduce max_tokens
- Check GPU utilization with nvidia-smi
- Ensure no other GPU processes running

## Comparison with Preset-Based API

| Aspect | Flexible API | Preset-Based |
|--------|-------------|--------------|
| Configuration | Per-request | Fixed preset |
| Flexibility | Maximum | Limited |
| Experimentation | Easy | Cumbersome |
| API Complexity | Simple | Requires preset setup |
| Scalability | Excellent | Limited to presets |
| User Control | Full | Restricted |

This flexible API is ideal for research, experimentation, and applications requiring dynamic steering configuration.
