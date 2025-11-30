# Flexible Steering Vector API - Usage Examples

Quick reference for using the flexible steering vector API.

## Quick Start

### 1. Basic Chat (No Steering)

```bash
curl -X POST "http://localhost:5000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello!",
    "model": {"path": "Qwen/Qwen2.5-7B-Instruct"}
  }'
```

Response:
```json
{
  "success": true,
  "normal_response": "Hi there! How can I help you today?",
  "steered_response": null,
  "config": {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "temperature": 0.8,
    "max_tokens": 512,
    "repetition_penalty": 1.1
  }
}
```

### 2. Chat with Happiness Steering

```bash
curl -X POST "http://localhost:5000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How are you?",
    "model": {"path": "Qwen/Qwen2.5-7B-Instruct"},
    "steering_vector": {
      "path": "vectors/happiness.pt",
      "scale": 2.0,
      "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    }
  }'
```

Response:
```json
{
  "success": true,
  "normal_response": "I'm doing fine, thank you.",
  "steered_response": "I'm absolutely wonderful! Thanks so much for asking! 😊 I'm feeling amazing today!",
  "config": {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "temperature": 0.8,
    "max_tokens": 512,
    "repetition_penalty": 1.1,
    "steering": {
      "path": "vectors/happiness.pt",
      "scale": 2.0,
      "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
      "algorithm": "direct",
      "normalize": false
    }
  }
}
```

## Language Models

### Qwen2.5-7B-Instruct

```json
{
  "message": "What is AI?",
  "model": {"path": "Qwen/Qwen2.5-7B-Instruct"}
}
```

### Qwen2.5-1.5B (Smaller, Faster)

```json
{
  "message": "What is AI?",
  "model": {"path": "Qwen/Qwen2.5-1.5B-Instruct"}
}
```

### LLaMA 2

```json
{
  "message": "What is AI?",
  "model": {"path": "meta-llama/Llama-2-7b-chat-hf"}
}
```

### Local Model Path

```json
{
  "message": "What is AI?",
  "model": {"path": "/home/user/models/my_model"}
}
```

## Steering Vectors

### Happiness Steering

```json
{
  "steering_vector": {
    "path": "vectors/happiness.pt",
    "scale": 2.0,
    "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
  }
}
```

### Sadness Steering

```json
{
  "steering_vector": {
    "path": "vectors/sadness.pt",
    "scale": 2.0,
    "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
  }
}
```

### Multiple Vector Formats

GGUF format:
```json
{
  "steering_vector": {
    "path": "vectors/diffmean.gguf",
    "scale": 2.0
  }
}
```

PT format:
```json
{
  "steering_vector": {
    "path": "vectors/happiness_response_avg_diff.pt",
    "scale": 2.0
  }
}
```

## Scale Parameters

### Subtle Effect (scale=0.5)

```json
{
  "steering_vector": {
    "path": "vectors/happiness.pt",
    "scale": 0.5,
    "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
  }
}
```

### Moderate Effect (scale=1.0-1.5)

```json
{
  "steering_vector": {
    "path": "vectors/happiness.pt",
    "scale": 1.2,
    "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
  }
}
```

### Strong Effect (scale=2.0-3.0)

```json
{
  "steering_vector": {
    "path": "vectors/happiness.pt",
    "scale": 2.5,
    "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
  }
}
```

### Very Strong Effect (scale=5.0+)

```json
{
  "steering_vector": {
    "path": "vectors/happiness.pt",
    "scale": 5.0,
    "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
  }
}
```

## Layer Configurations

### Early Layers (Lower-level semantics)

```json
{
  "steering_vector": {
    "path": "vectors/happiness.pt",
    "scale": 2.0,
    "target_layers": [5, 6, 7, 8, 9, 10]
  }
}
```

### Middle Layers (Mid-level semantics)

```json
{
  "steering_vector": {
    "path": "vectors/happiness.pt",
    "scale": 2.0,
    "target_layers": [12, 13, 14, 15, 16, 17, 18, 19]
  }
}
```

### Late Layers (Style/output)

```json
{
  "steering_vector": {
    "path": "vectors/happiness.pt",
    "scale": 2.0,
    "target_layers": [20, 21, 22, 23, 24, 25, 26, 27]
  }
}
```

### All Layers

```json
{
  "steering_vector": {
    "path": "vectors/happiness.pt",
    "scale": 2.0,
    "target_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
  }
}
```

## Chat History

### With History

```json
{
  "message": "What was I asking about before?",
  "history": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of AI..."},
    {"role": "user", "content": "Can you give me examples?"},
    {"role": "assistant", "content": "Sure! Examples include image recognition..."}
  ],
  "model": {"path": "Qwen/Qwen2.5-7B-Instruct"}
}
```

### With Different Steered History

```json
{
  "message": "What was I asking about?",
  "history": [
    {"role": "user", "content": "What is ML?"},
    {"role": "assistant", "content": "It's a subset of AI..."}
  ],
  "steered_history": [
    {"role": "user", "content": "What is ML?"},
    {"role": "assistant", "content": "Oh what an AMAZING question! ML is absolutely fantastic..."}
  ],
  "steering_vector": {
    "path": "vectors/happiness.pt",
    "scale": 2.0,
    "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
  },
  "model": {"path": "Qwen/Qwen2.5-7B-Instruct"}
}
```

## Generation Parameters

### Deterministic (Low Temperature)

```json
{
  "temperature": 0.2,
  "repetition_penalty": 1.05
}
```

### Balanced (Medium Temperature)

```json
{
  "temperature": 0.8,
  "repetition_penalty": 1.1
}
```

### Creative (High Temperature)

```json
{
  "temperature": 1.5,
  "repetition_penalty": 1.2
}
```

### Very Creative (Very High Temperature)

```json
{
  "temperature": 2.0,
  "repetition_penalty": 1.3
}
```

## Complete Examples

### Example 1: Happiness Steering with History

```bash
curl -X POST "http://localhost:5000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How did I feel yesterday?",
    "history": [
      {
        "role": "user",
        "content": "I had a bad day today"
      },
      {
        "role": "assistant",
        "content": "I am sorry to hear that. What happened?"
      }
    ],
    "model": {"path": "Qwen/Qwen2.5-7B-Instruct"},
    "steering_vector": {
      "path": "vectors/happiness.pt",
      "scale": 2.0,
      "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    },
    "temperature": 0.8,
    "max_tokens": 256
  }'
```

### Example 2: Experiment with Different Scales

```bash
for scale in 0.5 1.0 1.5 2.0 2.5 3.0; do
  echo "Testing scale=$scale"
  curl -X POST "http://localhost:5000/api/chat" \
    -H "Content-Type: application/json" \
    -d "{
      \"message\": \"Tell me about your day\",
      \"model\": {\"path\": \"Qwen/Qwen2.5-7B-Instruct\"},
      \"steering_vector\": {
        \"path\": \"vectors/happiness.pt\",
        \"scale\": $scale,
        \"target_layers\": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
      }
    }" | jq .
done
```

### Example 3: Python - Compare Multiple Vectors

```python
import requests

vectors = {
    "happiness": "vectors/happiness.pt",
    "sadness": "vectors/sadness.pt",
    "curiosity": "vectors/curiosity.pt"
}

message = "How do you feel?"

for name, path in vectors.items():
    response = requests.post(
        "http://localhost:5000/api/chat",
        json={
            "message": message,
            "model": {"path": "Qwen/Qwen2.5-7B-Instruct"},
            "steering_vector": {
                "path": path,
                "scale": 2.0,
                "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
            }
        }
    )
    data = response.json()
    print(f"\n[{name.upper()}]")
    print(f"Baseline: {data['normal_response'][:80]}...")
    print(f"Steered: {data['steered_response'][:80]}...")
```

### Example 4: JavaScript - Interactive App

```javascript
async function testSteering() {
    const formData = {
        message: document.getElementById('message').value,
        model: {
            path: document.getElementById('model').value || "Qwen/Qwen2.5-7B-Instruct"
        },
        temperature: parseFloat(document.getElementById('temperature').value) || 0.8,
        max_tokens: parseInt(document.getElementById('max_tokens').value) || 512
    };

    // Add steering if enabled
    if (document.getElementById('use_steering').checked) {
        formData.steering_vector = {
            path: document.getElementById('vector_path').value,
            scale: parseFloat(document.getElementById('scale').value) || 2.0,
            target_layers: JSON.parse(document.getElementById('layers').value || "[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]")
        };
    }

    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(formData)
    });

    const data = await response.json();

    document.getElementById('normal_output').textContent = data.normal_response;
    if (data.steered_response) {
        document.getElementById('steered_output').textContent = data.steered_response;
    }
}
```

## Tips

1. **Always test baseline first** (no steering) to see original response
2. **Start with scale=2.0** and adjust up/down based on effect
3. **Use specific layers** for finer control of behavior
4. **Compare outputs** to understand steering effect
5. **Use debug=true** for troubleshooting
6. **Save working configs** for reproducibility

## API Discovery

Get the schema for request validation:

```bash
curl -X POST "http://localhost:5000/api/schema"
```

This returns the full JSON schema for all request models, useful for:
- Form generation
- Request validation
- Documentation
- IDE integration
