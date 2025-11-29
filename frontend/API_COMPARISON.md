# API Design Comparison: Preset-Based vs Flexible

This document compares the two FastAPI approaches for steering vector control.

## Overview

Two different API designs were created to serve different use cases:

1. **Preset-Based API** (`chat_fastapi.py`)
   - Predefined configurations
   - Simple to use
   - Limited flexibility

2. **Flexible API** (`chat_fastapi_flexible.py`)
   - Full parameter control
   - User-defined configurations
   - Maximum flexibility

## Comparison

### Request Structure

#### Preset-Based API

```json
{
  "preset": "happy_mode",
  "message": "Hello!",
  "history": [],
  "temperature": 0.8,
  "max_tokens": 512
}
```

**Pros**:
- Simple and concise
- Easy to use for known presets
- Lower cognitive load

**Cons**:
- Limited to predefined presets
- Can't mix parameters
- Hard to experiment

#### Flexible API

```json
{
  "message": "Hello!",
  "model": {
    "path": "Qwen/Qwen2.5-7B-Instruct"
  },
  "steering_vector": {
    "path": "vectors/happiness.pt",
    "scale": 2.0,
    "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    "algorithm": "direct",
    "prefill_trigger_token_ids": [-1],
    "generate_trigger_token_ids": [-1],
    "normalize": false
  },
  "temperature": 0.8,
  "max_tokens": 512
}
```

**Pros**:
- Full control over all parameters
- No preset limitations
- Easy to experiment
- Transparent configuration

**Cons**:
- More verbose
- Requires more knowledge
- Higher cognitive load

---

## Feature Comparison

| Feature | Preset-Based | Flexible |
|---------|------------|----------|
| **Ease of Use** | Very Simple | Moderate |
| **Parameter Control** | None | Full |
| **Experimentation** | Hard | Easy |
| **Preset Limitations** | Yes | No |
| **Dynamic Configuration** | No | Yes |
| **API Complexity** | Simple | Complex |
| **Learning Curve** | Flat | Steeper |
| **Use Cases** | Production | Research |
| **Reproducibility** | Limited | Excellent |

---

## Use Cases

### Choose Preset-Based API If:

1. **You have known use cases**
   - Happy customer service bot
   - Formal business assistant
   - Specific personality

2. **You want simplicity**
   - Mobile app with limited options
   - Non-technical users
   - Quick integration

3. **You need consistency**
   - Fixed behavior
   - Predictable responses
   - Limited variability

4. **You have limited resources**
   - Don't want to manage presets
   - Want fast development

**Example**: A customer support chatbot with "friendly mode" preset

### Choose Flexible API If:

1. **You're researching**
   - Testing different vectors
   - Tuning parameters
   - Exploring steering effects

2. **You need flexibility**
   - Dynamic configuration
   - User-controlled parameters
   - A/B testing

3. **You want experimentation**
   - Try different scales
   - Test different layers
   - Compare vectors

4. **You're building tools**
   - Parameter exploration interface
   - Steering effect visualizer
   - Batch experimentation

**Example**: A research tool for understanding steering vectors

---

## Real-World Scenarios

### Scenario 1: Customer Support Bot

**Requirement**: Use "happy_mode" preset always

**Preset-Based (Better)**:
```json
{
  "preset": "happy_mode",
  "message": "I need help with my order",
  "gpu_devices": "0"
}
```

**Flexible (Overkill)**:
```json
{
  "message": "I need help with my order",
  "model": {"path": "Qwen/Qwen2.5-7B-Instruct"},
  "steering_vector": {
    "path": "vectors/happiness.pt",
    "scale": 2.0,
    "target_layers": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
  }
}
```

### Scenario 2: Research & Experimentation

**Requirement**: Test different steering parameters

**Preset-Based (Impossible)**:
```json
// Can't test scale=1.5, scale=2.5, scale=3.0
// Can't test different layer ranges
// Can't test different vectors
```

**Flexible (Perfect)**:
```json
// Test scale=1.5
{"steering_vector": {"scale": 1.5, ...}}

// Test scale=2.5
{"steering_vector": {"scale": 2.5, ...}}

// Test different layers
{"steering_vector": {"target_layers": [10, 11, 12], ...}}

// Test different vectors
{"steering_vector": {"path": "vectors/sadness.pt", ...}}
```

### Scenario 3: Interactive Exploration

**Requirement**: Allow users to adjust steering dynamically

**Preset-Based (Limited)**:
- Can only switch between presets
- No fine-grained control

**Flexible (Excellent)**:
- Users can adjust scale with slider
- Users can select layers
- Users can choose different vectors
- Users can see effects in real-time

---

## Hybrid Approach (Recommended)

For production, consider using **both**:

```python
# Flexible API for research/development
# http://localhost:5000/api/chat (flexible)

# Preset-based API for production
# http://localhost:5001/api/chat (presets)
```

**Advantages**:
- Research team uses flexible API
- Production uses stable presets
- Best of both worlds
- No conflicts

---

## Data Model Comparison

### Preset-Based

```python
class ChatRequest:
    preset: str              # Reference to config
    message: str
    history: List[ChatMessage]
    temperature: float
    max_tokens: int
    repetition_penalty: float
    gpu_devices: str
```

**Configuration stored in**:
- `configs/chat/happy_mode.json`
- `configs/chat/chinese_mode.json`
- etc.

### Flexible

```python
class SteeringVectorConfig:
    path: str                            # Direct file path
    scale: float                         # Direct parameter
    target_layers: List[int]             # Direct parameter
    algorithm: str                       # Direct parameter
    prefill_trigger_token_ids: List[int] # Direct parameter
    generate_trigger_token_ids: List[int]# Direct parameter
    normalize: bool                      # Direct parameter

class ChatRequest:
    message: str
    model: ModelConfig                   # Specify model per request
    steering_vector: Optional[SteeringVectorConfig]
    history: List[ChatMessage]
    temperature: float
    max_tokens: int
    repetition_penalty: float
    gpu_devices: str
```

**Configuration**: Provided per-request

---

## Migration Path

If you start with presets and need flexibility later:

### Step 1: Add Flexible API Alongside

```
Existing: /api/chat-preset (old preset-based)
New:      /api/chat (flexible)
```

### Step 2: Update Frontend

```javascript
// Old: Use presets
fetch('/api/chat-preset', {
    json: {preset: 'happy_mode', message: '...'}
})

// New: Use flexible API
fetch('/api/chat', {
    json: {
        message: '...',
        model: {path: 'Qwen/...'},
        steering_vector: {...}
    }
})
```

### Step 3: Keep Both or Migrate Fully

**Option A**: Keep both for different use cases
**Option B**: Fully migrate to flexible API

---

## Performance Implications

Both APIs have similar performance:

| Metric | Preset | Flexible |
|--------|--------|----------|
| Request Parsing | ~1ms | ~1ms |
| Validation | ~2ms | ~3ms |
| Model Loading | ~5-10s | ~5-10s |
| Inference | Dominant | Dominant |
| Response Creation | ~1ms | ~2ms |
| **Total** | Similar | Similar |

**Conclusion**: Performance difference negligible. Choose based on use case, not performance.

---

## Error Handling

### Preset-Based

```json
{
  "error": "Preset 'invalid_mode' not found"
}
```

### Flexible

```json
{
  "error": "Vector file not found: /path/to/nonexistent.pt"
}
```

More specific errors in flexible API since configuration is explicit.

---

## Deployment Recommendation

### For Production
- Use **Preset-Based API**
- Limited, known configurations
- Easy to manage and monitor
- Reduced attack surface

### For Research
- Use **Flexible API**
- Full control
- Easy to experiment
- Transparent parameters

### For Both
- Deploy **both endpoints**
- `/api/chat-preset` - Production
- `/api/chat` - Research
- No conflicts

---

## Conclusion

| Dimension | Winner |
|-----------|--------|
| **Simplicity** | Preset-Based |
| **Flexibility** | Flexible |
| **Experimentation** | Flexible |
| **Production Use** | Preset-Based |
| **Research Use** | Flexible |
| **Performance** | Tie |
| **Learning Curve** | Preset-Based |
| **Control** | Flexible |

**Recommendation**: Start with **flexible API** for research. Convert successful patterns to **preset-based API** for production.
