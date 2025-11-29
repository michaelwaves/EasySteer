"""
FastAPI routes for chat functionality with steering vectors.
Replaces the Flask chat_api.py blueprint with FastAPI + Pydantic.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import time
import json
import os

from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest

# Configure logging
logger = logging.getLogger(__name__)

# Router for chat endpoints
chat_router = APIRouter(prefix="/api", tags=["chat"])

# Store LLM instances (to avoid reloading)
chat_llm_instances = {}

# Preset configurations
preset_configs = {}

# Explicitly map preset keys to their config files
PRESET_CONFIG_PATHS = {
    "happy_mode": "configs/chat/happy_mode.json",
    "chinese": "configs/chat/chinese_mode.json",
    "reject_mode": "configs/chat/reject_mode.json",
    "cat_mode": "configs/chat/cat_mode.json"
}


# ============================================================================
# Pydantic Models
# ============================================================================

class ChatMessage(BaseModel):
    """A single message in chat history"""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request body for chat endpoint"""
    preset: str = Field(
        "happy_mode", description="Preset steering vector mode")
    message: str = Field(..., description="User message")
    history: List[ChatMessage] = Field(
        default_factory=list, description="Chat history for normal response")
    steered_history: List[ChatMessage] = Field(
        default_factory=list, description="Chat history for steered response")
    gpu_devices: str = Field(
        "0", description="GPU device IDs (comma-separated)")
    temperature: float = Field(
        0.8, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(
        512, ge=1, le=4096, description="Maximum tokens to generate")
    repetition_penalty: float = Field(
        1.1, ge=1.0, le=2.0, description="Repetition penalty")


class ChatResponse(BaseModel):
    """Response from chat endpoint"""
    success: bool
    normal_response: str = Field(..., description="Response without steering")
    steered_response: str = Field(...,
                                  description="Response with steering applied")
    preset: str
    error: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================

def load_preset_configs() -> None:
    """Load preset configurations from the explicit paths defined in PRESET_CONFIG_PATHS."""
    global preset_configs
    base_dir = os.path.dirname(__file__)

    for preset_name, config_path_str in PRESET_CONFIG_PATHS.items():
        try:
            config_path = os.path.join(base_dir, config_path_str)
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Store the loaded config
            preset_configs[preset_name] = {
                "vector_path": config["vector"]["path"],
                "scale": config["vector"]["scale"],
                "target_layers": config["vector"]["target_layers"],
                "algorithm": config["vector"]["algorithm"],
                "prefill_trigger_token_ids": config["vector"]["prefill_trigger_token_ids"],
                "generate_trigger_token_ids": config["vector"].get("generate_trigger_token_ids", None),
                "normalize": config["vector"].get("normalize", False),
                "model_path": config["model"]["path"]
            }
            logger.info(
                f"Successfully loaded config for preset: {preset_name}")
        except Exception as e:
            logger.error(
                f"Failed to load config for preset {preset_name}: {str(e)}")


def get_or_create_llm(model_path: str, gpu_devices: str = "0") -> LLM:
    """Get or create an LLM instance"""
    global chat_llm_instances

    key = f"{model_path}_{gpu_devices}"

    if key not in chat_llm_instances:
        try:
            # Set environment variables
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
            os.environ["VLLM_USE_V1"] = "1"

            # Calculate tensor_parallel_size
            gpu_count = len(gpu_devices.split(','))

            # Create LLM instance
            chat_llm_instances[key] = LLM(
                model=model_path,
                enable_steer_vector=True,
                enforce_eager=True,
                tensor_parallel_size=gpu_count
            )
            logger.info(f"Created LLM instance for chat model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to create LLM instance: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create LLM instance: {str(e)}"
            )

    return chat_llm_instances[key]


def get_model_prompt(model_path: str, message: str, history: List[ChatMessage]) -> str:
    """Generate appropriate prompt based on model type and chat history"""
    model_path_lower = model_path.lower()

    # Build history string
    history_str = ""
    for msg in history:
        if msg.role == "user":
            history_str += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            history_str += f"Assistant: {msg.content}\n"

    # Check if model path contains any identifiers to determine model type
    if "qwen" in model_path_lower:
        # Qwen format
        prompt = f"{history_str}<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
    elif "mistral" in model_path_lower or "mixtral" in model_path_lower:
        # Mistral format
        prompt = f"{history_str}[INST] {message} [/INST]\n"
    elif "llama" in model_path_lower:
        # LLaMA format
        prompt = f"{history_str}### Human: {message}\n### Assistant:\n"
    else:
        # Fallback format
        prompt = f"{history_str}User: {message}\nAssistant: "

    return prompt


# ============================================================================
# FastAPI Routes
# ============================================================================

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
    logger.info(
        f"Chat request received: preset={request_data.preset}, message={request_data.message[:50]}...")

    # Check if we have config for this preset
    if request_data.preset not in preset_configs:
        logger.warning(f"No config found for preset: {request_data.preset}")

        # Return dummy responses for unknown preset
        time.sleep(0.5)
        return ChatResponse(
            success=True,
            normal_response=f"This is a normal response to: {request_data.message}",
            steered_response=f"This is a steered response ({request_data.preset}) to: {request_data.message}",
            preset=request_data.preset
        )

    try:
        # Get config for the preset
        config = preset_configs[request_data.preset]
        model_path = config["model_path"]

        # Get or create LLM
        llm = get_or_create_llm(model_path, request_data.gpu_devices)

        # Format prompts
        prompt = get_model_prompt(
            model_path, request_data.message, request_data.history)
        steered_prompt = get_model_prompt(
            model_path, request_data.message, request_data.steered_history)

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request_data.temperature,
            max_tokens=request_data.max_tokens,
            repetition_penalty=request_data.repetition_penalty
        )

        # Create baseline (non-steered) request with scale=0
        baseline_request = SteerVectorRequest(
            steer_vector_name="baseline",
            steer_vector_int_id=1,
            steer_vector_local_path=config["vector_path"],
            scale=0.0,  # Zero scale = no steering
            target_layers=[0],
            algorithm="direct"
        )

        # Create the actual steering vector request
        steer_vector_request = SteerVectorRequest(
            steer_vector_name=f"{request_data.preset}_vector",
            steer_vector_int_id=2,
            steer_vector_local_path=config["vector_path"],
            scale=config["scale"],
            target_layers=config["target_layers"],
            prefill_trigger_tokens=config.get("prefill_trigger_token_ids"),
            generate_trigger_tokens=config.get("generate_trigger_token_ids"),
            algorithm=config["algorithm"],
            normalize=config.get("normalize", False)
        )

        # Generate baseline response
        baseline_output = llm.generate(
            prompt,
            sampling_params,
            steer_vector_request=baseline_request
        )
        normal_response = baseline_output[0].outputs[0].text.strip()

        # Generate steered response
        steered_output = llm.generate(
            steered_prompt,
            sampling_params,
            steer_vector_request=steer_vector_request
        )
        steered_response = steered_output[0].outputs[0].text.strip()

        logger.info(
            f"Chat request completed successfully for preset: {request_data.preset}")

        return ChatResponse(
            success=True,
            normal_response=normal_response,
            steered_response=steered_response,
            preset=request_data.preset
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation error: {str(e)}"
        )


@chat_router.get("/health/chat")
async def health_check():
    """Health check endpoint for chat service"""
    return {
        "status": "healthy",
        "service": "chat",
        "loaded_presets": len(preset_configs)
    }


@chat_router.get("/presets")
async def list_presets():
    """List available chat presets"""
    presets_info = {}
    for preset_name, config in preset_configs.items():
        presets_info[preset_name] = {
            "model": config["model_path"],
            "scale": config["scale"],
            "algorithm": config["algorithm"]
        }
    return {"presets": presets_info}


# Initialize preset configs when module is loaded
load_preset_configs()

if __name__ == "__main__":
    uvicorn.run("chat_fastapi:app", port=8000, host="0.0.0.0", reload=True)
