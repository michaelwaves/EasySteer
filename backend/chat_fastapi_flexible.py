"""
FastAPI routes for chat functionality with flexible steering vector configuration.
Allows users to specify steering parameters directly in the request instead of using presets.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import os

from vllm import LLM, SamplingParams
from vllm.steer_vectors.request import SteerVectorRequest

# Configure logging
logger = logging.getLogger(__name__)

# Router for chat endpoints
chat_router = APIRouter(prefix="/api", tags=["chat"])

# Store LLM instances (to avoid reloading)
llm_instances = {}


# ============================================================================
# Pydantic Models
# ============================================================================

class ChatMessage(BaseModel):
    """A single message in chat history"""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class SteeringVectorConfig(BaseModel):
    """Configuration for a steering vector"""
    path: str = Field(...,
                      description="Local path to steering vector file (.pt or .gguf)")
    scale: float = Field(1.0, description="Scale factor for the vector")
    target_layers: List[int] = Field(
        default_factory=lambda: list(range(8, 26)),
        description="Layer indices to apply steering to"
    )
    algorithm: str = Field(
        "direct", description="Algorithm: 'direct' or 'loreft'")
    prefill_trigger_token_ids: Optional[List[int]] = Field(
        default=[-1],
        description="Token IDs to trigger on during prefill phase (-1 = all tokens)"
    )
    generate_trigger_token_ids: Optional[List[int]] = Field(
        default=[-1],
        description="Token IDs to trigger on during generation phase (-1 = all tokens)"
    )
    normalize: bool = Field(
        False, description="Whether to normalize the vector")


class ModelConfig(BaseModel):
    """Configuration for the language model"""
    path: str = Field(...,
                      description="Model path (HuggingFace model ID or local path)")


class ChatRequest(BaseModel):
    """Request body for chat endpoint with flexible steering configuration"""
    message: str = Field(..., description="User message")

    # Chat history
    history: List[ChatMessage] = Field(
        default_factory=list,
        description="Chat history for normal response"
    )
    steered_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Chat history for steered response (uses history if not provided)"
    )

    # Model configuration
    model: ModelConfig = Field(..., description="Language model configuration")

    # Steering vector configuration (optional)
    steering_vector: Optional[SteeringVectorConfig] = Field(
        default=None,
        description="Steering vector configuration. If None, generates unsteered response only"
    )

    # Generation parameters
    gpu_devices: str = Field(
        "0", description="GPU device IDs (comma-separated)")
    temperature: float = Field(
        0.8, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(
        512, ge=1, le=4096, description="Maximum tokens to generate")
    repetition_penalty: float = Field(
        1.1, ge=1.0, le=2.0, description="Repetition penalty")

    # Additional options
    debug: bool = Field(False, description="Enable debug output")


class ChatResponse(BaseModel):
    """Response from chat endpoint"""
    success: bool
    normal_response: str = Field(..., description="Response without steering")
    steered_response: Optional[str] = Field(
        None,
        description="Response with steering applied (None if no steering config provided)"
    )
    config: dict = Field(..., description="Echo of the configuration used")
    error: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================

def get_or_create_llm(model_path: str, gpu_devices: str = "0") -> LLM:
    """Get or create an LLM instance"""
    global llm_instances

    key = f"{model_path}_{gpu_devices}"

    if key not in llm_instances:
        try:
            # Set environment variables
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
            os.environ["VLLM_USE_V1"] = "1"

            # Calculate tensor_parallel_size
            gpu_count = len(gpu_devices.split(','))

            # Create LLM instance
            llm_instances[key] = LLM(
                model=model_path,
                enable_steer_vector=True,
                enforce_eager=True,
                tensor_parallel_size=gpu_count
            )
            logger.info(f"Created LLM instance for model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to create LLM instance: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {str(e)}"
            )

    return llm_instances[key]


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
    Chat API endpoint with flexible steering vector configuration.

    Users provide:
    1. Model path and message
    2. Optional steering vector configuration with all parameters
    3. Generation parameters

    The endpoint:
    1. Loads the model
    2. Generates a baseline response (always)
    3. Generates a steered response (if steering config provided)
    4. Returns both responses for comparison
    """
    logger.info(
        f"Chat request received: model={request_data.model.path}, "
        f"steering={'yes' if request_data.steering_vector else 'no'}"
    )

    try:
        # Get or create LLM instance
        llm = get_or_create_llm(request_data.model.path,
                                request_data.gpu_devices)

        # Use steered_history if provided, otherwise use history
        steered_history = request_data.steered_history or request_data.history

        # Format prompts
        prompt = get_model_prompt(
            request_data.model.path, request_data.message, request_data.history)
        steered_prompt = get_model_prompt(
            request_data.model.path, request_data.message, steered_history)

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=request_data.temperature,
            max_tokens=request_data.max_tokens,
            repetition_penalty=request_data.repetition_penalty
        )

        # Create baseline (non-steered) request
        baseline_request = SteerVectorRequest(
            steer_vector_name="baseline",
            steer_vector_int_id=1,
            steer_vector_local_path=request_data.steering_vector.path,  # Dummy path
            scale=0.0,  # Zero scale = no steering
            target_layers=[0],
            algorithm="direct"
        )

        # Generate baseline response
        baseline_output = llm.generate(
            prompt,
            sampling_params,
            steer_vector_request=baseline_request
        )
        normal_response = baseline_output[0].outputs[0].text.strip()

        # Generate steered response if steering config provided
        steered_response = None
        if request_data.steering_vector:
            steer_vector_config = request_data.steering_vector

            steer_vector_request = SteerVectorRequest(
                steer_vector_name="user_steering",
                steer_vector_int_id=2,
                steer_vector_local_path=steer_vector_config.path,
                scale=steer_vector_config.scale,
                target_layers=steer_vector_config.target_layers,
                prefill_trigger_tokens=steer_vector_config.prefill_trigger_token_ids,
                generate_trigger_tokens=steer_vector_config.generate_trigger_token_ids,
                algorithm=steer_vector_config.algorithm,
                normalize=steer_vector_config.normalize,
                debug=request_data.debug
            )

            steered_output = llm.generate(
                steered_prompt,
                sampling_params,
                steer_vector_request=steer_vector_request
            )
            steered_response = steered_output[0].outputs[0].text.strip()

            logger.info(
                f"Chat request completed with steering: "
                f"scale={steer_vector_config.scale}, layers={len(steer_vector_config.target_layers)}"
            )
        else:
            logger.info("Chat request completed without steering")

        # Build config echo
        config = {
            "model": request_data.model.path,
            "temperature": request_data.temperature,
            "max_tokens": request_data.max_tokens,
            "repetition_penalty": request_data.repetition_penalty,
        }
        if request_data.steering_vector:
            config["steering"] = {
                "path": request_data.steering_vector.path,
                "scale": request_data.steering_vector.scale,
                "target_layers": request_data.steering_vector.target_layers,
                "algorithm": request_data.steering_vector.algorithm,
                "normalize": request_data.steering_vector.normalize,
            }

        return ChatResponse(
            success=True,
            normal_response=normal_response,
            steered_response=steered_response,
            config=config
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation error: {str(e)}"
        )


@chat_router.get("/health")
async def health_check():
    """Health check endpoint for chat service"""
    return {
        "status": "healthy",
        "service": "chat",
        "loaded_models": len(llm_instances)
    }


@chat_router.post("/schema")
async def get_schema():
    """
    Get the JSON schema for steering vector configuration.
    Useful for frontend form generation.
    """
    return {
        "steering_vector_schema": SteeringVectorConfig.model_json_schema(),
        "model_schema": ModelConfig.model_json_schema(),
        "chat_request_schema": ChatRequest.model_json_schema(),
    }
