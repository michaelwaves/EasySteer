"""
FastAPI application for EasySteer frontend.

This is a refactored version of the Flask application using FastAPI + Pydantic
for better type safety, validation, and OpenAPI documentation.

To run:
    uvicorn main_fastapi:app --host 0.0.0.0 --port 5000 --reload
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os

# Import routers
from chat_fastapi_flexible import chat_router
# from inference_fastapi import inference_router  # (if you create this)
# from extraction_fastapi import extraction_router  # (if you create this)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="EasySteer API",
    description="API for text generation with steering vectors",
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


# ============================================================================
# Lifecycle Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize configuration on startup"""
    logger.info("EasySteer API starting up...")
    # load_preset_configs()
    # logger.info("Preset configurations loaded")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("EasySteer API shutting down...")


# ============================================================================
# Root Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "EasySteer API",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "EasySteer API"
    }


# ============================================================================
# Include Routers
# ============================================================================

# Chat endpoint (from chat_fastapi.py)
app.include_router(chat_router)

# Uncomment these when they're created as FastAPI modules
# app.include_router(inference_router)
# app.include_router(extraction_router)


# ============================================================================
# Global Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error"
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
