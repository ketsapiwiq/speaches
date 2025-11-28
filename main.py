from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
import tempfile
import os
import logging
from typing import Optional
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Speaches ROCm API", version="1.0.0")

# Global model instances
whisper_model = None
parakeet_model = None


def load_models():
    """Load parakeet model only (CPU)"""
    global whisper_model, parakeet_model

    try:
        # Use parakeet model only for CPU
        logger.info("Loading parakeet model for CPU")
        parakeet_model_name = os.getenv("PARAKEET_MODEL", "istupakov/parakeet-tdt-0.6b-v3-onnx")
        logger.info(f"Loading parakeet model: {parakeet_model_name}")

        from speaches.config import OrtOptions
        from speaches.executors.parakeet import ParakeetModelManager

        ort_opts = OrtOptions()
        ort_opts.exclude_providers = ["CUDAExecutionProvider", "TensorrtExecutionProvider"]
        parakeet_model = ParakeetModelManager(ttl=3600, ort_opts=ort_opts)._load_fn(parakeet_model_name)

        logger.info("Parakeet model loaded successfully")

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": parakeet_model is not None}


@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "models": [
            {"id": "parakeet-tdt-0.6b-v3", "object": "model"},
        ]
    }


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...), model: str = "parakeet-1", language: str = "fr", response_format: str = "json"
):
    """Transcribe audio file using parakeet model"""

    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    if parakeet_model is None:
        raise HTTPException(status_code=500, detail="Parakeet model not loaded")

    try:
        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        logger.info(f"Transcribing with parakeet, language: {language}")

        # Transcribe with parakeet
        segments = parakeet_model.transcribe(tmp_file_path, language=language)
        text = " ".join(segment.text.strip() for segment in segments)

        # Clean up
        os.unlink(tmp_file_path)

        if response_format == "text":
            return Response(content=text, media_type="text/plain")
        else:
            return {"text": text}

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/speech")
async def generate_speech():
    """Generate speech (placeholder for TTS)"""
    raise HTTPException(status_code=501, detail="TTS not implemented in this service")


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
