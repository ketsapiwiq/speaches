from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from faster_whisper import WhisperModel
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
    """Load whisper and parakeet models"""
    global whisper_model, parakeet_model

    try:
        # Check ROCm/CUDA availability
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            logger.info(f"Using GPU: {device_name}")
            logger.info(f"Available GPU devices: {device_count}")
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if hasattr(torch.version, "hip"):
                logger.info(f"ROCm version: {torch.version.hip}")
        else:
            device = "cpu"
            compute_type = "int8"
            logger.warning("GPU not available, using CPU")
            logger.info(f"PyTorch version: {torch.__version__}")

        # Load whisper model
        whisper_model_name = os.getenv("WHISPER_MODEL", "large-v3")
        logger.info(f"Loading whisper model: {whisper_model_name}")
        whisper_model = WhisperModel(
            # FasterWhisper not working on ROCm
            # whisper_model_name, device=device, compute_type=compute_type            
            whisper_model_name, device="cpu", compute_type="int8"
        )

        # Load parakeet model
        parakeet_model_name = os.getenv("PARAKEET_MODEL", "istupakov/parakeet-tdt-0.6b-v3-onnx")
        logger.info(f"Loading parakeet model: {parakeet_model_name}")
        # Load parakeet model
        from speaches.config import OrtOptions
        from speaches.executors.parakeet import ParakeetModelManager
        ort_opts = OrtOptions()
        ort_opts.exclude_providers = ['CUDAExecutionProvider', 'TensorrtExecutionProvider']
        parakeet_model = ParakeetModelManager(ttl=3600, ort_opts=ort_opts)._load_fn(parakeet_model_name)

        logger.info("All models loaded successfully")

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
    return {"status": "healthy", "models_loaded": whisper_model is not None and parakeet_model is not None}

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "models": [
            {"id": "whisper-large-v3", "object": "model"},
            {"id": "istupakov/parakeet-tdt-0.6b-v3-onnx", "object": "model"},
        ]
    }

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = "whisper-1",
    language: str = "fr",
    response_format: str = "json"
):
    """Transcribe audio file"""
    
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    # Determine which model to use
    if "parakeet" in model.lower():
        current_model = parakeet_model
        model_name = "parakeet-tdt-1.1b"
    else:
        current_model = whisper_model
        model_name = "whisper-large-v3"
    
    if current_model is None:
        raise HTTPException(status_code=500, detail=f"Model {model_name} not loaded")
    
    try:
        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        logger.info(f"Transcribing with {model_name}, language: {language}")
        
        # Transcribe
        segments, info = current_model.transcribe(
            tmp_file_path,
            language=language,
            word_timestamps=False
        )
        
        # Combine segments
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
    uvicorn.run(app, host="0.0.0.0", port=8000)