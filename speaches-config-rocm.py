#!/usr/bin/env python3
"""
ROCm-optimized configuration for Speaches
This configuration enables ROCm support for Kokoro TTS and Parakeet STT
"""

import os
from pathlib import Path

# ROCm-specific environment variables
os.environ.update(
    {
        "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "MIOPEN_LOG_LEVEL": "3",
        "HIP_VISIBLE_DEVICES": "0",
    }
)

# Model configurations optimized for ROCm
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "Systran/faster-whisper-large-v3")
KOKORO_MODEL = os.getenv("KOKORO_MODEL", "speaches-ai/Kokoro-82M-v1.0-ONNX")

# ROCm-optimized ONNX Runtime settings
ORT_SETTINGS = {
    "providers": [
        (
            "MIGraphXExecutionProvider",
            {
                "device_id": 0,
                "migraphx_fp16_enable": True,
                "migraphx_int8_enable": False,
            },
        ),
        (
            "ROCMExecutionProvider",
            {
                "device_id": 0,
                "gpu_mem_limit": 4 * 1024 * 1024 * 1024,  # 4GB
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_use_max_workspace": True,
            },
        ),
        "CPUExecutionProvider",
    ],
    "provider_options": {
        "MIGraphXExecutionProvider": {
            "migraphx_fp16_enable": True,
            "migraphx_int8_enable": False,
        },
        "ROCMExecutionProvider": {
            "device_id": 0,
            "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
        },
    },
}

# Cache directories
CACHE_DIR = Path("/home/ubuntu/.cache/huggingface/hub")
MODELS_DIR = Path("/app/models")

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ROCm device detection
def detect_rocm_device():
    """Detect ROCm device and return appropriate settings"""
    try:
        import subprocess

        result = subprocess.run(["rocm-smi", "--showid"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False


# Configure based on ROCm availability
if detect_rocm_device():
    print("🚀 ROCm device detected - enabling GPU acceleration")
    DEVICE = "rocm"
    COMPUTE_TYPE = "float16"
else:
    print("ℹ️  ROCm not detected - using CPU")
    DEVICE = "cpu"
    COMPUTE_TYPE = "float32"

# Model-specific configurations
MODEL_CONFIGS = {
    "kokoro": {
        "model_id": KOKORO_MODEL,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "batch_size": 8 if DEVICE == "rocm" else 4,
    },
    "whisper": {
        "model_id": WHISPER_MODEL,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "beam_size": 5,
    },
    "parakeet": {
        "model_id": "istupakov/parakeet-tdt-0.6b-v3-onnx",
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "batch_size": 8 if DEVICE == "rocm" else 4,
    },
}

# Server settings
HOST = "0.0.0.0"
PORT = 8000
LOG_LEVEL = "INFO"

# Performance settings
MAX_WORKERS = 4
TIMEOUT = 300
