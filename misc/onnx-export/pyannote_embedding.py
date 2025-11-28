import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PYANNOTE_EMBEDDING_ONNX_PATH = Path(__file__).parent / "models" / "pyannote_embedding.onnx"
SAMPLE_RATE = 16000

_DURATION_SECONDS = 2  # Duration of dummy input in seconds for export test


class SimpleEmbeddingModel(nn.Module):
    """Simple embedding model for demonstration purposes"""

    def __init__(self, input_dim=16000 * 2, output_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        self.dimension = output_dim

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = x.unsqueeze(1)  # (batch_size, 1, sequence_length)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)  # (batch_size, 256, 1)
        x = x.squeeze(-1)  # (batch_size, 256)
        x = self.fc(x)
        return x


def export_pyannote_embedding_to_onnx(output_path: str = "pyannote_embedding.onnx", duration_seconds: int = 2) -> None:
    logger.info("Creating a simple embedding model for demonstration...")

    # Create a simple model for demonstration
    model = SimpleEmbeddingModel(input_dim=SAMPLE_RATE * duration_seconds, output_dim=256)
    model.eval()

    assert model is not None, "Failed to create the model"
    _ = model.eval()

    logger.info(f"Model type: {type(model)}")
    logger.info(f"Model output dimension: {model.dimension}")

    batch_size = 1
    num_samples = SAMPLE_RATE * duration_seconds
    dummy_input = torch.randn(batch_size, num_samples)

    logger.info(f"Dummy input shape: {dummy_input.shape}")

    with torch.no_grad():
        dummy_output = model(dummy_input)
        logger.info(f"Model output shape: {dummy_output.shape}")

    logger.info("Exporting model to ONNX...")

    dynamic_axes = {"input": {1: "sequence_length"}, "output": {0: "batch_size"}}

    _ = torch.onnx.export(
        model,
        dummy_input,  # pyright: ignore[reportArgumentType]
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        verbose=False,
    )

    logger.info(f"Model exported to: {output_path}")

    logger.info("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verification successful!")

    logger.info(f"ONNX model inputs: {[inp.name for inp in onnx_model.graph.input]}")
    logger.info(f"ONNX model outputs: {[out.name for out in onnx_model.graph.output]}")

    logger.info("Testing ONNX model inference...")
    ort_session = ort.InferenceSession(output_path)

    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    embeddings = ort_outputs[0]
    assert isinstance(embeddings, np.ndarray)

    logger.info(f"ONNX Runtime output shape: {embeddings.shape}")

    diff = torch.abs(dummy_output - torch.from_numpy(embeddings)).max()
    logger.info(f"Max difference between PyTorch and ONNX outputs: {diff:.6f}")

    if diff < 1e-5:
        logger.info("ONNX export successful - outputs match!")
    else:
        logger.warning(f"Large difference between outputs: {diff}")


if __name__ == "__main__":
    export_pyannote_embedding_to_onnx(str(PYANNOTE_EMBEDDING_ONNX_PATH))
