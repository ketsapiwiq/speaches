# Speaches Agent Notes

## ROCm Compatibility Issues

### CTranslate2 + ROCm
- CTranslate2 (used by faster-whisper) is **incompatible with AMD ROCm**
- This causes crashes and import errors on AMD GPU systems
- Workaround: Force CPU mode or avoid CTranslate2 entirely

### Transformers + ROCm  
- PyTorch Transformers library also has **ROCm compatibility issues**
- Can cause GPU-related errors even when trying to run on CPU
- Workaround: Explicitly set CPU device and avoid GPU initialization

### Recommended Setup for AMD ROCm Systems
1. **Use CPU-only mode** for all inference
2. **Set environment variables**:
   ```bash
   export CUDA_VISIBLE_DEVICES=""  # Disable GPU detection
   export torch_cuda_arch_list=""  # Disable CUDA arch detection
   ```
3. **Prefer ONNX models** over PyTorch when possible (better CPU performance)
4. **Use INT8 quantization** for better CPU performance

### Model Recommendations for CPU
- **Parakeet ONNX**: Best CPU performance, ROCm-safe
- **Distil-Whisper (CTranslate2)**: Good CPU performance if CTranslate2 works
- **OpenAI Whisper (Transformers)**: Fallback option, slower but compatible

### Testing Commands
```bash
# Test CPU-only configuration
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CPU works: {torch.randn(1,1).sum().item() > 0')"
```