# Version Info

Current Version: **v1.1.0**
Release Date: 2026-03-29
Status: Stable
Description: Updated implementation for a simplified, modular, and automated benchmarking workflow.
- **Improved Workflow**: Consolidated `main.py` task into a multi-precision benchmarking automation.
- **Enhanced Accuracy Evaluation**: Dedicated `compute_perplexity` function to measure the "accuracy tax" of quantization.
- **VRAM Profiling**: Improved GPU memory measurement using `torch.cuda.memory_allocated()`.
- **Memory Management**: Added explicit memory clearance (`gc.collect()` and `torch.cuda.empty_cache()`) between benchmark cycles.
- **Modular Refactoring**: Separated model loading, benchmarking, and evaluation logic.
