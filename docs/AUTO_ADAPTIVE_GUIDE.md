# 🎯 Auto-Adaptive System Guide

## Overview

Open-Sable automatically detects your device specifications and selects the optimal AI model and settings. No manual configuration needed!

## How It Works

### 1. System Detection

On startup, Open-Sable analyzes:
- **RAM**: Total and available memory
- **CPU**: Cores and frequency
- **GPU**: Availability and VRAM (if present)
- **Storage**: Free disk space
- **OS**: Linux, macOS, or Windows

### 2. Device Tier Classification

Based on specs, your device is categorized:

| Tier | Requirements | Best For |
|------|-------------|----------|
| **High-End** | 16GB+ RAM, GPU | Large models (70B) |
| **Mid-Range** | 8-16GB RAM or GPU | Balanced models (8B-13B) |
| **Low-End** | 4-8GB RAM | Lighter models (7B) |
| **Minimal** | <4GB RAM | Tiny models (1-2B) |

### 3. Model Selection

Open-Sable automatically picks the best available model:

**High-End Devices:**
- llama3.1:70b (best quality)
- mixtral:8x7b
- llama3.1:13b

**Mid-Range Devices:**
- llama3.1:8b (balanced)
- mistral:7b
- phi-3:medium

**Low-End Devices:**
- llama3.1:7b (lighter)
- phi-3:mini
- tinyllama:1.1b

**Minimal Devices:**
- tinyllama:1.1b (fastest)
- phi-2:2.7b

### 4. Performance Optimization

Auto-adjusts:
- Batch size for processing
- Number of worker threads
- GPU usage (if available)
- Memory limits

## Check Your System

Run the system checker:

```bash
python3 check_system.py
```

Output example:
```
╔═══════════════════════════════════════════╗
║   Open-Sable System Detector               ║
╚═══════════════════════════════════════════╝

System Specifications
┌──────────────┬─────────────┐
│ Component    │ Value       │
├──────────────┼─────────────┤
│ RAM          │ 16.0 GB     │
│ CPU Cores    │ 8           │
│ CPU Freq     │ 3.20 GHz    │
│ GPU          │ Yes         │
│ GPU Memory   │ 8.0 GB      │
│ Free Storage │ 250.0 GB    │
│ OS           │ Linux       │
└──────────────┴─────────────┘

Device Tier: MID-RANGE

Recommended Models:
  1. llama3.1:8b (min RAM: 8GB)
  2. mistral:7b (min RAM: 8GB)
  3. phi-3:medium (min RAM: 8GB)

Recommended Configuration
┌──────────────┬─────────────────┐
│ Setting      │ Value           │
├──────────────┼─────────────────┤
│ Model        │ llama3.1:8b     │
│ Batch Size   │ 5               │
│ Use GPU      │ Yes             │
│ Max Workers  │ 8               │
└──────────────┴─────────────────┘
```

## Configuration

### Automatic (Recommended)

Enable in `.env`:
```bash
AUTO_SELECT_MODEL=true
```

Open-Sable will always use the best model for your device.

### Manual Override

Disable auto-selection:
```bash
AUTO_SELECT_MODEL=false
DEFAULT_MODEL=llama3.1:8b
```

### Hybrid Approach

Let auto-selection suggest, but override for specific cases:

```python
# In your code
from core.system_detector import auto_configure_system

config = auto_configure_system()
print(f"Recommended: {config['recommended_model']}")

# Override if needed
config['recommended_model'] = 'custom-model'
```

## Resource Monitoring

### Real-Time Usage

Check current resource usage:

```python
from core.system_detector import ResourceMonitor

usage = ResourceMonitor.get_current_usage()
print(f"RAM: {usage['ram_used_percent']}%")
print(f"CPU: {usage['cpu_percent']}%")
```

### Auto-Throttling

Open-Sable automatically throttles when resources are low:
- Reduces batch size
- Pauses non-critical tasks
- Waits for resources to free up

Check if throttling:
```python
if ResourceMonitor.should_throttle():
    print("System under heavy load, throttling...")
```

## Model Requirements

Each model has minimum requirements:

| Model | Min RAM | Recommended RAM | VRAM (GPU) |
|-------|---------|----------------|------------|
| 70B | 64GB | 128GB | 40GB |
| 34B | 32GB | 64GB | 20GB |
| 13B | 16GB | 32GB | 8GB |
| 8B | 8GB | 16GB | 6GB |
| 7B | 8GB | 16GB | 6GB |
| 3B | 4GB | 8GB | 3GB |
| 1B | 2GB | 4GB | 2GB |

## GPU Acceleration

If you have a GPU, Open-Sable automatically uses it:

```bash
# Check GPU in system info
python3 check_system.py

# Look for:
# GPU Available: Yes
# GPU Memory: 8.0 GB
```

Supported GPUs:
- NVIDIA (CUDA)
- AMD (ROCm)
- Apple Silicon (Metal)

## Performance Tips

### Low-End Devices

```bash
# Use smallest model
DEFAULT_MODEL=tinyllama:1.1b

# Reduce batch size
# (Automatically handled, but you can force)
```

### High-End Devices

```bash
# Use largest model
DEFAULT_MODEL=llama3.1:70b

# Enable all cores
# (Automatically detected)
```

### Slow Performance?

1. **Check system:** `python3 check_system.py`
2. **View current usage:** Check RAM/CPU in output
3. **Try smaller model:** Edit DEFAULT_MODEL in .env
4. **Close other apps:** Free up RAM
5. **Check disk space:** Need room for model files

## Switching Models

### Pull New Model

```bash
ollama pull llama3.1:13b
```

Restart Open-Sable - it will auto-detect the new model.

### Force Specific Model

```bash
# .env
AUTO_SELECT_MODEL=false
DEFAULT_MODEL=llama3.1:13b
```

### Check Available Models

```bash
ollama list
```

## Advanced Configuration

### Custom Tier Definitions

Edit `core/system_detector.py`:

```python
@staticmethod
def get_device_tier(specs: DeviceSpecs) -> str:
    # Customize tiers
    if specs.ram_gb >= 32:  # Your custom threshold
        return "high-end"
    # ...
```

### Custom Model Recommendations

Edit `MODEL_RECOMMENDATIONS` dict:

```python
MODEL_RECOMMENDATIONS = {
    "high-end": ["your-custom-model", "llama3.1:70b"],
    # ...
}
```

### Disable Auto-Selection

```python
# In core/llm.py
if False:  # Disable auto-selection
    auto_config = auto_configure_system()
```

## Troubleshooting

### "Model not found"
- Pull the model: `ollama pull <model-name>`
- Check available: `ollama list`

### "Out of memory"
- System selected too large a model
- Manually set smaller: `DEFAULT_MODEL=tinyllama:1.1b`
- Close other applications

### "GPU not detected"
- Install GPU drivers
- Install CUDA/ROCm
- Restart Open-Sable

### Auto-selection wrong
- Run: `python3 check_system.py`
- Verify specs are correct
- Override with manual selection

## API Reference

### SystemDetector

```python
from core.system_detector import SystemDetector

specs = SystemDetector.detect()
print(f"RAM: {specs.ram_gb}GB")

tier = SystemDetector.get_device_tier(specs)
print(f"Tier: {tier}")
```

### ModelSelector

```python
from core.system_detector import ModelSelector

model = ModelSelector.select_model(specs, available_models)
print(f"Best model: {model}")

can_run = ModelSelector.can_run_model(specs, "llama3.1:70b")
print(f"Can run 70B: {can_run}")
```

### ResourceMonitor

```python
from core.system_detector import ResourceMonitor

usage = ResourceMonitor.get_current_usage()
should_wait = ResourceMonitor.should_throttle()
batch_size = ResourceMonitor.get_optimal_batch_size(specs)
```

---

**Let Open-Sable optimize itself for your hardware! 🚀**
