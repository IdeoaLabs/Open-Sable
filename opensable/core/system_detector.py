"""
System detection and auto-configuration for Open-Sable
Detects device specs and selects optimal model
"""

import logging
import psutil
from typing import Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DeviceSpecs:
    """Device hardware specifications"""

    ram_gb: float
    cpu_cores: int
    cpu_freq_ghz: float
    gpu_available: bool
    gpu_memory_gb: float
    storage_free_gb: float
    system: str  # Linux, Darwin, Windows


class SystemDetector:
    """Detect system resources and capabilities"""

    @staticmethod
    def detect() -> DeviceSpecs:
        """Detect all system specifications"""
        logger.info("Detecting system specifications...")

        # RAM
        ram = psutil.virtual_memory()
        ram_gb = ram.total / (1024**3)

        # CPU
        cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        cpu_freq_ghz = cpu_freq.max / 1000 if cpu_freq else 2.0

        # GPU
        gpu_available, gpu_memory_gb = SystemDetector._detect_gpu()

        # Storage
        disk = psutil.disk_usage("/")
        storage_free_gb = disk.free / (1024**3)

        # System
        import platform

        system = platform.system()

        specs = DeviceSpecs(
            ram_gb=ram_gb,
            cpu_cores=cpu_cores,
            cpu_freq_ghz=cpu_freq_ghz,
            gpu_available=gpu_available,
            gpu_memory_gb=gpu_memory_gb,
            storage_free_gb=storage_free_gb,
            system=system,
        )

        logger.info(f"System specs: {specs}")
        return specs

    @staticmethod
    def _detect_gpu() -> tuple[bool, float]:
        """Detect GPU availability and memory"""
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return True, gpu.memoryTotal / 1024  # Convert MB to GB
        except Exception as e:
            logger.debug(f"GPU detection failed: {e}")

        # Try CUDA
        try:
            import torch

            if torch.cuda.is_available():
                mem = torch.cuda.get_device_properties(0).total_memory
                return True, mem / (1024**3)
        except Exception:
            pass

        return False, 0.0

    @staticmethod
    def get_device_tier(specs: DeviceSpecs) -> str:
        """Categorize device into performance tier"""
        # Tier 1: High-end (16GB+ RAM, GPU)
        if specs.ram_gb >= 16 and specs.gpu_available:
            return "high-end"

        # Tier 2: Mid-range (8-16GB RAM, or GPU)
        elif specs.ram_gb >= 8 or specs.gpu_available:
            return "mid-range"

        # Tier 3: Low-end (4-8GB RAM)
        elif specs.ram_gb >= 4:
            return "low-end"

        # Tier 4: Minimal (<4GB RAM)
        else:
            return "minimal"


class ModelSelector:
    """Automatically select optimal LLM model based on device specs"""

    # Model recommendations by tier
    MODEL_RECOMMENDATIONS = {
        "high-end": [
            "llama3.1:70b",  # Best quality
            "mixtral:8x7b",
            "llama3.1:13b",
        ],
        "mid-range": [
            "llama3.1:8b",  # Balanced
            "mistral:7b",
            "phi-3:medium",
        ],
        "low-end": [
            "llama3.1:7b",  # Lighter
            "phi-3:mini",
            "tinyllama:1.1b",
        ],
        "minimal": [
            "tinyllama:1.1b",  # Minimal resource usage
            "phi-2:2.7b",
        ],
    }

    @staticmethod
    def select_model(specs: DeviceSpecs, available_models: list = None) -> str:
        """Select optimal model for device"""
        tier = SystemDetector.get_device_tier(specs)
        recommended = ModelSelector.MODEL_RECOMMENDATIONS[tier]

        logger.info(f"Device tier: {tier}")
        logger.info(f"Recommended models: {recommended}")

        # If we have available models, pick the best available
        if available_models:
            for model in recommended:
                # Check exact match or partial match
                for available in available_models:
                    if model in available or available in model:
                        logger.info(f"Selected model: {available}")
                        return available

            # No recommended model is available — pick best available model
            # Prefer larger parameter counts among what's installed
            preferred_order = ["14b", "13b", "8b", "7b", "4b", "3b", "1b"]
            for size in preferred_order:
                for available in available_models:
                    if size in available and "embed" not in available.lower():
                        logger.info(f"Selected best available model: {available}")
                        return available

            # Last resort: pick first non-embedding available model
            for available in available_models:
                if "embed" not in available.lower():
                    logger.info(f"Selected fallback available model: {available}")
                    return available

        # Return first recommendation as default
        selected = recommended[0]
        logger.info(f"Selected model (default): {selected}")
        return selected

    @staticmethod
    def get_memory_requirements(model_name: str) -> Dict[str, float]:
        """Estimate memory requirements for a model"""
        # Rough estimates in GB
        requirements = {
            # Large models
            "70b": {"min_ram": 64, "recommended_ram": 128, "vram": 40},
            "34b": {"min_ram": 32, "recommended_ram": 64, "vram": 20},
            # Medium models
            "13b": {"min_ram": 16, "recommended_ram": 32, "vram": 8},
            "8b": {"min_ram": 8, "recommended_ram": 16, "vram": 6},
            "7b": {"min_ram": 8, "recommended_ram": 16, "vram": 6},
            # Small models
            "3b": {"min_ram": 4, "recommended_ram": 8, "vram": 3},
            "1.1b": {"min_ram": 2, "recommended_ram": 4, "vram": 2},
        }

        # Find matching size
        for size, reqs in requirements.items():
            if size in model_name.lower():
                return reqs

        # Default to medium
        return {"min_ram": 8, "recommended_ram": 16, "vram": 6}

    @staticmethod
    def can_run_model(specs: DeviceSpecs, model_name: str) -> bool:
        """Check if device can run a specific model"""
        reqs = ModelSelector.get_memory_requirements(model_name)

        # Check RAM
        if specs.ram_gb < reqs["min_ram"]:
            logger.warning(
                f"Insufficient RAM for {model_name}: {specs.ram_gb}GB < {reqs['min_ram']}GB"
            )
            return False

        # Check GPU VRAM if using GPU
        if specs.gpu_available and specs.gpu_memory_gb < reqs.get("vram", 0):
            logger.warning(
                f"Insufficient VRAM for {model_name}: {specs.gpu_memory_gb}GB < {reqs['vram']}GB"
            )
            # Can still run on CPU

        return True


class ResourceMonitor:
    """Monitor resource usage in real-time"""

    @staticmethod
    def get_current_usage() -> Dict[str, Any]:
        """Get current resource usage"""
        ram = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage("/")

        usage = {
            "ram_used_percent": ram.percent,
            "ram_used_gb": ram.used / (1024**3),
            "ram_available_gb": ram.available / (1024**3),
            "cpu_percent": cpu,
            "disk_used_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3),
        }

        return usage

    @staticmethod
    def should_throttle(max_cpu_percent: float = 90, max_ram_percent: float = 90) -> bool:
        """Check if system is under heavy load and should throttle"""
        usage = ResourceMonitor.get_current_usage()

        if usage["ram_used_percent"] > max_ram_percent:
            logger.warning(f"High RAM usage: {usage['ram_used_percent']}%")
            return True

        if usage["cpu_percent"] > max_cpu_percent:
            logger.warning(f"High CPU usage: {usage['cpu_percent']}%")
            return True

        return False

    @staticmethod
    def get_optimal_batch_size(specs: DeviceSpecs) -> int:
        """Get optimal batch size for processing based on available RAM"""
        # Simple heuristic: 1 item per 2GB of available RAM
        available_ram = specs.ram_gb * 0.7  # Use 70% max
        batch_size = max(1, int(available_ram / 2))
        return min(batch_size, 10)  # Cap at 10


def auto_configure_system() -> Dict[str, Any]:
    """Auto-configure system based on detected specs"""
    specs = SystemDetector.detect()
    tier = SystemDetector.get_device_tier(specs)

    # Get available Ollama models
    try:
        import ollama

        client = ollama.Client()
        models = client.list()
        available_models = [
            getattr(m, "model", None) or m.get("name") or m.get("model", "")
            for m in models.get("models", [])
        ]
    except Exception as e:
        logger.warning(f"Could not check available models: {e}")
        available_models = []

    # Select model
    recommended_model = ModelSelector.select_model(specs, available_models)

    # Get optimal settings
    batch_size = ResourceMonitor.get_optimal_batch_size(specs)

    config = {
        "device_tier": tier,
        "specs": specs.__dict__,
        "recommended_model": recommended_model,
        "batch_size": batch_size,
        "use_gpu": specs.gpu_available,
        "max_workers": specs.cpu_cores,
    }

    logger.info(f"Auto-configuration complete: {config}")
    return config
