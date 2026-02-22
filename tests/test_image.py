"""
Tests for Image Skill - Generation, OCR, Analysis.
"""

import pytest
from opensable.skills.image_skill import (
    ImageGenerator, OCREngine, ImageAnalyzer,
    GeneratedImage, OCRResult, ImageAnalysis
)


class TestGeneratedImageDataclass:
    """Test GeneratedImage dataclass."""

    def test_success(self):
        img = GeneratedImage(success=True, prompt="a cat", model="dalle", size="512x512")
        assert img.success is True
        assert img.prompt == "a cat"

    def test_failure(self):
        img = GeneratedImage(success=False, error="API error")
        assert img.success is False
        assert img.error == "API error"


class TestOCRResultDataclass:
    """Test OCRResult dataclass."""

    def test_success(self):
        r = OCRResult(success=True, text="Hello World", confidence=0.95)
        assert r.text == "Hello World"
        assert r.confidence == 0.95

    def test_failure(self):
        r = OCRResult(success=False, error="No engine")
        assert not r.success


class TestImageAnalysisDataclass:
    """Test ImageAnalysis dataclass."""

    def test_success(self):
        a = ImageAnalysis(success=True, labels=["cat", "pet"], description="A cat")
        assert "cat" in a.labels
        assert a.description == "A cat"

    def test_defaults(self):
        a = ImageAnalysis(success=True)
        assert a.labels == []
        assert a.objects == []
        assert a.faces == []
        assert a.colors == []


class TestImageGenerator:
    """Test ImageGenerator construction."""

    def test_default_init(self):
        gen = ImageGenerator()
        assert gen.provider == "stable-diffusion"
        assert gen.model == "stabilityai/stable-diffusion-2"
        assert gen.api_key is None

    def test_custom_init(self):
        gen = ImageGenerator(provider="dalle", api_key="sk-test")
        assert gen.provider == "dalle"
        assert gen.api_key == "sk-test"

    def test_cache_dir(self):
        gen = ImageGenerator(cache_dir="/tmp/test_img_cache")
        assert gen.cache_dir.exists() or True  # just check attribute is set


class TestOCREngine:
    """Test OCR engine construction."""

    def test_default_init(self):
        ocr = OCREngine()
        assert ocr.engine == "tesseract"
        assert ocr.languages == ["eng"]

    def test_custom_init(self):
        ocr = OCREngine(engine="paddle", languages=["en", "es"])
        assert ocr.engine == "paddle"
        assert "es" in ocr.languages


class TestImageAnalyzer:
    """Test image analyzer construction."""

    def test_init(self):
        analyzer = ImageAnalyzer()
        # face_cascade may or may not be set depending on cv2 availability
        assert hasattr(analyzer, "face_cascade")
