"""
Tests for Image Generation and Vision features.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from skills.image_skill import ImageGenerator, OCREngine, ImageAnalyzer


class TestImageGenerator:
    """Test image generation"""
    
    @pytest.fixture
    def generator(self):
        return ImageGenerator(backend="stable_diffusion")
    
    @pytest.mark.asyncio
    async def test_generate_image(self, generator):
        """Test basic image generation"""
        result = await generator.generate(
            prompt="A test image",
            size=(512, 512),
            num_inference_steps=10
        )
        
        assert result is not None
        assert result.image_path.exists()
        assert result.metadata["prompt"] == "A test image"
        assert result.metadata["size"] == (512, 512)
    
    @pytest.mark.asyncio
    async def test_generate_with_negative_prompt(self, generator):
        """Test generation with negative prompt"""
        result = await generator.generate(
            prompt="Beautiful landscape",
            negative_prompt="blurry, low quality",
            size=(256, 256)
        )
        
        assert result is not None
        assert "negative_prompt" in result.metadata
    
    @pytest.mark.asyncio
    async def test_caching(self, generator):
        """Test image caching"""
        prompt = "Cached test image"
        
        # First generation
        result1 = await generator.generate(prompt, num_inference_steps=5)
        
        # Second generation (should use cache)
        result2 = await generator.generate(prompt, num_inference_steps=5)
        
        # Should return same image
        assert result1.image_path == result2.image_path
    
    def test_different_backends(self):
        """Test different generation backends"""
        sd_gen = ImageGenerator(backend="stable_diffusion")
        dalle_gen = ImageGenerator(backend="dalle")
        
        assert sd_gen.backend == "stable_diffusion"
        assert dalle_gen.backend == "dalle"


class TestOCREngine:
    """Test OCR text extraction"""
    
    @pytest.fixture
    def ocr(self):
        return OCREngine(backend="tesseract")
    
    @pytest.mark.asyncio
    async def test_extract_text_basic(self, ocr, tmp_path):
        """Test basic text extraction"""
        # Create test image with text
        from PIL import Image, ImageDraw
        
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 30), "Test OCR Text", fill='black')
        
        img_path = tmp_path / "test.png"
        img.save(img_path)
        
        result = await ocr.extract_text(str(img_path))
        
        assert result is not None
        assert "Test" in result.text or "OCR" in result.text
        assert result.confidence >= 0
    
    @pytest.mark.asyncio
    async def test_word_level_extraction(self, ocr, tmp_path):
        """Test word-level details"""
        from PIL import Image, ImageDraw
        
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 30), "Hello World", fill='black')
        
        img_path = tmp_path / "words.png"
        img.save(img_path)
        
        result = await ocr.extract_text(str(img_path))
        
        assert len(result.words) > 0
        assert all('text' in word for word in result.words)
    
    def test_backend_selection(self):
        """Test OCR backend selection"""
        tesseract = OCREngine(backend="tesseract")
        paddle = OCREngine(backend="paddleocr")
        
        assert tesseract.backend == "tesseract"
        assert paddle.backend == "paddleocr"


class TestImageAnalyzer:
    """Test image analysis"""
    
    @pytest.fixture
    def analyzer(self):
        return ImageAnalyzer()
    
    @pytest.mark.asyncio
    async def test_color_extraction(self, analyzer, tmp_path):
        """Test dominant color extraction"""
        from PIL import Image
        
        # Create solid color image
        img = Image.new('RGB', (100, 100), color=(255, 0, 0))
        img_path = tmp_path / "red.png"
        img.save(img_path)
        
        result = await analyzer.analyze(str(img_path))
        
        assert len(result.colors) > 0
        # Red should be dominant
        assert any('255' in color or 'red' in color.lower() for color in result.colors)
    
    @pytest.mark.asyncio
    async def test_metadata_extraction(self, analyzer, tmp_path):
        """Test image metadata extraction"""
        from PIL import Image
        
        img = Image.new('RGB', (800, 600))
        img_path = tmp_path / "metadata.png"
        img.save(img_path)
        
        result = await analyzer.analyze(str(img_path))
        
        assert result.metadata['width'] == 800
        assert result.metadata['height'] == 600
        assert 'format' in result.metadata
    
    @pytest.mark.asyncio
    async def test_face_detection(self, analyzer, tmp_path):
        """Test face detection (basic)"""
        from PIL import Image
        
        img = Image.new('RGB', (400, 400), color='white')
        img_path = tmp_path / "faces.png"
        img.save(img_path)
        
        result = await analyzer.analyze(str(img_path))
        
        # Should have faces list (may be empty for test image)
        assert isinstance(result.faces, list)
