"""
Image Generation & Vision Skill - Generate images, OCR, and image analysis.

Features:
- Image generation (DALL-E, Stable Diffusion)
- OCR (Tesseract, PaddleOCR)
- Image analysis and classification
- Image editing and manipulation
- Face detection and recognition
- Object detection
"""

import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import io
import hashlib

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

    # Create dummy Image class to avoid NameError
    class Image:
        class Image:
            pass


try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from paddleocr import PaddleOCR

    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    import cv2
    import numpy as np

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class GeneratedImage:
    """Result from image generation."""

    success: bool
    image_path: Optional[str] = None
    image_data: Optional[bytes] = None
    prompt: str = ""
    model: str = ""
    size: str = ""
    error: Optional[str] = None
    generation_time: float = 0.0

    def save(self, path: str):
        """Save image to file."""
        if self.image_data:
            Path(path).write_bytes(self.image_data)
            self.image_path = path


@dataclass
class OCRResult:
    """Result from OCR."""

    success: bool
    text: str = ""
    confidence: float = 0.0
    words: List[Dict[str, Any]] = field(default_factory=list)
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    language: str = "eng"
    error: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class ImageAnalysis:
    """Result from image analysis."""

    success: bool
    labels: List[str] = field(default_factory=list)
    objects: List[Dict[str, Any]] = field(default_factory=list)
    faces: List[Dict[str, Any]] = field(default_factory=list)
    colors: List[str] = field(default_factory=list)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class ImageGenerator:
    """
    Generate images using AI models.

    Supports:
    - Stable Diffusion (local/API)
    - DALL-E (OpenAI)
    - Custom models
    """

    def __init__(
        self,
        provider: str = "stable-diffusion",
        model: str = "stabilityai/stable-diffusion-2",
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize image generator.

        Args:
            provider: Provider (stable-diffusion, dalle, custom)
            model: Model name or path
            api_key: API key for cloud providers
            cache_dir: Directory for caching images
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key

        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".opensable" / "images"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        size: str = "512x512",
        num_images: int = 1,
        steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> List[GeneratedImage]:
        """
        Generate images from text prompt.

        Args:
            prompt: Text description of image
            negative_prompt: What to avoid in image
            size: Image size (e.g., "512x512", "1024x1024")
            num_images: Number of images to generate
            steps: Number of diffusion steps
            guidance_scale: How closely to follow prompt

        Returns:
            List of GeneratedImage objects
        """
        start_time = datetime.now()

        if self.provider == "stable-diffusion":
            results = await self._generate_stable_diffusion(
                prompt, negative_prompt, size, num_images, steps, guidance_scale
            )
        elif self.provider == "dalle":
            results = await self._generate_dalle(prompt, size, num_images)
        else:
            results = [
                GeneratedImage(
                    success=False,
                    error=f"Unsupported provider: {self.provider}",
                    prompt=prompt,
                    model=self.model,
                )
            ]

        # Set generation time
        elapsed = (datetime.now() - start_time).total_seconds()
        for result in results:
            result.generation_time = elapsed / len(results)

        return results

    async def _generate_stable_diffusion(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        size: str,
        num_images: int,
        steps: int,
        guidance_scale: float,
    ) -> List[GeneratedImage]:
        """Generate images using Stable Diffusion."""
        try:
            from diffusers import StableDiffusionPipeline
            import torch

            # Load pipeline (cached)
            pipe = StableDiffusionPipeline.from_pretrained(
                self.model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )

            if torch.cuda.is_available():
                pipe = pipe.to("cuda")

            # Parse size
            width, height = map(int, size.split("x"))

            # Generate
            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            ).images

            # Convert to results
            results = []
            for i, image in enumerate(images):
                # Save to bytes
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="PNG")
                img_bytes.seek(0)

                # Save to cache
                cache_key = hashlib.sha256(
                    f"{prompt}_{i}_{datetime.now().isoformat()}".encode()
                ).hexdigest()[:16]
                cache_path = self.cache_dir / f"{cache_key}.png"
                image.save(cache_path)

                results.append(
                    GeneratedImage(
                        success=True,
                        image_path=str(cache_path),
                        image_data=img_bytes.getvalue(),
                        prompt=prompt,
                        model=self.model,
                        size=size,
                    )
                )

            return results

        except ImportError:
            return [
                GeneratedImage(
                    success=False,
                    error="diffusers not installed: pip install diffusers transformers torch",
                    prompt=prompt,
                    model=self.model,
                )
            ]
        except Exception as e:
            return [GeneratedImage(success=False, error=str(e), prompt=prompt, model=self.model)]

    async def _generate_dalle(
        self, prompt: str, size: str, num_images: int
    ) -> List[GeneratedImage]:
        """Generate images using DALL-E."""
        try:
            import openai

            if not self.api_key:
                return [
                    GeneratedImage(
                        success=False,
                        error="OpenAI API key required for DALL-E",
                        prompt=prompt,
                        model="dall-e-3",
                    )
                ]

            openai.api_key = self.api_key

            # DALL-E size format
            dalle_size = size if size in ["1024x1024", "1792x1024", "1024x1792"] else "1024x1024"

            results = []

            for _ in range(num_images):
                response = await asyncio.to_thread(
                    openai.images.generate,
                    model="dall-e-3",
                    prompt=prompt,
                    size=dalle_size,
                    quality="standard",
                    n=1,
                )

                # Download image
                import httpx

                image_url = response.data[0].url

                async with httpx.AsyncClient() as client:
                    img_response = await client.get(image_url)
                    img_data = img_response.content

                # Save to cache
                cache_key = hashlib.sha256(
                    f"{prompt}_{datetime.now().isoformat()}".encode()
                ).hexdigest()[:16]
                cache_path = self.cache_dir / f"{cache_key}.png"
                cache_path.write_bytes(img_data)

                results.append(
                    GeneratedImage(
                        success=True,
                        image_path=str(cache_path),
                        image_data=img_data,
                        prompt=prompt,
                        model="dall-e-3",
                        size=dalle_size,
                    )
                )

            return results

        except Exception as e:
            return [GeneratedImage(success=False, error=str(e), prompt=prompt, model="dall-e-3")]


class OCREngine:
    """
    Extract text from images using OCR.

    Supports:
    - Tesseract OCR (free, local)
    - PaddleOCR (deep learning, multilingual)
    - Cloud OCR services
    """

    def __init__(self, engine: str = "tesseract", languages: List[str] = None):
        """
        Initialize OCR engine.

        Args:
            engine: Engine type (tesseract, paddle, cloud)
            languages: List of language codes
        """
        self.engine = engine
        self.languages = languages or ["eng"]

        if engine == "paddle" and PADDLEOCR_AVAILABLE:
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")
        else:
            self.paddle_ocr = None

    async def extract_text(
        self, image_path: str, languages: Optional[List[str]] = None
    ) -> OCRResult:
        """
        Extract text from image.

        Args:
            image_path: Path to image file
            languages: Override default languages

        Returns:
            OCRResult with extracted text
        """
        start_time = datetime.now()
        languages = languages or self.languages

        try:
            if self.engine == "tesseract":
                result = await self._extract_tesseract(image_path, languages)
            elif self.engine == "paddle":
                result = await self._extract_paddle(image_path)
            else:
                result = OCRResult(success=False, error=f"Unsupported engine: {self.engine}")

            result.processing_time = (datetime.now() - start_time).total_seconds()
            return result

        except Exception as e:
            return OCRResult(
                success=False,
                error=str(e),
                processing_time=(datetime.now() - start_time).total_seconds(),
            )

    async def _extract_tesseract(self, image_path: str, languages: List[str]) -> OCRResult:
        """Extract text using Tesseract."""
        if not TESSERACT_AVAILABLE:
            return OCRResult(
                success=False, error="pytesseract not installed: pip install pytesseract"
            )

        if not PIL_AVAILABLE:
            return OCRResult(success=False, error="Pillow not installed: pip install Pillow")

        # Load image
        image = Image.open(image_path)

        # Extract text
        lang_str = "+".join(languages)
        text = pytesseract.image_to_string(image, lang=lang_str)

        # Extract word data
        data = pytesseract.image_to_data(image, lang=lang_str, output_type=pytesseract.Output.DICT)

        words = []
        for i in range(len(data["text"])):
            if data["text"][i].strip():
                words.append(
                    {
                        "text": data["text"][i],
                        "confidence": float(data["conf"][i]),
                        "box": {
                            "x": data["left"][i],
                            "y": data["top"][i],
                            "width": data["width"][i],
                            "height": data["height"][i],
                        },
                    }
                )

        # Calculate average confidence
        confidences = [w["confidence"] for w in words if w["confidence"] > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResult(
            success=True,
            text=text.strip(),
            confidence=avg_confidence,
            words=words,
            language=lang_str,
        )

    async def _extract_paddle(self, image_path: str) -> OCRResult:
        """Extract text using PaddleOCR."""
        if not PADDLEOCR_AVAILABLE:
            return OCRResult(success=False, error="paddleocr not installed: pip install paddleocr")

        # Run OCR
        result = await asyncio.to_thread(self.paddle_ocr.ocr, image_path, cls=True)

        # Parse results
        text_lines = []
        words = []
        total_confidence = 0.0
        count = 0

        for line in result[0]:
            box = line[0]
            text_data = line[1]
            text = text_data[0]
            confidence = float(text_data[1])

            text_lines.append(text)
            words.append({"text": text, "confidence": confidence, "box": {"points": box}})

            total_confidence += confidence
            count += 1

        avg_confidence = total_confidence / count if count > 0 else 0.0

        return OCRResult(
            success=True,
            text="\n".join(text_lines),
            confidence=avg_confidence,
            words=words,
            language="multi",
        )


class ImageAnalyzer:
    """
    Analyze images for objects, faces, colors, etc.

    Features:
    - Object detection
    - Face detection
    - Color analysis
    - Image classification
    """

    def __init__(self):
        """Initialize image analyzer."""
        self.face_cascade = None

        if CV2_AVAILABLE:
            # Load Haar cascade for face detection
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

    async def analyze(self, image_path: str) -> ImageAnalysis:
        """
        Analyze image.

        Args:
            image_path: Path to image

        Returns:
            ImageAnalysis with results
        """
        try:
            if not PIL_AVAILABLE:
                return ImageAnalysis(
                    success=False, error="Pillow not installed: pip install Pillow"
                )

            image = Image.open(image_path)

            # Get metadata
            metadata = {
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "width": image.width,
                "height": image.height,
            }

            # Extract dominant colors
            colors = await self._extract_colors(image)

            # Detect faces
            faces = []
            if CV2_AVAILABLE and self.face_cascade:
                faces = await self._detect_faces(image_path)

            return ImageAnalysis(
                success=True,
                colors=colors,
                faces=faces,
                metadata=metadata,
                description=f"{image.width}x{image.height} {image.format} image",
            )

        except Exception as e:
            return ImageAnalysis(success=False, error=str(e))

    async def _extract_colors(self, image: Any, num_colors: int = 5) -> List[str]:
        """Extract dominant colors from image."""
        # Resize for performance
        small_image = image.copy()
        small_image.thumbnail((150, 150))

        # Convert to RGB
        if small_image.mode != "RGB":
            small_image = small_image.convert("RGB")

        # Get color data
        pixels = list(small_image.getdata())

        # Simple color quantization
        from collections import Counter

        color_counts = Counter(pixels)
        dominant_colors = color_counts.most_common(num_colors)

        # Convert to hex
        hex_colors = []
        for color, _ in dominant_colors:
            hex_color = "#{:02x}{:02x}{:02x}".format(*color)
            hex_colors.append(hex_color)

        return hex_colors

    async def _detect_faces(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect faces in image."""
        # Load image with OpenCV
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces_data = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        faces = []
        for x, y, w, h in faces_data:
            faces.append(
                {
                    "box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    "confidence": 1.0,  # Haar cascades don't provide confidence
                }
            )

        return faces


# Example usage
async def main():
    """Example image processing."""

    print("=" * 50)
    print("Image Generation & Vision Examples")
    print("=" * 50)

    # OCR Example
    print("\n1. OCR (Tesseract)")
    ocr = OCREngine(engine="tesseract")

    # Create sample image with text
    if PIL_AVAILABLE:
        img = Image.new("RGB", (400, 100), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 30), "Hello, this is a test!", fill="black")
        test_img_path = "/tmp/test_ocr.png"
        img.save(test_img_path)

        result = await ocr.extract_text(test_img_path)
        if result.success:
            print(f"  Extracted text: {result.text}")
            print(f"  Confidence: {result.confidence:.2f}%")
            print(f"  Words found: {len(result.words)}")
        else:
            print(f"  Error: {result.error}")

    # Image Analysis
    print("\n2. Image Analysis")
    analyzer = ImageAnalyzer()

    if PIL_AVAILABLE:
        # Create test image
        test_img = Image.new("RGB", (300, 200), color="#3498db")
        test_img_path2 = "/tmp/test_analyze.png"
        test_img.save(test_img_path2)

        analysis = await analyzer.analyze(test_img_path2)
        if analysis.success:
            print(f"  Description: {analysis.description}")
            print(f"  Dominant colors: {', '.join(analysis.colors)}")
            print(f"  Faces detected: {len(analysis.faces)}")
        else:
            print(f"  Error: {analysis.error}")

    print("\n✅ Image processing examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
