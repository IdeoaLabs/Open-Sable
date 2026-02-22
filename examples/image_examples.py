"""
Image Generation, OCR, and Analysis Examples.

Demonstrates DALL-E/Stable Diffusion generation, Tesseract/PaddleOCR text extraction,
and image analysis with face detection and color extraction.
"""

import asyncio
from skills.image_skill import ImageGenerator, OCREngine, ImageAnalyzer


async def main():
    """Run image skill examples."""

    print("=" * 60)
    print("Image Generation & Vision Examples")
    print("=" * 60)

    # Example 1: Image generation with Stable Diffusion
    print("\n1. Image Generation (Stable Diffusion)")
    print("-" * 40)

    generator = ImageGenerator()

    prompt = "A futuristic AI assistant robot helping humans, digital art style"
    result = await generator.generate(
        prompt=prompt,
        negative_prompt="blurry, low quality",
        size=(512, 512),
        num_inference_steps=20,
    )

    if result:
        print(f"Generated image: {result.image_path}")
        print(f"Size: {result.metadata.get('size')}")
        print(f"Model: {result.metadata.get('model')}")

    # Example 2: Image generation with different sizes
    print("\n2. Different Image Sizes")
    print("-" * 40)

    sizes = [(256, 256), (512, 512), (768, 768)]

    for width, height in sizes:
        result = await generator.generate(
            prompt="Simple geometric pattern", size=(width, height), num_inference_steps=10
        )
        print(f"  Generated {width}x{height} image: {result.image_path if result else 'failed'}")

    # Example 3: OCR with Tesseract
    print("\n3. OCR Text Extraction (Tesseract)")
    print("-" * 40)

    ocr = OCREngine(backend="tesseract")

    # Create a test image with text
    from PIL import Image, ImageDraw

    test_image = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(test_image)
    draw.text((10, 30), "Hello from Open-Sable!", fill="black")
    test_image.save("/tmp/test_text.png")

    result = await ocr.extract_text("/tmp/test_text.png")

    print(f"Extracted text: '{result.text.strip()}'")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Words detected: {len(result.words)}")

    # Example 4: OCR with word-level details
    print("\n4. Word-Level OCR Details")
    print("-" * 40)

    if result.words:
        print("Word details:")
        for word in result.words[:5]:
            print(f"  - '{word['text']}' (confidence: {word.get('confidence', 0):.2f})")

    # Example 5: Image analysis
    print("\n5. Image Analysis")
    print("-" * 40)

    analyzer = ImageAnalyzer()

    # Create a colored test image
    color_image = Image.new("RGB", (200, 200), color=(100, 150, 200))
    color_image.save("/tmp/test_color.png")

    analysis = await analyzer.analyze("/tmp/test_color.png")

    print(f"Dominant colors: {', '.join(analysis.colors[:3])}")
    print(f"Faces detected: {len(analysis.faces)}")
    print(f"Image format: {analysis.metadata.get('format')}")
    print(f"Image size: {analysis.metadata.get('width')}x{analysis.metadata.get('height')}")

    # Example 6: Face detection
    print("\n6. Face Detection")
    print("-" * 40)

    # Note: For real face detection, you'd need an actual photo with faces
    print("Face detection requires OpenCV Haar cascades")
    print("Example: Load an image with faces for detection")

    # Example 7: Batch image generation
    print("\n7. Batch Image Generation")
    print("-" * 40)

    prompts = ["Sunset over mountains", "Abstract colorful pattern", "Minimalist landscape"]

    tasks = [generator.generate(prompt, num_inference_steps=10) for prompt in prompts]

    results = await asyncio.gather(*tasks)

    successful = sum(1 for r in results if r is not None)
    print(f"Generated {successful}/{len(prompts)} images")

    # Example 8: Image metadata extraction
    print("\n8. Image Metadata Extraction")
    print("-" * 40)

    if result:
        print("Metadata:")
        for key, value in result.metadata.items():
            print(f"  {key}: {value}")

    # Example 9: Color extraction from real image
    print("\n9. Dominant Color Extraction")
    print("-" * 40)

    # Create gradient image
    gradient = Image.new("RGB", (300, 300))
    pixels = gradient.load()
    for y in range(300):
        for x in range(300):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    gradient.save("/tmp/gradient.png")

    analysis = await analyzer.analyze("/tmp/gradient.png")
    print("Top 5 colors:")
    for i, color in enumerate(analysis.colors[:5], 1):
        print(f"  {i}. {color}")

    # Example 10: Generation with caching
    print("\n10. Generation with Caching")
    print("-" * 40)

    import time

    prompt = "Test caching with same prompt"

    # First generation
    start = time.time()
    result1 = await generator.generate(prompt, num_inference_steps=5)
    time1 = time.time() - start
    print(f"First generation: {time1:.2f}s")

    # Second generation (should be cached)
    start = time.time()
    result2 = await generator.generate(prompt, num_inference_steps=5)
    time2 = time.time() - start
    print(f"Second generation (cached): {time2:.2f}s")

    if time2 < time1:
        print("✅ Caching working!")

    # Example 11: OCR on generated image
    print("\n11. OCR on Generated Image")
    print("-" * 40)

    # Generate image with text
    text_prompt = "A white sign with black text saying 'SABLECORE AI'"
    text_result = await generator.generate(text_prompt, num_inference_steps=15)

    if text_result and text_result.image_path:
        ocr_result = await ocr.extract_text(text_result.image_path)
        print(f"Extracted from generated image: '{ocr_result.text.strip()}'")

    # Example 12: Multiple OCR backends comparison
    print("\n12. OCR Backend Comparison")
    print("-" * 40)

    tesseract_ocr = OCREngine(backend="tesseract")
    paddleocr_ocr = OCREngine(backend="paddleocr")

    print("Tesseract backend: Initialized")
    print("PaddleOCR backend: Initialized")
    print("Both can be used for different scenarios")

    print("\n" + "=" * 60)
    print("✅ Image examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
