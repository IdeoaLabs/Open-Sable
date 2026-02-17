"""
Image Analysis Integration

Integrates vision capabilities into messaging platforms:
- Image captioning and description
- OCR (text extraction)
- Object detection
- Visual question answering
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """Handles image analysis and vision tasks"""
    
    def __init__(self, config):
        self.config = config
        self.vision_model = None
        self.ocr_engine = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize vision models"""
        if self._initialized:
            return
        
        try:
            # Try to load local vision model (LLaVA, Qwen-VL, etc.)
            await self._init_vision_model()
            await self._init_ocr()
            self._initialized = True
            logger.info("✅ Image analyzer initialized")
        except Exception as e:
            logger.warning(f"Image analysis not available: {e}")
            self._initialized = False
    
    async def _init_vision_model(self):
        """Initialize local vision model"""
        model_name = getattr(self.config, 'vision_model', 'llava:7b')
        
        try:
            # Check if using Ollama for vision
            if hasattr(self.config, 'ollama_base_url'):
                import ollama
                client = ollama.AsyncClient(host=self.config.ollama_base_url)
                
                # Test if vision model is available
                models = await client.list()
                vision_models = [m for m in models.get('models', []) if 'llava' in m['name'].lower() or 'vision' in m['name'].lower()]
                
                if vision_models:
                    self.vision_model = client
                    logger.info(f"Using Ollama vision model: {vision_models[0]['name']}")
                else:
                    logger.warning("No vision models found in Ollama")
        except Exception as e:
            logger.warning(f"Failed to init vision model: {e}")
    
    async def _init_ocr(self):
        """Initialize OCR engine"""
        try:
            import pytesseract
            from PIL import Image
            
            # Test if tesseract is available
            pytesseract.get_tesseract_version()
            self.ocr_engine = pytesseract
            logger.info("OCR engine ready (Tesseract)")
        except Exception as e:
            logger.warning(f"OCR not available: {e}")
    
    async def analyze_image(
        self,
        image_bytes: bytes,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze image with optional question
        
        Args:
            image_bytes: Raw image data
            query: Optional question about the image
            
        Returns:
            Analysis results
        """
        if not self._initialized:
            await self.initialize()
        
        if not self._initialized:
            return {
                "success": False,
                "error": "Image analysis not available"
            }
        
        temp_file = None
        
        try:
            # Save to temp file
            temp_file = Path(tempfile.mktemp(suffix=".jpg"))
            temp_file.write_bytes(image_bytes)
            
            results = {
                "success": True,
                "description": None,
                "ocr_text": None,
                "objects": []
            }
            
            # Vision model analysis
            if self.vision_model:
                if query:
                    description = await self._vision_query(str(temp_file), query)
                else:
                    description = await self._vision_describe(str(temp_file))
                results["description"] = description
            
            # OCR if requested
            if self.ocr_engine and (query and "text" in query.lower() or not query):
                ocr_text = await self._extract_text(str(temp_file))
                if ocr_text:
                    results["ocr_text"] = ocr_text
            
            return results
            
        except Exception as e:
            logger.error(f"Image analysis error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            if temp_file and temp_file.exists():
                temp_file.unlink()
    
    async def _vision_describe(self, image_path: str) -> str:
        """Generate image description"""
        try:
            import ollama
            
            response = await self.vision_model.chat(
                model='llava:7b',
                messages=[{
                    'role': 'user',
                    'content': 'Describe this image in detail.',
                    'images': [image_path]
                }]
            )
            
            return response['message']['content']
        except Exception as e:
            logger.error(f"Vision description failed: {e}")
            return "Could not analyze image"
    
    async def _vision_query(self, image_path: str, question: str) -> str:
        """Answer question about image"""
        try:
            import ollama
            
            response = await self.vision_model.chat(
                model='llava:7b',
                messages=[{
                    'role': 'user',
                    'content': question,
                    'images': [image_path]
                }]
            )
            
            return response['message']['content']
        except Exception as e:
            logger.error(f"Vision query failed: {e}")
            return f"Could not answer question about image: {e}"
    
    async def _extract_text(self, image_path: str) -> Optional[str]:
        """Extract text using OCR"""
        try:
            from PIL import Image
            
            image = Image.open(image_path)
            text = self.ocr_engine.image_to_string(image)
            
            # Clean up whitespace
            text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
            
            return text if text else None
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return None
    
    async def detect_objects(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """Detect objects in image"""
        # Placeholder for object detection
        # Would use YOLO, Detectron2, etc.
        return []


# Telegram photo handler integration
async def handle_telegram_photo(message, image_analyzer, agent, user_id):
    """Handle photo message from Telegram"""
    try:
        # Get largest photo
        photo = message.photo[-1]
        
        # Download
        photo_file = await photo.download_to_drive()
        image_bytes = Path(photo_file.name).read_bytes()
        
        # Get caption as query
        query = message.caption if message.caption else None
        
        # Analyze
        result = await image_analyzer.analyze_image(image_bytes, query)
        
        if not result.get("success"):
            return f"❌ {result.get('error', 'Analysis failed')}"
        
        # Format response
        response_parts = ["🖼️ **Image Analysis:**\n"]
        
        if result.get("description"):
            response_parts.append(f"**Description:** {result['description']}\n")
        
        if result.get("ocr_text"):
            response_parts.append(f"\n📝 **Text detected:**\n```\n{result['ocr_text']}\n```")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.error(f"Photo handling error: {e}", exc_info=True)
        return f"❌ Error: {str(e)}"
