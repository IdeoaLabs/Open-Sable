"""
Voice Message Handler - Telegram Voice Integration

Handles voice messages from Telegram with automatic transcription and TTS responses.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class VoiceMessageHandler:
    """Handles voice message transcription and synthesis"""

    def __init__(self, config, agent):
        self.config = config
        self.agent = agent
        self.voice_skill = None
        self._initialized = False

    async def initialize(self):
        """Initialize voice engines"""
        if self._initialized:
            return

        try:
            from opensable.skills.voice_skill import VoiceSkill

            self.voice_skill = VoiceSkill(self.config)
            await self.voice_skill.initialize()
            self._initialized = True
            logger.info("✅ Voice message handler initialized")
        except ImportError as e:
            logger.warning(f"Voice dependencies not available: {e}")
            self._initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize voice handler: {e}")
            self._initialized = False

    async def process_voice_message(
        self, audio_bytes: bytes, user_id: str, respond_with_voice: bool = False
    ) -> Dict[str, Any]:
        """
        Process voice message from Telegram

        Args:
            audio_bytes: Raw audio data (OGG/OPUS from Telegram)
            user_id: User identifier
            respond_with_voice: Whether to generate voice response

        Returns:
            Dict with transcription, text response, and optional voice file
        """
        if not self._initialized:
            await self.initialize()

        if not self._initialized or not self.voice_skill:
            return {"success": False, "error": "Voice processing not available"}

        temp_input = None
        temp_output = None

        try:
            # Save audio to temp file
            temp_input = Path(tempfile.mktemp(suffix=".ogg"))
            temp_input.write_bytes(audio_bytes)

            # Transcribe audio
            logger.info("🎙️ Transcribing voice message...")
            transcription = await self.voice_skill.listen(str(temp_input))

            if not transcription:
                return {"success": False, "error": "Could not transcribe audio"}

            logger.info(f"User said: {transcription}")

            # Process through agent
            response_text = await self.agent.process_message(user_id, transcription)

            result = {
                "success": True,
                "transcription": transcription,
                "response_text": response_text,
                "voice_file": None,
            }

            # Generate voice response if requested
            if respond_with_voice:
                logger.info("🔊 Generating voice response...")
                temp_output = await self.voice_skill.speak(response_text)

                if temp_output:
                    # Read voice file
                    voice_data = Path(temp_output).read_bytes()
                    result["voice_data"] = voice_data
                    result["voice_file"] = temp_output

            return result

        except Exception as e:
            logger.error(f"Voice processing error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
        finally:
            # Cleanup temp files
            if temp_input and temp_input.exists():
                temp_input.unlink()

    async def convert_ogg_to_wav(self, ogg_file: Path) -> Optional[Path]:
        """Convert OGG/OPUS to WAV for better compatibility"""
        try:
            import subprocess

            wav_file = ogg_file.with_suffix(".wav")

            # Use ffmpeg to convert
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    str(ogg_file),
                    "-ar",
                    "16000",  # 16kHz sample rate
                    "-ac",
                    "1",  # Mono
                    "-y",  # Overwrite
                    str(wav_file),
                ],
                check=True,
                capture_output=True,
            )

            logger.info(f"Converted {ogg_file} -> {wav_file}")
            return wav_file

        except Exception as e:
            logger.warning(f"Audio conversion failed: {e}")
            return ogg_file  # Return original if conversion fails
