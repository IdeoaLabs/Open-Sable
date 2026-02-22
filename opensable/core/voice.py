"""
Voice Support - Text-to-Speech and Speech-to-Text
"""

import logging
import asyncio
from pathlib import Path
from typing import Optional
import tempfile

logger = logging.getLogger(__name__)


class TTSEngine:
    """Text-to-Speech engine with multiple backends"""

    def __init__(self, config):
        self.config = config
        self.engine_type = config.tts_engine if hasattr(config, "tts_engine") else "local"
        self.engine = None

    async def initialize(self):
        """Initialize TTS engine"""
        if self.engine_type == "elevenlabs":
            await self._init_elevenlabs()
        elif self.engine_type == "openai":
            await self._init_openai()
        else:
            await self._init_local()

    async def _init_local(self):
        """Initialize local TTS (pyttsx3)"""
        try:
            import pyttsx3

            self.engine = pyttsx3.init()

            # Configure voice
            voices = self.engine.getProperty("voices")
            if voices:
                # Use first available voice
                self.engine.setProperty("voice", voices[0].id)

            # Set rate and volume
            self.engine.setProperty("rate", 150)  # Speed
            self.engine.setProperty("volume", 0.9)  # Volume

            logger.info("Initialized local TTS (pyttsx3)")
        except Exception as e:
            logger.error(f"Failed to initialize local TTS: {e}")
            self.engine = None

    async def _init_elevenlabs(self):
        """Initialize ElevenLabs TTS"""
        try:
            from elevenlabs import set_api_key, generate, voices

            if not hasattr(self.config, "elevenlabs_api_key"):
                logger.error("ElevenLabs API key not configured")
                return

            set_api_key(self.config.elevenlabs_api_key)

            # List available voices
            available_voices = voices()
            logger.info(f"ElevenLabs voices: {[v.name for v in available_voices]}")

            self.engine = "elevenlabs"
            logger.info("Initialized ElevenLabs TTS")

        except ImportError:
            logger.error("elevenlabs library not installed")
            self.engine = None
        except Exception as e:
            logger.error(f"Failed to initialize ElevenLabs: {e}")
            self.engine = None

    async def _init_openai(self):
        """Initialize OpenAI TTS"""
        try:
            from openai import AsyncOpenAI

            if not hasattr(self.config, "openai_api_key"):
                logger.error("OpenAI API key not configured")
                return

            self.engine = AsyncOpenAI(api_key=self.config.openai_api_key)
            logger.info("Initialized OpenAI TTS")

        except ImportError:
            logger.error("openai library not installed")
            self.engine = None
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI TTS: {e}")
            self.engine = None

    async def speak(self, text: str, output_file: Optional[Path] = None) -> Optional[Path]:
        """
        Convert text to speech

        Returns path to audio file or None if failed
        """
        if not self.engine:
            logger.warning("TTS engine not initialized")
            return None

        try:
            if self.engine_type == "local":
                return await self._speak_local(text, output_file)
            elif self.engine_type == "elevenlabs":
                return await self._speak_elevenlabs(text, output_file)
            elif self.engine_type == "openai":
                return await self._speak_openai(text, output_file)
        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            return None

    async def _speak_local(self, text: str, output_file: Optional[Path]) -> Optional[Path]:
        """Local TTS using pyttsx3"""
        if not output_file:
            output_file = Path(tempfile.mktemp(suffix=".mp3"))

        # pyttsx3 is synchronous, run in executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._pyttsx3_save, text, str(output_file))

        return output_file

    def _pyttsx3_save(self, text: str, filename: str):
        """Helper to save pyttsx3 audio"""
        self.engine.save_to_file(text, filename)
        self.engine.runAndWait()

    async def _speak_elevenlabs(self, text: str, output_file: Optional[Path]) -> Optional[Path]:
        """ElevenLabs TTS"""
        from elevenlabs import generate

        if not output_file:
            output_file = Path(tempfile.mktemp(suffix=".mp3"))

        # Generate audio
        audio = generate(
            text=text,
            voice=getattr(self.config, "elevenlabs_voice", "Bella"),
            model="eleven_monolingual_v1",
        )

        # Save to file
        with open(output_file, "wb") as f:
            f.write(audio)

        logger.info(f"Generated ElevenLabs audio: {output_file}")
        return output_file

    async def _speak_openai(self, text: str, output_file: Optional[Path]) -> Optional[Path]:
        """OpenAI TTS"""
        if not output_file:
            output_file = Path(tempfile.mktemp(suffix=".mp3"))

        # Generate speech
        response = await self.engine.audio.speech.create(
            model="tts-1", voice=getattr(self.config, "openai_voice", "alloy"), input=text
        )

        # Save to file
        with open(output_file, "wb") as f:
            f.write(response.content)

        logger.info(f"Generated OpenAI audio: {output_file}")
        return output_file


class STTEngine:
    """Speech-to-Text engine with multiple backends"""

    def __init__(self, config):
        self.config = config
        self.engine_type = config.stt_engine if hasattr(config, "stt_engine") else "whisper"
        self.engine = None

    async def initialize(self):
        """Initialize STT engine"""
        if self.engine_type == "whisper_local":
            await self._init_whisper_local()
        elif self.engine_type == "openai":
            await self._init_openai()
        else:
            await self._init_whisper_local()

    async def _init_whisper_local(self):
        """Initialize local Whisper"""
        try:
            import whisper

            # Load model (base, small, medium, large)
            model_size = getattr(self.config, "whisper_model", "base")
            self.engine = whisper.load_model(model_size)

            logger.info(f"Initialized local Whisper ({model_size})")

        except ImportError:
            logger.error("whisper library not installed")
            self.engine = None
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {e}")
            self.engine = None

    async def _init_openai(self):
        """Initialize OpenAI Whisper API"""
        try:
            from openai import AsyncOpenAI

            if not hasattr(self.config, "openai_api_key"):
                logger.error("OpenAI API key not configured")
                return

            self.engine = AsyncOpenAI(api_key=self.config.openai_api_key)
            self.engine_type = "openai"
            logger.info("Initialized OpenAI Whisper API")

        except ImportError:
            logger.error("openai library not installed")
            self.engine = None
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI Whisper: {e}")
            self.engine = None

    async def transcribe(self, audio_file: Path, language: str = "en") -> Optional[str]:
        """
        Transcribe audio file to text

        Returns transcribed text or None if failed
        """
        if not self.engine:
            logger.warning("STT engine not initialized")
            return None

        try:
            if self.engine_type == "whisper_local":
                return await self._transcribe_local(audio_file, language)
            elif self.engine_type == "openai":
                return await self._transcribe_openai(audio_file, language)
        except Exception as e:
            logger.error(f"STT error: {e}", exc_info=True)
            return None

    async def _transcribe_local(self, audio_file: Path, language: str) -> Optional[str]:
        """Local Whisper transcription"""
        # Run in executor (CPU intensive)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.engine.transcribe(str(audio_file), language=language)
        )

        text = result["text"].strip()
        logger.info(f"Transcribed: {text[:100]}...")
        return text

    async def _transcribe_openai(self, audio_file: Path, language: str) -> Optional[str]:
        """OpenAI Whisper API transcription"""
        with open(audio_file, "rb") as f:
            transcript = await self.engine.audio.transcriptions.create(
                model="whisper-1", file=f, language=language
            )

        text = transcript.text.strip()
        logger.info(f"Transcribed: {text[:100]}...")
        return text


class VoiceManager:
    """Manages TTS and STT for the agent"""

    def __init__(self, config):
        self.config = config
        self.tts = TTSEngine(config)
        self.stt = STTEngine(config)
        self.initialized = False

    async def initialize(self):
        """Initialize both TTS and STT"""
        await self.tts.initialize()
        await self.stt.initialize()
        self.initialized = True
        logger.info("Voice manager initialized")

    async def text_to_speech(self, text: str, output_file: Optional[Path] = None) -> Optional[Path]:
        """Convert text to speech audio file"""
        if not self.initialized:
            await self.initialize()

        return await self.tts.speak(text, output_file)

    async def speech_to_text(self, audio_file: Path, language: str = "en") -> Optional[str]:
        """Convert speech audio file to text"""
        if not self.initialized:
            await self.initialize()

        return await self.stt.transcribe(audio_file, language)
