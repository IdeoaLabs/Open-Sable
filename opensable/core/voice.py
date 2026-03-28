"""
Voice Support - Text-to-Speech and Speech-to-Text

TTS backends: local (pyttsx3), ElevenLabs (cloud), OpenAI (cloud)
STT backends: local (Whisper), OpenAI Whisper API (cloud)
Default is local,  set TTS_PROVIDER=elevenlabs + ELEVENLABS_API_KEY for cloud.
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
        self.engine_type = getattr(config, "tts_provider", "local")
        self.engine = None

    async def initialize(self):
        """Initialize TTS engine"""
        if self.engine_type == "piper":
            await self._init_piper()
        elif self.engine_type == "elevenlabs":
            await self._init_elevenlabs()
        elif self.engine_type == "openai":
            await self._init_openai()
        else:
            await self._init_local()

    async def _init_piper(self):
        """Verify Piper binary and model file are available."""
        import shutil

        binary    = getattr(self.config, "piper_binary", "piper")
        voice     = getattr(self.config, "piper_voice", "en_US-kusal-medium")
        model_dir = Path(getattr(self.config, "piper_model_dir", ""))
        model_path = model_dir / f"{voice}.onnx"

        resolved = shutil.which(binary) or (binary if Path(binary).is_file() else None)
        if not resolved:
            raise RuntimeError(
                f"Piper binary '{binary}' not found. "
                "Run install-pi.sh or set PIPER_BINARY to the full path."
            )
        if not model_path.exists():
            raise RuntimeError(
                f"Piper model not found: {model_path}. "
                "Run install-pi.sh to download it."
            )
        self.engine = {"binary": resolved, "model": str(model_path)}
        logger.info(f"Piper TTS ready — voice: {voice}")

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
        """Initialize ElevenLabs TTS (SDK v2+)"""
        try:
            from elevenlabs.client import AsyncElevenLabs

            api_key = getattr(self.config, "elevenlabs_api_key", None)
            if not api_key:
                logger.error("ElevenLabs API key not configured (set ELEVENLABS_API_KEY)")
                return

            self.engine = AsyncElevenLabs(api_key=api_key)
            logger.info("Initialized ElevenLabs TTS (async client)")

        except ImportError:
            logger.error(
                "elevenlabs library not installed. "
                "Install with: pip install 'opensable[voice]'  or  pip install elevenlabs"
            )
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
            if self.engine_type == "piper":
                return await self._speak_piper(text, output_file)
            elif self.engine_type == "local":
                return await self._speak_local(text, output_file)
            elif self.engine_type == "elevenlabs":
                return await self._speak_elevenlabs(text, output_file)
            elif self.engine_type == "openai":
                return await self._speak_openai(text, output_file)
        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            return None

    async def _speak_piper(self, text: str, output_file: Optional[Path]) -> Optional[Path]:
        """Piper TTS: synthesize WAV + optionally play via aplay (Pi speaker)"""
        if not output_file:
            output_file = Path(tempfile.mktemp(suffix=".wav"))

        binary = self.engine["binary"]
        model  = self.engine["model"]

        proc = await asyncio.create_subprocess_exec(
            binary,
            "--model", model,
            "--output-file", str(output_file),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate(input=text.encode())
        if proc.returncode != 0:
            logger.error(f"Piper failed: {stderr.decode().strip()}")
            return None

        logger.info(f"Piper TTS saved to {output_file}")

        # Play on the local speaker if auto_play is enabled (Pi mode)
        if getattr(self.config, "piper_auto_play", True):
            device = getattr(self.config, "piper_aplay_device", "plug:dmix")
            play = await asyncio.create_subprocess_exec(
                "aplay", f"-D{device}", str(output_file),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await play.wait()

        return output_file

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
        """ElevenLabs TTS (SDK v2+)"""
        if not output_file:
            output_file = Path(tempfile.mktemp(suffix=".mp3"))

        voice_id = getattr(self.config, "elevenlabs_voice_id", None) or "JBFqnCBsd6RMkjVDRZzb"
        model_id = getattr(self.config, "elevenlabs_model", "eleven_multilingual_v2")

        # AsyncElevenLabs.text_to_speech.convert() returns an async iterator of bytes
        audio_iter = await self.engine.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            output_format="mp3_44100_128",
        )

        # Collect bytes and write to file
        with open(output_file, "wb") as f:
            async for chunk in audio_iter:
                if isinstance(chunk, bytes):
                    f.write(chunk)

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
        self.engine_type = getattr(config, "stt_provider", "whisper_local")
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
