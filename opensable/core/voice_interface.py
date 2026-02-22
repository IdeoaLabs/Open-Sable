"""
Voice Interface - Speech-to-Text (STT) and Text-to-Speech (TTS) for local operation.

Uses:
- Whisper (OpenAI) for STT - runs locally via faster-whisper
- Piper TTS for speech synthesis - fast, local, natural-sounding
- Voice Activity Detection (VAD) for efficient processing
- Audio streaming support for real-time interaction
"""

import asyncio
import logging
import wave
import io
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, AsyncIterator
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Supported audio formats."""

    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"


class WhisperModel(Enum):
    """Whisper model sizes."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large-v3"


class TTSVoice(Enum):
    """Available TTS voices."""

    EN_US_FEMALE = "en_US-lessac-medium"
    EN_US_MALE = "en_US-libritts-high"
    EN_GB_FEMALE = "en_GB-alba-medium"
    ES_ES_FEMALE = "es_ES-davefx-medium"
    ES_MX_MALE = "es_MX-claude-high"


@dataclass
class TranscriptionResult:
    """Speech-to-text result."""

    text: str
    language: str
    confidence: float
    segments: List[Dict[str, Any]]
    duration: float
    timestamp: datetime


@dataclass
class SynthesisResult:
    """Text-to-speech result."""

    audio_data: bytes
    format: AudioFormat
    duration: float
    sample_rate: int
    channels: int


class WhisperSTT:
    """
    Speech-to-Text using faster-whisper (local).

    Features:
    - Fast transcription with CTranslate2
    - Multiple model sizes (tiny to large)
    - Language detection
    - Timestamp alignment
    - VAD (Voice Activity Detection)
    """

    def __init__(
        self,
        model_size: WhisperModel = WhisperModel.BASE,
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
    ):
        """
        Initialize Whisper STT.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v3)
            device: Device to use (cpu, cuda)
            compute_type: Computation type (int8, float16, float32)
            language: Force specific language (None for auto-detect)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.model = None

        logger.info(f"Initializing Whisper STT: {model_size.value} on {device}")

    async def load_model(self):
        """Load Whisper model (lazy loading)."""
        if self.model is not None:
            return

        try:
            from faster_whisper import WhisperModel as FasterWhisperModel

            logger.info(f"Loading Whisper model: {self.model_size.value}")
            self.model = FasterWhisperModel(
                self.model_size.value, device=self.device, compute_type=self.compute_type
            )
            logger.info("Whisper model loaded successfully")
        except ImportError:
            logger.error("faster-whisper not installed. Install with: pip install faster-whisper")
            raise
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise

    async def transcribe_file(
        self, audio_path: str, task: str = "transcribe", vad_filter: bool = True, beam_size: int = 5
    ) -> TranscriptionResult:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            task: 'transcribe' or 'translate' (to English)
            vad_filter: Use Voice Activity Detection
            beam_size: Beam search size (higher = better but slower)

        Returns:
            TranscriptionResult with text and metadata
        """
        await self.load_model()

        start_time = datetime.utcnow()

        try:
            # Transcribe
            segments, info = self.model.transcribe(
                audio_path,
                task=task,
                language=self.language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                word_timestamps=True,
            )

            # Collect segments
            all_segments = []
            full_text = []

            for segment in segments:
                seg_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "confidence": segment.avg_logprob,
                }
                all_segments.append(seg_dict)
                full_text.append(segment.text)

            duration = (datetime.utcnow() - start_time).total_seconds()

            result = TranscriptionResult(
                text=" ".join(full_text).strip(),
                language=info.language,
                confidence=info.language_probability,
                segments=all_segments,
                duration=duration,
                timestamp=start_time,
            )

            logger.info(f"Transcribed {len(all_segments)} segments in {duration:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise

    async def transcribe_bytes(
        self, audio_data: bytes, sample_rate: int = 16000, **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio from bytes.

        Args:
            audio_data: Raw audio bytes
            sample_rate: Sample rate in Hz
            **kwargs: Additional args for transcribe_file

        Returns:
            TranscriptionResult
        """
        # Save to temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            # Write WAV header
            with wave.open(tmp.name, "wb") as wav:
                wav.setnchannels(1)  # Mono
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(sample_rate)
                wav.writeframes(audio_data)

            tmp_path = tmp.name

        try:
            result = await self.transcribe_file(tmp_path, **kwargs)
            return result
        finally:
            # Cleanup
            try:
                os.unlink(tmp_path)
            except:
                pass


class PiperTTS:
    """
    Text-to-Speech using Piper (local, fast, high-quality).

    Features:
    - Fast synthesis (<1s for typical sentences)
    - Natural-sounding voices
    - Multiple languages and accents
    - Runs completely locally
    - Low resource usage
    """

    def __init__(self, voice: TTSVoice = TTSVoice.EN_US_FEMALE, models_dir: Optional[Path] = None):
        """
        Initialize Piper TTS.

        Args:
            voice: Voice to use
            models_dir: Directory containing Piper voice models
        """
        self.voice = voice
        self.models_dir = models_dir or Path("./models/piper")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.process = None

        logger.info(f"Initializing Piper TTS with voice: {voice.value}")

    async def download_voice_model(self):
        """Download voice model if not present."""
        model_path = self.models_dir / f"{self.voice.value}.onnx"
        config_path = self.models_dir / f"{self.voice.value}.onnx.json"

        if model_path.exists() and config_path.exists():
            logger.info(f"Voice model already exists: {self.voice.value}")
            return

        logger.info(f"Downloading voice model: {self.voice.value}")

        # Download from Piper releases
        base_url = "https://github.com/rhasspy/piper/releases/download/v1.2.0"

        import aiohttp

        async with aiohttp.ClientSession() as session:
            # Download model
            model_url = f"{base_url}/{self.voice.value}.onnx"
            async with session.get(model_url) as resp:
                if resp.status == 200:
                    with open(model_path, "wb") as f:
                        f.write(await resp.read())
                    logger.info(f"Downloaded model: {model_path}")

            # Download config
            config_url = f"{base_url}/{self.voice.value}.onnx.json"
            async with session.get(config_url) as resp:
                if resp.status == 200:
                    with open(config_path, "wb") as f:
                        f.write(await resp.read())
                    logger.info(f"Downloaded config: {config_path}")

    async def synthesize(
        self,
        text: str,
        output_format: AudioFormat = AudioFormat.WAV,
        speed: float = 1.0,
        output_path: Optional[str] = None,
    ) -> SynthesisResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            output_format: Audio format
            speed: Speech speed multiplier (0.5 - 2.0)
            output_path: Optional path to save audio

        Returns:
            SynthesisResult with audio data
        """
        await self.download_voice_model()

        model_path = self.models_dir / f"{self.voice.value}.onnx"

        # Use piper binary or Python library
        try:
            # Try using piper-tts Python library
            from piper import PiperVoice
            import numpy as np

            voice = PiperVoice.load(str(model_path))

            # Synthesize
            audio_data = []
            for audio_chunk in voice.synthesize_stream_raw(text):
                audio_data.append(audio_chunk)

            audio_array = np.concatenate(audio_data)

            # Convert to bytes
            audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()

            # Create WAV
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(voice.config.sample_rate)
                wav.writeframes(audio_bytes)

            audio_data_final = wav_buffer.getvalue()

            result = SynthesisResult(
                audio_data=audio_data_final,
                format=output_format,
                duration=len(audio_array) / voice.config.sample_rate,
                sample_rate=voice.config.sample_rate,
                channels=1,
            )

            # Save if requested
            if output_path:
                with open(output_path, "wb") as f:
                    f.write(audio_data_final)
                logger.info(f"Saved audio to {output_path}")

            logger.info(f"Synthesized {len(text)} chars in {result.duration:.2f}s")
            return result

        except ImportError:
            # Fallback to piper binary
            logger.warning("piper-tts library not found, using piper binary")
            return await self._synthesize_with_binary(text, output_format, speed, output_path)

    async def _synthesize_with_binary(
        self, text: str, output_format: AudioFormat, speed: float, output_path: Optional[str]
    ) -> SynthesisResult:
        """Synthesize using piper binary."""
        import tempfile

        model_path = self.models_dir / f"{self.voice.value}.onnx"

        # Create temp output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Run piper
            cmd = [
                "piper",
                "--model",
                str(model_path),
                "--output_file",
                tmp_path,
                "--length_scale",
                str(1.0 / speed),
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate(input=text.encode())

            if process.returncode != 0:
                raise Exception(f"Piper failed: {stderr.decode()}")

            # Read audio
            with open(tmp_path, "rb") as f:
                audio_data = f.read()

            # Get duration
            with wave.open(tmp_path, "rb") as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                duration = frames / float(rate)
                sample_rate = rate
                channels = wav.getnchannels()

            result = SynthesisResult(
                audio_data=audio_data,
                format=output_format,
                duration=duration,
                sample_rate=sample_rate,
                channels=channels,
            )

            # Save if requested
            if output_path:
                with open(output_path, "wb") as f:
                    f.write(audio_data)

            return result

        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass


class VoiceInterface:
    """
    Complete voice interface combining STT and TTS.

    Features:
    - Voice commands (speech → text → processing → speech)
    - Real-time audio streaming
    - Voice Activity Detection
    - Configurable voices and languages
    - Conversation mode
    """

    def __init__(
        self,
        whisper_model: WhisperModel = WhisperModel.BASE,
        tts_voice: TTSVoice = TTSVoice.EN_US_FEMALE,
        language: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize voice interface.

        Args:
            whisper_model: Whisper model size
            tts_voice: TTS voice
            language: STT language (None for auto-detect)
            device: Device for inference
        """
        self.stt = WhisperSTT(model_size=whisper_model, device=device, language=language)

        self.tts = PiperTTS(voice=tts_voice)

        self.conversation_mode = False
        self.conversation_callback: Optional[Callable] = None

        logger.info("Voice interface initialized")

    async def voice_command(
        self,
        audio_input: str,
        respond_with_voice: bool = True,
        command_handler: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Process voice command end-to-end.

        Args:
            audio_input: Path to audio file or bytes
            respond_with_voice: Generate voice response
            command_handler: Async function to process text command

        Returns:
            Dict with transcription, response text, and audio
        """
        # Step 1: Speech to text
        if isinstance(audio_input, str):
            transcription = await self.stt.transcribe_file(audio_input)
        else:
            transcription = await self.stt.transcribe_bytes(audio_input)

        logger.info(f"User said: {transcription.text}")

        # Step 2: Process command
        response_text = None
        if command_handler:
            response_text = await command_handler(transcription.text)
        else:
            response_text = f"I heard: {transcription.text}"

        logger.info(f"Response: {response_text}")

        # Step 3: Text to speech
        response_audio = None
        if respond_with_voice and response_text:
            synthesis = await self.tts.synthesize(response_text)
            response_audio = synthesis.audio_data

        return {
            "transcription": transcription.text,
            "language": transcription.language,
            "confidence": transcription.confidence,
            "response_text": response_text,
            "response_audio": response_audio,
            "timestamp": transcription.timestamp,
        }

    async def start_conversation_mode(self, callback: Callable[[str], str]):
        """
        Start continuous conversation mode.

        Args:
            callback: Async function that takes text and returns response
        """
        self.conversation_mode = True
        self.conversation_callback = callback

        logger.info("Conversation mode started")

    async def stop_conversation_mode(self):
        """Stop conversation mode."""
        self.conversation_mode = False
        self.conversation_callback = None

        logger.info("Conversation mode stopped")

    async def process_audio_stream(
        self, audio_stream: AsyncIterator[bytes], chunk_duration: float = 2.0
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process streaming audio in real-time.

        Args:
            audio_stream: Async iterator of audio chunks
            chunk_duration: Duration of each chunk in seconds

        Yields:
            Processing results
        """
        buffer = bytearray()

        async for chunk in audio_stream:
            buffer.extend(chunk)

            # Process when buffer is large enough
            # (simplified - real implementation would use VAD)
            if len(buffer) >= 16000 * chunk_duration * 2:  # 16kHz, 16-bit
                result = await self.voice_command(
                    bytes(buffer),
                    respond_with_voice=True,
                    command_handler=self.conversation_callback,
                )

                yield result

                buffer.clear()


# Example usage
async def main():
    """Example voice interface usage."""

    print("=" * 60)
    print("Voice Interface - STT + TTS Example")
    print("=" * 60)

    # Initialize voice interface
    print("\n🎤 Initializing voice interface...")
    voice = VoiceInterface(
        whisper_model=WhisperModel.BASE, tts_voice=TTSVoice.EN_US_FEMALE, language="en"
    )
    print("  ✅ Ready")

    # Example 1: Text to speech
    print("\n🔊 Example 1: Text-to-Speech")
    text = "Hello! I am Open-Sable, your autonomous AI assistant. How can I help you today?"
    print(f"  Synthesizing: '{text}'")

    synthesis = await voice.tts.synthesize(text, output_path="./test_output.wav")
    print(f"  ✅ Generated {synthesis.duration:.2f}s of audio")
    print(f"     Format: {synthesis.format.value}, Rate: {synthesis.sample_rate}Hz")

    # Example 2: Speech to text (if audio file exists)
    print("\n🎙️  Example 2: Speech-to-Text")
    test_audio = "./test_audio.wav"

    if os.path.exists(test_audio):
        print(f"  Transcribing: {test_audio}")
        transcription = await voice.stt.transcribe_file(test_audio)
        print(f"  ✅ Transcribed: '{transcription.text}'")
        print(
            f"     Language: {transcription.language}, Confidence: {transcription.confidence:.2%}"
        )
    else:
        print(f"  ⚠️  No test audio file found at {test_audio}")

    # Example 3: Voice command (round-trip)
    print("\n🔄 Example 3: Voice Command (STT → Processing → TTS)")

    async def simple_handler(text: str) -> str:
        """Simple command handler."""
        text_lower = text.lower()

        if "hello" in text_lower or "hi" in text_lower:
            return "Hello! Nice to meet you!"
        elif "time" in text_lower:
            from datetime import datetime

            return f"The current time is {datetime.now().strftime('%I:%M %p')}"
        elif "weather" in text_lower:
            return "I don't have weather data yet, but it's probably nice outside!"
        else:
            return f"You said: {text}"

    if os.path.exists(test_audio):
        result = await voice.voice_command(
            test_audio, respond_with_voice=True, command_handler=simple_handler
        )
        print(f"  📝 User: {result['transcription']}")
        print(f"  🤖 Agent: {result['response_text']}")
        print(f"  🔊 Audio response generated: {len(result['response_audio'])} bytes")

    print("\n✅ Voice interface examples complete!")
    print("\n💡 To use voice interface:")
    print("  • Install: pip install faster-whisper piper-tts")
    print("  • Record audio or provide audio file")
    print("  • Call voice.voice_command(audio_path)")
    print("  • Get transcription + voice response")


if __name__ == "__main__":
    asyncio.run(main())
