"""
Voice Support - Text-to-Speech and Speech-to-Text

Supports multiple providers:
- Local: pyttsx3 (TTS), Whisper (STT)
- Cloud: ElevenLabs, OpenAI TTS/Whisper
"""

import logging
import asyncio
from typing import Optional
from pathlib import Path
import tempfile
import os

logger = logging.getLogger(__name__)


class TTSEngine:
    """Text-to-Speech engine with multiple providers"""
    
    def __init__(self, config):
        self.config = config
        self.provider = config.tts_provider or "local"
        self.engine = None
        
    async def initialize(self):
        """Initialize TTS engine"""
        if self.provider == "local":
            await self._init_local()
        elif self.provider == "elevenlabs":
            await self._init_elevenlabs()
        elif self.provider == "openai":
            await self._init_openai()
        else:
            logger.error(f"Unknown TTS provider: {self.provider}")
            return False
        
        logger.info(f"TTS initialized with provider: {self.provider}")
        return True
    
    async def _init_local(self):
        """Initialize local pyttsx3"""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            
            # Configure voice
            voices = self.engine.getProperty('voices')
            if self.config.tts_voice_gender == "female" and len(voices) > 1:
                self.engine.setProperty('voice', voices[1].id)
            elif voices:
                self.engine.setProperty('voice', voices[0].id)
            
            # Set rate and volume
            self.engine.setProperty('rate', self.config.tts_rate or 150)
            self.engine.setProperty('volume', self.config.tts_volume or 0.9)
            
            logger.info("Local TTS engine initialized")
        except ImportError:
            logger.error("pyttsx3 not installed. Install with: pip install pyttsx3")
            raise
    
    async def _init_elevenlabs(self):
        """Initialize ElevenLabs TTS"""
        try:
            from elevenlabs import set_api_key
            set_api_key(self.config.elevenlabs_api_key)
            logger.info("ElevenLabs TTS initialized")
        except ImportError:
            logger.error("elevenlabs not installed. Install with: pip install elevenlabs")
            raise
    
    async def _init_openai(self):
        """Initialize OpenAI TTS"""
        try:
            from openai import OpenAI
            self.engine = OpenAI(api_key=self.config.openai_api_key)
            logger.info("OpenAI TTS initialized")
        except ImportError:
            logger.error("openai not installed. Install with: pip install openai")
            raise
    
    async def synthesize(self, text: str, output_file: Optional[str] = None) -> str:
        """
        Convert text to speech
        
        Args:
            text: Text to synthesize
            output_file: Optional output file path
            
        Returns:
            Path to generated audio file
        """
        if not output_file:
            output_file = tempfile.mktemp(suffix=".mp3")
        
        if self.provider == "local":
            return await self._synthesize_local(text, output_file)
        elif self.provider == "elevenlabs":
            return await self._synthesize_elevenlabs(text, output_file)
        elif self.provider == "openai":
            return await self._synthesize_openai(text, output_file)
    
    async def _synthesize_local(self, text: str, output_file: str) -> str:
        """Synthesize with pyttsx3"""
        # pyttsx3 runs in a loop, need to run in executor
        loop = asyncio.get_event_loop()
        
        def _save():
            self.engine.save_to_file(text, output_file)
            self.engine.runAndWait()
        
        await loop.run_in_executor(None, _save)
        
        logger.info(f"Local TTS saved to {output_file}")
        return output_file
    
    async def _synthesize_elevenlabs(self, text: str, output_file: str) -> str:
        """Synthesize with ElevenLabs"""
        from elevenlabs import generate, save
        
        audio = generate(
            text=text,
            voice=self.config.elevenlabs_voice_id or "21m00Tcm4TlvDq8ikWAM",
            model="eleven_monolingual_v1"
        )
        
        save(audio, output_file)
        logger.info(f"ElevenLabs TTS saved to {output_file}")
        return output_file
    
    async def _synthesize_openai(self, text: str, output_file: str) -> str:
        """Synthesize with OpenAI"""
        response = self.engine.audio.speech.create(
            model="tts-1",
            voice=self.config.openai_tts_voice or "alloy",
            input=text
        )
        
        response.stream_to_file(output_file)
        logger.info(f"OpenAI TTS saved to {output_file}")
        return output_file


class STTEngine:
    """Speech-to-Text engine with multiple providers"""
    
    def __init__(self, config):
        self.config = config
        self.provider = config.stt_provider or "local"
        self.engine = None
        
    async def initialize(self):
        """Initialize STT engine"""
        if self.provider == "local":
            await self._init_local()
        elif self.provider == "openai":
            await self._init_openai()
        else:
            logger.error(f"Unknown STT provider: {self.provider}")
            return False
        
        logger.info(f"STT initialized with provider: {self.provider}")
        return True
    
    async def _init_local(self):
        """Initialize local Whisper"""
        try:
            import whisper
            model_size = self.config.whisper_model_size or "base"
            self.engine = whisper.load_model(model_size)
            logger.info(f"Whisper model '{model_size}' loaded")
        except ImportError:
            logger.error("whisper not installed. Install with: pip install openai-whisper")
            raise
    
    async def _init_openai(self):
        """Initialize OpenAI Whisper API"""
        try:
            from openai import OpenAI
            self.engine = OpenAI(api_key=self.config.openai_api_key)
            logger.info("OpenAI Whisper API initialized")
        except ImportError:
            logger.error("openai not installed. Install with: pip install openai")
            raise
    
    async def transcribe(self, audio_file: str, language: Optional[str] = None) -> str:
        """
        Convert speech to text
        
        Args:
            audio_file: Path to audio file
            language: Optional language code (e.g., 'en')
            
        Returns:
            Transcribed text
        """
        if self.provider == "local":
            return await self._transcribe_local(audio_file, language)
        elif self.provider == "openai":
            return await self._transcribe_openai(audio_file, language)
    
    async def _transcribe_local(self, audio_file: str, language: Optional[str]) -> str:
        """Transcribe with local Whisper"""
        loop = asyncio.get_event_loop()
        
        def _transcribe():
            result = self.engine.transcribe(audio_file, language=language)
            return result["text"]
        
        text = await loop.run_in_executor(None, _transcribe)
        logger.info(f"Transcribed {audio_file}: {text[:50]}...")
        return text
    
    async def _transcribe_openai(self, audio_file: str, language: Optional[str]) -> str:
        """Transcribe with OpenAI Whisper API"""
        with open(audio_file, 'rb') as f:
            response = self.engine.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=language
            )
        
        text = response.text
        logger.info(f"Transcribed {audio_file}: {text[:50]}...")
        return text


class VoiceSkill:
    """Voice interaction skill combining TTS and STT"""
    
    def __init__(self, config):
        self.config = config
        self.tts = TTSEngine(config)
        self.stt = STTEngine(config)
        
    async def initialize(self):
        """Initialize voice engines"""
        tts_ok = await self.tts.initialize()
        stt_ok = await self.stt.initialize()
        
        if not (tts_ok and stt_ok):
            logger.error("Failed to initialize voice engines")
            return False
        
        logger.info("Voice skill initialized successfully")
        return True
    
    async def speak(self, text: str) -> str:
        """
        Convert text to speech and return audio file path
        
        Args:
            text: Text to speak
            
        Returns:
            Path to generated audio file
        """
        return await self.tts.synthesize(text)
    
    async def listen(self, audio_file: str, language: Optional[str] = None) -> str:
        """
        Convert speech to text
        
        Args:
            audio_file: Path to audio file
            language: Optional language code
            
        Returns:
            Transcribed text
        """
        return await self.stt.transcribe(audio_file, language)
    
    async def process_voice_message(self, audio_file: str, agent) -> tuple[str, str]:
        """
        Process voice message: transcribe -> agent -> speak
        
        Args:
            audio_file: Input audio file
            agent: Agent instance to process message
            
        Returns:
            Tuple of (transcribed_text, response_audio_file)
        """
        # Transcribe input
        user_text = await self.listen(audio_file)
        logger.info(f"User said: {user_text}")
        
        # Process through agent
        response_text = await agent.run(user_text)
        logger.info(f"Agent response: {response_text}")
        
        # Convert response to speech
        response_audio = await self.speak(response_text)
        
        return user_text, response_audio
    
    def cleanup_temp_files(self, *files):
        """Clean up temporary audio files"""
        for file_path in files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.debug(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up {file_path}: {e}")
