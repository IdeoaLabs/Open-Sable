"""
Open-Sable Voice Call Handler

Handles voice calls via Twilio, WebRTC, or other telephony providers.
Integrates STT/TTS for voice conversation capabilities.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from opensable.core.agent import SableAgent
from opensable.core.config import Config
from opensable.core.session_manager import SessionManager
from opensable.core.voice import VoiceManager

logger = logging.getLogger(__name__)


class VoiceCallHandler:
    """Voice call interface for Open-Sable"""

    def __init__(self, config: Config):
        self.config = config
        self.agent = SableAgent(config)
        self.session_manager = SessionManager()
        self.voice_manager = VoiceManager(config)

        # Twilio settings (if available)
        self.twilio_enabled = getattr(config, "twilio_enabled", False)
        self.twilio_account_sid = getattr(config, "twilio_account_sid", None)
        self.twilio_auth_token = getattr(config, "twilio_auth_token", None)
        self.twilio_phone_number = getattr(config, "twilio_phone_number", None)

        # Call settings
        self.greeting_message = getattr(
            config, "voice_greeting", "Hello, this is Open-Sable AI assistant. How can I help you?"
        )
        self.goodbye_message = getattr(config, "voice_goodbye", "Thank you for calling. Goodbye!")

        # Active calls
        self.active_calls: Dict[str, Dict[str, Any]] = {}

    async def handle_incoming_call(self, call_sid: str, from_number: str) -> str:
        """Handle incoming voice call (returns TwiML)"""
        logger.info(f"Incoming call from {from_number}, SID: {call_sid}")

        # Get or create session for this caller
        session = self.session_manager.get_or_create_session(channel="voice", user_id=from_number)

        # Store call info
        self.active_calls[call_sid] = {
            "from": from_number,
            "session_id": session.session_id,
            "start_time": datetime.utcnow(),
            "state": "active",
        }

        # Generate greeting audio
        greeting_audio = await self.voice_manager.text_to_speech(self.greeting_message)

        # Return TwiML for Twilio
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">{self.greeting_message}</Say>
    <Gather input="speech" action="/voice/process" method="POST" timeout="3" speechTimeout="auto">
        <Say>Please tell me how I can help you.</Say>
    </Gather>
    <Say>I didn't hear anything. Please call back.</Say>
    <Hangup/>
</Response>"""

        return twiml

    async def process_speech(self, call_sid: str, speech_text: str) -> str:
        """Process speech input from call"""
        if call_sid not in self.active_calls:
            logger.warning(f"Unknown call SID: {call_sid}")
            return self._generate_error_twiml()

        call_info = self.active_calls[call_sid]
        session = self.session_manager.get_session(call_info["session_id"])

        if not session:
            logger.error(f"Session not found: {call_info['session_id']}")
            return self._generate_error_twiml()

        logger.info(f"Processing speech: {speech_text}")

        try:
            # Process through agent
            response = await self.agent.run(speech_text, session)

            # Generate TwiML with response
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">{self._sanitize_text(response)}</Say>
    <Gather input="speech" action="/voice/process" method="POST" timeout="3" speechTimeout="auto">
        <Say>Is there anything else I can help you with?</Say>
    </Gather>
    <Say>{self.goodbye_message}</Say>
    <Hangup/>
</Response>"""

            return twiml

        except Exception as e:
            logger.error(f"Error processing speech: {e}", exc_info=True)
            return self._generate_error_twiml()

    async def handle_hangup(self, call_sid: str):
        """Handle call hangup"""
        if call_sid in self.active_calls:
            call_info = self.active_calls[call_sid]
            duration = (datetime.utcnow() - call_info["start_time"]).total_seconds()

            logger.info(f"Call ended: {call_sid}, duration: {duration}s")

            del self.active_calls[call_sid]

    def _sanitize_text(self, text: str) -> str:
        """Sanitize text for TTS (remove special characters)"""
        # Remove Markdown formatting
        text = text.replace("**", "")
        text = text.replace("*", "")
        text = text.replace("_", "")
        text = text.replace("#", "")

        # Remove URLs
        import re

        text = re.sub(r"http[s]?://\S+", "link", text)

        # Escape XML special characters
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&apos;")

        return text

    def _generate_error_twiml(self) -> str:
        """Generate error TwiML"""
        return """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">I'm sorry, I encountered an error. Please try again later.</Say>
    <Hangup/>
</Response>"""

    async def make_outbound_call(self, to_number: str, message: str) -> bool:
        """Make outbound voice call (requires Twilio)"""
        if not self.twilio_enabled or not self.twilio_account_sid:
            logger.error("Twilio not configured for outbound calls")
            return False

        try:
            from twilio.rest import Client

            client = Client(self.twilio_account_sid, self.twilio_auth_token)

            # Create TwiML for message
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">{self._sanitize_text(message)}</Say>
    <Hangup/>
</Response>"""

            # Make call
            call = client.calls.create(to=to_number, from_=self.twilio_phone_number, twiml=twiml)

            logger.info(f"Outbound call initiated: {call.sid} to {to_number}")
            return True

        except Exception as e:
            logger.error(f"Error making outbound call: {e}", exc_info=True)
            return False


class VoiceWebRTC:
    """WebRTC voice interface for browser-based calls"""

    def __init__(self, config: Config):
        self.config = config
        self.agent = SableAgent(config)
        self.voice_manager = VoiceManager(config)
        self.session_manager = SessionManager()

        # WebRTC connections
        self.connections: Dict[str, Any] = {}

    async def handle_audio_stream(self, connection_id: str, audio_data: bytes, user_id: str):
        """Handle incoming audio stream from WebRTC"""
        try:
            # Get or create session
            session = self.session_manager.get_or_create_session(
                channel="voice_webrtc", user_id=user_id
            )

            # Convert audio to text
            text = await self.voice_manager.speech_to_text(audio_data)

            if not text:
                logger.warning("No speech detected in audio stream")
                return None

            logger.info(f"Transcribed: {text}")

            # Process through agent
            response = await self.agent.run(text, session)

            # Convert response to speech
            response_audio = await self.voice_manager.text_to_speech(response)

            return response_audio

        except Exception as e:
            logger.error(f"Error processing audio stream: {e}", exc_info=True)
            return None


# FastAPI integration for voice endpoints
try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import PlainTextResponse
except ImportError:
    FastAPI = Request = Response = None
    PlainTextResponse = None


def create_voice_app(config: Config) -> FastAPI:
    """Create FastAPI app for voice endpoints"""
    app = FastAPI(title="Open-Sable Voice API", version="0.2.0")
    handler = VoiceCallHandler(config)

    @app.post("/voice/incoming")
    async def incoming_call(request: Request):
        """Handle incoming Twilio call"""
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        from_number = form_data.get("From")

        twiml = await handler.handle_incoming_call(call_sid, from_number)

        return Response(content=twiml, media_type="application/xml")

    @app.post("/voice/process")
    async def process_speech(request: Request):
        """Process speech from call"""
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        speech_result = form_data.get("SpeechResult", "")

        twiml = await handler.process_speech(call_sid, speech_result)

        return Response(content=twiml, media_type="application/xml")

    @app.post("/voice/hangup")
    async def hangup(request: Request):
        """Handle call hangup"""
        form_data = await request.form()
        call_sid = form_data.get("CallSid")

        await handler.handle_hangup(call_sid)

        return PlainTextResponse("OK")

    return app


if __name__ == "__main__":
    from opensable.core.config import load_config
    import uvicorn

    config = load_config()
    app = create_voice_app(config)

    uvicorn.run(app, host="0.0.0.0", port=8001)
