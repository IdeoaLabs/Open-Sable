"""
Open-Sable Email Interface

Monitors mailbox and processes emails through Open-Sable agent.
Supports IMAP/SMTP with auto-reply capabilities.
"""

import asyncio
import logging
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
from typing import Optional, List, Dict, Any
import imaplib
import smtplib
import ssl
from datetime import datetime

from opensable.core.agent import SableAgent
from opensable.core.config import Config
from opensable.core.session_manager import SessionManager
from opensable.core.commands import CommandHandler

logger = logging.getLogger(__name__)


class EmailInterface:
    """Email bot interface for Open-Sable"""
    
    def __init__(self, config: Config):
        self.config = config
        self.agent = SableAgent(config)
        self.session_manager = SessionManager()
        self.command_handler = CommandHandler(self.session_manager)
        
        # Email settings
        self.imap_server = getattr(config, 'email_imap_server', 'imap.gmail.com')
        self.imap_port = getattr(config, 'email_imap_port', 993)
        self.smtp_server = getattr(config, 'email_smtp_server', 'smtp.gmail.com')
        self.smtp_port = getattr(config, 'email_smtp_port', 587)
        self.email_address = getattr(config, 'email_address', None)
        self.email_password = getattr(config, 'email_password', None)
        
        # Auto-reply settings
        self.auto_reply = getattr(config, 'email_auto_reply', False)
        self.reply_signature = getattr(config, 'email_signature', 
                                       '\n\n---\nThis is an automated response from Open-Sable AI Assistant.')
        
        # Monitoring
        self.check_interval = getattr(config, 'email_check_interval', 60)  # seconds
        self.running = False
        self.processed_uids: set = set()
        
    def decode_header_value(self, header_value: str) -> str:
        """Decode email header value"""
        if not header_value:
            return ""
        
        decoded_parts = []
        for part, encoding in decode_header(header_value):
            if isinstance(part, bytes):
                decoded_parts.append(part.decode(encoding or 'utf-8', errors='ignore'))
            else:
                decoded_parts.append(part)
        
        return ' '.join(decoded_parts)
    
    def extract_email_address(self, address_str: str) -> str:
        """Extract email address from header"""
        if '<' in address_str and '>' in address_str:
            start = address_str.index('<') + 1
            end = address_str.index('>')
            return address_str[start:end].strip()
        return address_str.strip()
    
    def get_email_body(self, msg: email.message.Message) -> str:
        """Extract plain text body from email"""
        body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    try:
                        payload = part.get_payload(decode=True)
                        charset = part.get_content_charset() or 'utf-8'
                        body += payload.decode(charset, errors='ignore')
                    except Exception as e:
                        logger.error(f"Error decoding email part: {e}")
        else:
            try:
                payload = msg.get_payload(decode=True)
                charset = msg.get_content_charset() or 'utf-8'
                body = payload.decode(charset, errors='ignore')
            except Exception as e:
                logger.error(f"Error decoding email body: {e}")
        
        return body.strip()
    
    async def fetch_unread_emails(self) -> List[Dict[str, Any]]:
        """Fetch unread emails from mailbox"""
        if not self.email_address or not self.email_password:
            logger.warning("Email credentials not configured")
            return []
        
        emails = []
        
        try:
            # Connect to IMAP
            context = ssl.create_default_context()
            mail = imaplib.IMAP4_SSL(self.imap_server, self.imap_port, ssl_context=context)
            mail.login(self.email_address, self.email_password)
            mail.select('INBOX')
            
            # Search for unread emails
            status, messages = mail.search(None, 'UNSEEN')
            
            if status != 'OK':
                logger.error("Failed to search emails")
                return emails
            
            email_ids = messages[0].split()
            
            for email_id in email_ids:
                uid = email_id.decode()
                
                # Skip if already processed
                if uid in self.processed_uids:
                    continue
                
                # Fetch email
                status, msg_data = mail.fetch(email_id, '(RFC822)')
                
                if status != 'OK':
                    continue
                
                # Parse email
                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)
                
                # Extract details
                subject = self.decode_header_value(msg.get('Subject', ''))
                from_address = self.extract_email_address(self.decode_header_value(msg.get('From', '')))
                to_address = self.decode_header_value(msg.get('To', ''))
                date_str = msg.get('Date', '')
                body = self.get_email_body(msg)
                
                emails.append({
                    'uid': uid,
                    'subject': subject,
                    'from': from_address,
                    'to': to_address,
                    'date': date_str,
                    'body': body
                })
                
                self.processed_uids.add(uid)
            
            mail.close()
            mail.logout()
            
        except Exception as e:
            logger.error(f"Error fetching emails: {e}", exc_info=True)
        
        return emails
    
    async def send_email(self, to_address: str, subject: str, body: str, 
                        in_reply_to: Optional[str] = None) -> bool:
        """Send email via SMTP"""
        if not self.email_address or not self.email_password:
            logger.error("Email credentials not configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_address
            msg['To'] = to_address
            msg['Subject'] = subject
            
            if in_reply_to:
                msg['In-Reply-To'] = in_reply_to
                msg['References'] = in_reply_to
            
            # Add body with signature
            full_body = body + self.reply_signature
            msg.attach(MIMEText(full_body, 'plain'))
            
            # Connect and send
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.email_address, self.email_password)
                server.send_message(msg)
            
            logger.info(f"Sent email to {to_address}: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}", exc_info=True)
            return False
    
    async def process_email(self, email_data: Dict[str, Any]):
        """Process incoming email through agent"""
        try:
            from_address = email_data['from']
            subject = email_data['subject']
            body = email_data['body']
            
            logger.info(f"Processing email from {from_address}: {subject}")
            
            # Get or create session for this email address
            session = self.session_manager.get_or_create_session(
                channel='email',
                user_id=from_address
            )
            
            # Build context message
            context = f"Email Subject: {subject}\n\n{body}"
            
            # Check for commands
            if body.strip().startswith('/'):
                result = self.command_handler.handle_command(
                    body.strip(), 
                    session,
                    is_admin=False  # Email users are not admin by default
                )
                
                if result and self.auto_reply:
                    # Send command result as reply
                    reply_subject = f"Re: {subject}"
                    await self.send_email(from_address, reply_subject, result.message)
                
                return
            
            # Process through agent
            response = await self.agent.run(context, session)
            
            # Send auto-reply if enabled
            if self.auto_reply:
                reply_subject = f"Re: {subject}"
                await self.send_email(from_address, reply_subject, response)
                logger.info(f"Sent auto-reply to {from_address}")
            
        except Exception as e:
            logger.error(f"Error processing email: {e}", exc_info=True)
    
    async def monitor_mailbox(self):
        """Monitor mailbox for new emails"""
        logger.info(f"Starting email monitoring (checking every {self.check_interval}s)")
        
        while self.running:
            try:
                # Fetch unread emails
                emails = await self.fetch_unread_emails()
                
                # Process each email
                for email_data in emails:
                    await self.process_email(email_data)
                
                # Wait before next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in mailbox monitoring: {e}", exc_info=True)
                await asyncio.sleep(self.check_interval)
    
    async def start(self):
        """Start email interface"""
        if not self.email_address or not self.email_password:
            logger.error("Email credentials not configured. Skipping email interface.")
            return
        
        logger.info(f"Starting Email interface for {self.email_address}")
        logger.info(f"IMAP: {self.imap_server}:{self.imap_port}")
        logger.info(f"SMTP: {self.smtp_server}:{self.smtp_port}")
        logger.info(f"Auto-reply: {self.auto_reply}")
        
        self.running = True
        
        try:
            await self.monitor_mailbox()
        except KeyboardInterrupt:
            logger.info("Email interface stopped by user")
        except Exception as e:
            logger.error(f"Email interface error: {e}", exc_info=True)
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop email interface"""
        logger.info("Stopping email interface")
        self.running = False


# Standalone execution
if __name__ == "__main__":
    from opensable.core.config import load_config
    
    config = load_config()
    bot = EmailInterface(config)
    
    asyncio.run(bot.start())
