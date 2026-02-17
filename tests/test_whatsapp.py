#!/usr/bin/env python3
"""
WhatsApp Integration Test
Tests sending messages via Venom Bot bridge
"""

import asyncio
import aiohttp

async def test_whatsapp():
    """Test WhatsApp bridge connectivity"""
    
    # Replace with your phone number (international format, no +)
    # Example: 5491234567890 for Argentina +54 9 11 2345-6789
    phone = input("Enter phone number to test (format: 5491234567890): ")
    
    if not phone:
        print("❌ Phone number required")
        return
    
    # Ensure proper WhatsApp format
    if '@c.us' not in phone:
        phone = f"{phone}@c.us"
    
    message = "🤖 Hello from SableCore! WhatsApp integration is working! ✅"
    
    print(f"\n📤 Sending message to {phone}...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:3333/send',
                json={
                    'phone': phone,
                    'message': message
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                result = await response.json()
                
                if response.status == 200:
                    print(f"✅ Message sent successfully!")
                    print(f"Response: {result}")
                else:
                    print(f"❌ Failed to send message: {result}")
                    
    except aiohttp.ClientConnectorError:
        print("❌ Cannot connect to bridge. Is it running on port 3333?")
    except asyncio.TimeoutError:
        print("❌ Request timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("SableCore WhatsApp Integration Test")
    print("=" * 60)
    asyncio.run(test_whatsapp())
