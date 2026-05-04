#!/usr/bin/env python3
"""Test that nano-sweaters responds as Yasmin Müller, not Sable."""

import asyncio
import json
import aiohttp


async def test_identity():
    async with aiohttp.ClientSession() as session:
        ws = await session.ws_connect("http://127.0.0.1:8789/ws")

        await ws.send_json({
            "type": "agents.chat",
            "profile": "nano-sweaters",
            "text": "Hi, who are you? What is your name?",
            "session_id": "test_identity_check",
            "user_id": "test_user",
        })

        print("⏳ Waiting for nano-sweaters response...")
        chunks = []
        while True:
            msg = await asyncio.wait_for(ws.receive(), timeout=120)
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                t = data.get("type", "")
                if t == "message.chunk":
                    chunks.append(data.get("text", ""))
                elif t == "message.done":
                    done_text = data.get("text", "")
                    if done_text:
                        chunks = [done_text]
                    break
                elif t == "error":
                    print(f"❌ ERROR: {data.get('text', '')}")
                    await ws.close()
                    return
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                print("❌ WebSocket closed unexpectedly")
                return

        await ws.close()

    reply = "".join(chunks).strip()
    print(f"\n📨 Response:\n{reply}\n")

    lower = reply.lower()
    if "yasmin" in lower or "müller" in lower or "metatex" in lower:
        print("✅ PASS,  nano-sweaters identified as Yasmin Müller")
    elif "sable" in lower:
        print("❌ FAIL,  still responding as Sable")
    else:
        print("⚠️  Identity unclear,  check response manually")


if __name__ == "__main__":
    asyncio.run(test_identity())
