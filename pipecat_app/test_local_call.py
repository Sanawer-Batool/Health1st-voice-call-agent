"""
Simulates what Twilio sends over the Media Streams WebSocket, so you can
test server.py's /ws handler locally — no real phone call, no webhook
configuration needed. Useful right now while waiting on your team lead,
and useful later any time you want to test a change quickly.

This does NOT test real audio processing (we send silence/dummy bytes,
not your actual voice) — it tests that the connection handshake, the
'start' event parsing, and the pipeline setup all work without crashing.

Run the real server first in one terminal:
    uvicorn server:app --port 8000

Then in a separate terminal:
    python test_local_call.py
"""
import asyncio
import base64
import json
import websockets

WS_URL = "ws://localhost:8000/ws"


async def simulate_call():
    print(f"Connecting to {WS_URL} ...")
    async with websockets.connect(WS_URL) as ws:
        print("Connected. Sending Twilio 'connected' event...")
        await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))

        print("Sending Twilio 'start' event (this is what your server waits for)...")
        start_event = {
            "event": "start",
            "sequenceNumber": "1",
            "start": {
                "streamSid": "MZtest1234567890",
                "callSid": "CAtest1234567890",
                "accountSid": "ACtest1234567890",
                "tracks": ["inbound"],
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
            },
            "streamSid": "MZtest1234567890",
        }
        await ws.send(json.dumps(start_event))

        print("Server should now have parsed stream_sid/call_sid and set up the pipeline.")
        print("Sending a few fake audio chunks (silence, mu-law encoded)...")

        # mu-law silence byte is 0xFF; send a few small fake media frames
        silence_chunk = base64.b64encode(bytes([0xFF] * 160)).decode()  # 160 bytes = 20ms @ 8kHz
        for i in range(5):
            media_event = {
                "event": "media",
                "sequenceNumber": str(i + 2),
                "media": {
                    "track": "inbound",
                    "chunk": str(i + 1),
                    "timestamp": str(i * 20),
                    "payload": silence_chunk,
                },
                "streamSid": "MZtest1234567890",
            }
            await ws.send(json.dumps(media_event))
            await asyncio.sleep(0.02)

        print("Sent 5 fake audio chunks. Listening briefly for any response frames...")
        try:
            for _ in range(3):
                response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                print(f"Received from server: {response[:100] if isinstance(response, str) else f'<{len(response)} bytes binary>'}")
        except asyncio.TimeoutError:
            print("No response within timeout — fine for a silence-only test, nothing to echo back meaningfully.")

        print("Sending Twilio 'stop' event to end the simulated call...")
        await ws.send(json.dumps({"event": "stop", "streamSid": "MZtest1234567890"}))

        await asyncio.sleep(0.5)
        print("Test complete — if no exceptions were raised above, the connection handling and pipeline setup work correctly.")


if __name__ == "__main__":
    asyncio.run(simulate_call())
