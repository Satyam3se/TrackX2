import asyncio, json, base64, sys
import websockets

async def test():
    uri = 'ws://localhost:8000/ws/live-video/'
    try:
        async with websockets.connect(uri) as ws:
            # Load a pre‑generated placeholder image for testing
            placeholder_path = r'C:/Users/acer/.gemini/antigravity-ide/brain/f7acdce0-185b-4321-9619-c032ae7a85b6/sample_placeholder_1789833927070.jpg'
            with open(placeholder_path, 'rb') as img_f:
                img_bytes = img_f.read()
            dummy = 'data:image/jpeg;base64,' + base64.b64encode(img_bytes).decode()
            await ws.send(json.dumps({'frame': dummy, 'scale': 1}))
            resp = await ws.recv()
            print('Response:', resp)
    except Exception as e:
        print('Error:', e)

asyncio.run(test())
