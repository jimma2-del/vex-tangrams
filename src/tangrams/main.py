from pathlib import Path

from fastapi import FastAPI, WebSocket

app = FastAPI()

# serve static files
app.frontend("/", directory=Path(__file__).resolve().parent / 'client')

# handle user dashboard websocket connections

@app.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        data = await websocket.receive_text()
        print(data)
        #await websocket.send_text(f"Message text was: {data}")


# handle robot websocket connections

@app.websocket("/ws/robots")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")