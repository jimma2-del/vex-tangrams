from fastapi import WebSocket, APIRouter

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    
    async for event in ws.iter_json():
        print(event)
        #await ws.send_text(f"Message text was: {data}")
