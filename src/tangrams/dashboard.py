from fastapi import WebSocket, APIRouter

dashboard_router = APIRouter()

@dashboard_router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    
    async for event in ws.iter_json():
        print(event)
        #await ws.send_text(f"Message text was: {data}")
