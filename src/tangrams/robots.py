from fastapi import WebSocket, APIRouter, WebSocketDisconnect

import time

ROBOT_IDS = { 
    'smalltriangle1', 'smalltriangle2',
    'mediumtriangle',
    'largetriangle1', 'largetriangle2',
    'square',
    'parallelogram'
}

START_TIME = time.monotonic_ns()

robot_connections = {}

class Connection:
    def __init__(self, ws):
        self.ws = ws

        self.id = None

        self.event_handlers = {
            'id': self.on_id
        }

    async def on_event(self, event: dict):
        event_type = event.get('type')

        if event_type not in self.event_handlers: 
            return print("INVALID EVENT TYPE:", event_type)

        await self.event_handlers[event['type']](event)

    async def on_id(self, event: dict):
        sent_id = event.get('id')

        if sent_id not in ROBOT_IDS: 
            return print("INVALID ROBOT ID:", sent_id)

        if sent_id in robot_connections:
            return print(f"ROBOT '{sent_id}' ALREADY CONNECTED")

        self.id = sent_id
        robot_connections[self.id] = self

    async def on_disconnect(self):
        print("ROBOT WEBSOCKET DISCONNECTED")

        if self.id is not None:
            print(f"ROBOT '{self.id}' DISCONNECTED")
            del robot_connections[self.id]

    async def send_action(self, wheel_speeds):
        await self.ws.send_json({ 
            'time': time.monotonic_ns(),
            'wheel_speeds': wheel_speeds
        })

robots_router = APIRouter()

@robots_router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    print("ROBOT WEBSOCKET CONNECTED")

    connection = Connection(ws)
    
    try:
        async for event in ws.iter_json():
            await connection.on_event(event)
    except WebSocketDisconnect:
        await connection.on_disconnect()


