from pathlib import Path

import threading
import asyncio

from fastapi import FastAPI
import uvicorn

import pygame

from tangrams.dashboard import dashboard_router
from tangrams.robots import robots_router, robot_connections

app = FastAPI()

# serve static files
app.frontend("/", directory=Path(__file__).resolve().parent / 'client')

# handle user dashboard websocket connections
app.include_router(dashboard_router, prefix='/dashboard', tags=["Dashboard"])

# handle robot websocket connections
app.include_router(robots_router, prefix='/robots', tags=["Robots"])

server_loop = asyncio.new_event_loop()

def run_server():
    asyncio.set_event_loop(server_loop)
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8000))
    server_loop.run_until_complete(server.serve())

threading.Thread(target=run_server, daemon=True).start()

pygame.init()
window = pygame.display.set_mode((100,100))

clock = pygame.time.Clock()
FPS = 20

done = False

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
                
    keys = pygame.key.get_pressed()

    wheel_speeds = [ 0, 0, 0 ]
    
    if keys[pygame.K_d]:
        wheel_speeds[0] = 1

    if 'smalltriangle1' in robot_connections:
        asyncio.run_coroutine_threadsafe(robot_connections['smalltriangle1'].send_action(wheel_speeds), server_loop)
        

    pygame.display.flip()
    clock.tick(20)

pygame.quit()