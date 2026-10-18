from pathlib import Path

import threading
import asyncio

from fastapi import FastAPI
import uvicorn

import pygame

import numpy as np

from tangrams.dashboard import dashboard_router
from tangrams.robots import robots_router, robot_connections

from tangrams.wheel_controller import get_wheel_speeds

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
    

    direction = [ 0, 0 ]

    if keys[pygame.K_s]:
        direction[0] += 1
    if keys[pygame.K_f]:
        direction[0] -= 1
    
    if keys[pygame.K_e]:
        direction[1] -= 1
    if keys[pygame.K_d]:
        direction[1] += 1

    rot = 0
    if keys[pygame.K_j]:
        rot += 0.1
    if keys[pygame.K_k]:
        rot -= 0.1

    wheel_speeds = get_wheel_speeds(direction, rot)

    # ensure power never exceeds max of 1 for each wheel
    wheel_powers = wheel_speeds
    max_wheel_speed = np.max(np.abs(wheel_powers))
    if max_wheel_speed != 0:
        wheel_powers /= max_wheel_speed

    if 'smalltriangle1' in robot_connections:
        send_action = robot_connections['smalltriangle1'].send_action(wheel_powers.tolist())
        asyncio.run_coroutine_threadsafe(send_action, server_loop)
        

    pygame.display.flip()
    clock.tick(20)

pygame.quit()