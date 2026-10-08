from pathlib import Path

from fastapi import FastAPI

from tangrams.dashboard import router as dashboard_router
from tangrams.robots import router as robots_router

app = FastAPI()

# serve static files
app.frontend("/", directory=Path(__file__).resolve().parent / 'client')

# handle user dashboard websocket connections
app.include_router(dashboard_router, prefix='/dashboard', tags=["Dashboard"])

# handle robot websocket connections
app.include_router(robots_router, prefix='/robots', tags=["Robots"])