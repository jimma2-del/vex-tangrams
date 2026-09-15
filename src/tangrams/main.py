from fastapi import FastAPI

from fastapi.staticfiles import StaticFiles

from starlette.responses import FileResponse 

app = FastAPI()

app.mount("/public", StaticFiles(directory="./src/tangrams/public"), name="public")


@app.get("/")
async def read_index():
    return FileResponse('./src/tangrams/index.html')