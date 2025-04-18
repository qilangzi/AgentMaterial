from fastapi import FastAPI
from utils.lifespan import lifespan

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Hello World"}

