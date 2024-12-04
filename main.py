from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def root():
    return {"message": "ChatsAPI"}


@app.post("/chat")
async def chat(request: ChatRequest, http_request: Request):
    return {"message": f"{request.message}"}
