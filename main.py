from fastapi import FastAPI, Request
from pydantic import BaseModel
from chatsapi import ChatsAPI

app = FastAPI()
chats_api = ChatsAPI([
    "Want to cancel a credit card.",
    "Want to know the account balance.",
    "Want to update the account details.",
    "Want to update the billing details."
])


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def root():
    return {"message": "ChatsAPI"}


@app.post("/chat")
async def chat(request: ChatRequest, http_request: Request):
    reply = chats_api.chat(request.message)
    return {"message": f"{reply}"}
