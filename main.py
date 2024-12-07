import os

from fastapi import FastAPI, Request
from pydantic import BaseModel
from chatsapi.chatsapi import ChatsAPI

app = FastAPI()
chat = ChatsAPI()


@chat.trigger("Want to cancel a credit card.")
@chat.extract([("card_number", "Credit card number (a 12 digit number)", str, None)])
async def cancel_credit_card(chat_message: str, extracted: dict):
    return {"message": chat_message, "extracted": extracted}


@chat.trigger("Need help with account settings.")
@chat.extract([
    ("account_number", "Account number (a nine digit number)", int, None),
    ("holder_name", "Account holder's name (a person name)", str, None)
])
async def account_help(chat_message: str, extracted: dict):
    return {"message": chat_message, "extracted": extracted}


class RequestModel(BaseModel):
    message: str


@app.get("/")
async def root():
    return {"message": "ChatsAPI"}


@app.post("/chat")
async def message(request: RequestModel, http_request: Request):
    reply = await chat.run(request.message)
    return {"message": f"{reply}"}
