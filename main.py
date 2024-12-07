from fastapi import FastAPI, Request
from pydantic import BaseModel
from chatsapi.chatsapi import ChatsAPI

app = FastAPI()
chat = ChatsAPI()


@chat.trigger("Want to cancel a credit card.")
@chat.extract([("Credit card number", str, None)])
async def cancel_credit_card(chat_message: str, extracted: dict):
    print("Message:", chat_message)
    print("Extracted:", extracted)
    return f"Credit card cancellation process initiated."


@chat.trigger("Need help with account settings.")
@chat.extract([("Account ID", int, 12), ("Account type", str, "Savings")])
async def account_help(chat_message: str, extracted: dict):
    print("Message:", chat_message)
    print("Extracted:", extracted)
    return "Account help process initiated."


class RequestModel(BaseModel):
    message: str


@app.get("/")
async def root():
    return {"message": "ChatsAPI"}


@app.post("/chat")
async def message(request: RequestModel, http_request: Request):
    reply = await chat.run(request.message)
    return {"message": f"{reply}"}
