from fastapi import FastAPI, Request
from pydantic import BaseModel
from chatsapi import ChatsAPI

app = FastAPI()
chat = ChatsAPI()


@chat.trigger("Want to cancel a credit card.")
async def cancel_credit_card():
    return "Credit card cancellation process initiated."


@chat.trigger("Want to know the account balance.")
async def get_account_balance():
    return "Your account balance is $1,000."


# Define the request body model
class RequestModel(BaseModel):
    message: str


@app.get("/")
async def root():
    return {"message": "ChatsAPI"}


@app.post("/chat")
async def chat(request: RequestModel, http_request: Request):
    reply = await chat.run(request.message)
    return {"message": f"{reply}"}
