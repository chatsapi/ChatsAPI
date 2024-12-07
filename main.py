import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from chatsapi.chatsapi import ChatsAPI

# Load environment variables from .env file
load_dotenv()

app = FastAPI()
chat = ChatsAPI(
    llm_type="gemini",
    llm_model="models/gemini-pro",
    llm_api_key=os.getenv("GOOGLE_API_KEY"),
)


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


@app.post("/one-chat")
async def one_message(request: RequestModel):
    reply_1 = await chat.query(request.message)
    return {"message": f"{reply_1}"}


@app.post("/chat")
async def message(request: RequestModel, response: Response, http_request: Request):
    # reply_1 = await chat.run(request.message)

    # conversation
    session_id = http_request.cookies.get("session_id")
    reply_2 = await chat.conversation(request.message, session_id)

    return {"message": f"{reply_2}"}


@app.post("/set-session")
def set_session(response: Response):
    session_id = chat.set_session()
    response.set_cookie(key="session_id", value=session_id)
    return {"message": "Session set"}


@app.post("/end-session")
def end_session(response: Response, http_request: Request):
    session_id = http_request.cookies.get("session_id")
    chat.end_session(session_id)
    response.delete_cookie("session_id")
    return {"message": "Session ended"}