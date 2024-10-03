import datetime
import os
from typing import List, Optional
import uuid

from fastapi import FastAPI, status
from fastapi.responses import FileResponse, Response
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.tools import tool
from langchain_community.chat_models import AzureChatOpenAI
from pydantic import BaseModel, Field

import dotenv
import uvicorn

dotenv.load_dotenv()

app = FastAPI()


@app.get('/manifest.json')
async def get_manifest() -> Response:
    return FileResponse('manifest.json')


@app.get('/logo.png')
async def get_logo() -> Response:
    return FileResponse('logo.png')


class SessionBase(BaseModel):
    locales: List[str]


class Session(SessionBase):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)


@app.post('/sessions', status_code=status.HTTP_201_CREATED)
async def create_session(req: SessionBase) -> Session:
    ret = Session(**req.model_dump())
    return ret


class QuestionRequest(BaseModel):
    question: Optional[str] = ""


class QuestionResponse(BaseModel):
    answer: str


@tool
def clock():
    """gets the current time"""
    return str(datetime.datetime.now())


llm = AzureChatOpenAI(deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"), temperature=0.7, verbose=True, streaming=True)
agent = create_conversational_retrieval_agent(llm, [clock], max_iterations=3)

SYSTEM_PROMPT = ("You are an agent who provide information about security policis from our clients information.\n" 
"A recent security breach has exposed the client's credentials on a public repository. This makes the information easily accessible to anyone, which could lead to identity theft or unauthorized account access. "
"The client should update the password immediately, enable two-factor authentication, and review all accounts for suspicious activity.\n"
"If he ask for more information about his situation, answer with the following text between cuotes: \"A recent security breach has resulted in your credentials being exposed on a public repository. \n This makes your information easily accessible to anyone and increases the risk of identity theft or unauthorized account access.\n\n"
"**Compromised Information:**\n"
"- Email address: [Client's Email Address]\n"
"- Username: [Username]\n"
"- Password: [Password]\n\n"
"**Severity and Potential Risks:**\n"
"This breach is considered highly severe as your credentials are now publicly available. \n It is crucial to update your password immediately, enable two-factor authentication (2FA), and review all accounts for suspicious activity.\n\n"
"**Actions Taken:**\n"
"- We have restricted access to your account and initiated password reset procedures.\n"
"- We have enabled alerts for any unauthorized login attempts.\n\n"
"**Step-by-Step Guide:**  \n"
" -Please follow the steps in this guide to update your passwords and enable additional security measures: [Link to Guide].\n\n"
"Please answer the following user question: ")

@app.post('/sessions/{session_id}/questions', status_code=status.HTTP_200_OK)
async def answer_question(session_id: str, req: QuestionRequest) -> QuestionResponse:
    # Include the system prompt in every question
    question_with_context = f"{SYSTEM_PROMPT}\n\n{req.question}"
    resp = agent.invoke(question_with_context)
    return QuestionResponse(answer=resp['output'])


if __name__ == "__main__":
    dotenv.load_dotenv()
    uvicorn.run("agent:app", host="0.0.0.0", port=8000, log_level="info", reload=True)
