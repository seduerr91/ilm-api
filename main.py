from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from infill import INFILL


class Message(BaseModel):
    input: str
    output: str = None


app = FastAPI()
infill = INFILL()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/infill/")
async def fill_blank(message: Message):
    message.output = str(infill.infilling(message.input))
    return {"output": message.output}
