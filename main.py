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
    return {"Hello": "Javier"}

@app.post("/infill_sentence/")
async def infill_sentence(message: Message):
    message.output = str(infill.infilling_sentence(message.input))
    return {"output": message.output}

@app.post("/infill_word/")
async def infill_word(message: Message):
    message.output = str(infill.infilling_word(message.input))
    return {"output": message.output}

@app.post("/infill_ngram/")
async def infill_ngram(message: Message):
    message.output = str(infill.infilling_ngram(message.input))
    return {"output": message.output}