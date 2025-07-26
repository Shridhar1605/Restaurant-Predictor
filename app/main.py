from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
from app.model_utils import predict_visitors

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your domain instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VisitorInput(BaseModel):
    visit_datetime: list
    reserve_visitors: list

@app.post("/predict")
def get_predictions(data: VisitorInput):
    result = predict_visitors(data.visit_datetime, data.reserve_visitors)
    print(type(result))
    return {"predictions":result}