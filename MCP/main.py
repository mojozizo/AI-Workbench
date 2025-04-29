from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

class ScoringItem(BaseModel):
    YearsAtCompany: int
    EmployeeSatisfaction: float
    Position: str
    Salary: int


with open("rfmodel.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/")
def read_root(item: ScoringItem):

    df = pd.DataFrame([item.model_dump_json().values()], columns=item.model_dump_json().keys())
    yhat = model.predict(df)

    return {
        "YearsAtCompany": item.YearsAtCompany,
        "EmployeeSatisfaction": item.EmployeeSatisfaction,
        "Position": item.Position,
        "Salary": item.Salary,
        "PredictedChurn": yhat[0]
    }



