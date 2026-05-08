from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from decision_risk_scorer import evaluate

app = FastAPI()

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EvaluateRequest(BaseModel):
    decision: str
    industry: str
    model_name: str
    evaluated_by: str

    dq: int
    ov: int
    ca: int
    ms: int

    dq_notes: str = ""
    ov_notes: str = ""
    ca_notes: str = ""
    ms_notes: str = ""

    notes: str = ""

@app.post("/evaluate")
async def evaluate_api(data: EvaluateRequest):

    result = evaluate(
        decision=data.decision,
        industry=data.industry,
        model_name=data.model_name,
        evaluated_by=data.evaluated_by,

        dq=data.dq,
        ov=data.ov,
        ca=data.ca,
        ms=data.ms,

        dq_notes=data.dq_notes,
        ov_notes=data.ov_notes,
        ca_notes=data.ca_notes,
        ms_notes=data.ms_notes,

        notes=data.notes
    )

    return {
        "decision": result.decision,
        "industry": result.industry,
        "model": result.model_name,
        "composite_score": result.composite_score(),
        "risk_band": result.risk_band()[0],
        "recommendation": result.risk_band()[2],
        "scores": [
            {
                "code": s.code,
                "score": s.score,
                "notes": s.notes
            }
            for s in result.scores
        ]
    }