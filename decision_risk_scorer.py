"""
Decision Risk Scoring Tool
AI in Business · Decision Risk Framework
Arivue Analytics × UTD Spring 2026

4 Primary Indicators:
  DQ  — Data Quality (Data Freshness)
  OV  — Output Variability
  CA  — Contextual Assumption Validity
  MS  — Model Stability
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import date


# ── Rubrics ───────────────────────────────────────────────────────────────────

RUBRICS = {
    "DQ": {
        "name": "Data Quality — Data Freshness",
        "description": "How recent and complete is the input data feeding the AI model at the time of decision?",
        "levels": {
            1: "≥95% fields populated, <1% anomalies, data updated within expected refresh window",
            2: "90–94% populated, 1–3% anomalies, minor staleness (within 1 refresh cycle)",
            3: "80–89% populated, 3–7% anomalies, or data 1–2 cycles overdue",
            4: "70–79% populated, 7–15% anomalies, or data significantly stale",
            5: "<70% populated, >15% anomalies, or data critically out of date",
        },
    },
    "OV": {
        "name": "Output Variability",
        "description": "How much has the AI output (score, forecast, recommendation) fluctuated over recent periods?",
        "levels": {
            1: "Output highly stable — coefficient of variation (CV) <5% over rolling window",
            2: "Minor fluctuation — CV 5–10%, consistent trend direction",
            3: "Moderate variability — CV 10–20%, some directional inconsistency",
            4: "High variability — CV 20–35%, frequent directional swings",
            5: "Severe instability — CV >35% or output reverses direction frequently",
        },
    },
    "CA": {
        "name": "Contextual Assumption Validity",
        "description": "Are the business conditions and environment under which the model was built still holding?",
        "levels": {
            1: "All assumptions valid — normal seasonality, no disruptions, no policy changes",
            2: "1 minor assumption flagged (e.g., small promotional activity, minor market shift)",
            3: "1–2 moderate assumptions flagged (e.g., active promotion, mild supply shift, new competitor)",
            4: "2–3 significant assumptions violated (e.g., major market event, regulatory change pending)",
            5: "Core assumptions invalid — major external shock, model context no longer applies",
        },
    },
    "MS": {
        "name": "Model Stability",
        "description": "Has the model drifted, been recently retrained, or shown changes in behaviour or performance metrics?",
        "levels": {
            1: "No drift detected, retrained within 30 days, performance metric delta <2%",
            2: "Minor drift signal, retrained 30–60 days ago, 2–5% metric delta",
            3: "Moderate drift, retrained 60–90 days ago, 5–10% metric delta",
            4: "Significant drift, retrained 90–180 days ago, 10–20% metric delta",
            5: "Severe drift or retrained >180 days ago, >20% delta, or no monitoring in place",
        },
    },
}

RISK_BANDS = [
    (2.0, "Low",      "🟢", "Proceed with standard monitoring"),
    (3.0, "Moderate", "🟡", "Proceed with documented caveats"),
    (4.0, "High",     "🟠", "Pause — escalate for review before acting"),
    (5.0, "Critical", "🔴", "Block — do not act without remediation"),
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class IndicatorScore:
    code: str
    score: int
    notes: str = ""

    def __post_init__(self):
        if self.score not in range(1, 6):
            raise ValueError(f"Score for {self.code} must be between 1 and 5, got {self.score}")


@dataclass
class DecisionRiskEvaluation:
    decision: str
    industry: str
    model_name: str
    evaluated_by: str
    eval_date: date
    scores: list[IndicatorScore]
    notes: str = ""

    def composite_score(self) -> float:
        return round(sum(s.score for s in self.scores) / len(self.scores), 2)

    def risk_band(self) -> tuple:
        cs = self.composite_score()
        for threshold, band, emoji, action in RISK_BANDS:
            if cs <= threshold:
                return (band, emoji, action)
        return RISK_BANDS[-1][1], RISK_BANDS[-1][2], RISK_BANDS[-1][3]


# ── Scoring functions ─────────────────────────────────────────────────────────

def score_indicator(code: str, score: int, notes: str = "") -> IndicatorScore:
    """Create a scored indicator. Score must be 1–5."""
    if code not in RUBRICS:
        raise ValueError(f"Unknown indicator code '{code}'. Valid codes: {list(RUBRICS.keys())}")
    return IndicatorScore(code=code, score=score, notes=notes)


def evaluate(
    decision: str,
    industry: str,
    model_name: str,
    evaluated_by: str,
    dq: int, ov: int, ca: int, ms: int,
    dq_notes: str = "", ov_notes: str = "",
    ca_notes: str = "", ms_notes: str = "",
    notes: str = "",
    eval_date: Optional[date] = None,
) -> DecisionRiskEvaluation:
    """
    Run a decision risk evaluation.

    Parameters
    ----------
    decision    : Description of the decision being evaluated
    industry    : Industry / domain
    model_name  : Name of the AI model or system
    evaluated_by: Name of evaluator
    dq, ov, ca, ms : Scores (1–5) for each indicator
    *_notes     : Optional notes per indicator
    notes       : Overall notes
    eval_date   : Date of evaluation (defaults to today)
    """
    return DecisionRiskEvaluation(
        decision=decision,
        industry=industry,
        model_name=model_name,
        evaluated_by=evaluated_by,
        eval_date=eval_date or date.today(),
        scores=[
            score_indicator("DQ", dq, dq_notes),
            score_indicator("OV", ov, ov_notes),
            score_indicator("CA", ca, ca_notes),
            score_indicator("MS", ms, ms_notes),
        ],
        notes=notes,
    )


# ── Display ───────────────────────────────────────────────────────────────────

def print_rubric(code: str):
    """Print the scoring rubric for a given indicator."""
    r = RUBRICS[code]
    print(f"\n{'─'*65}")
    print(f"  {r['name']} ({code})")
    print(f"  {r['description']}")
    print(f"{'─'*65}")
    for score, condition in r["levels"].items():
        print(f"  {score}  {condition}")
    print(f"{'─'*65}")


def print_all_rubrics():
    """Print rubrics for all 4 indicators."""
    for code in RUBRICS:
        print_rubric(code)


def print_report(ev: DecisionRiskEvaluation):
    """Print a formatted risk evaluation report."""
    cs = ev.composite_score()
    band, emoji, action = ev.risk_band()

    print(f"\n{'═'*65}")
    print(f"  DECISION RISK EVALUATION REPORT")
    print(f"{'═'*65}")
    print(f"  Decision        : {ev.decision}")
    print(f"  Industry        : {ev.industry}")
    print(f"  Model / System  : {ev.model_name}")
    print(f"  Evaluated By    : {ev.evaluated_by}")
    print(f"  Date            : {ev.eval_date}")
    print(f"{'─'*65}")
    print(f"  {'Indicator':<38} {'Code':<6} {'Score'}")
    print(f"  {'─'*38} {'─'*6} {'─'*5}")
    for s in ev.scores:
        name = RUBRICS[s.code]["name"]
        note = f"  ↳ {s.notes}" if s.notes else ""
        print(f"  {name:<38} {s.code:<6} {s.score}/5")
        if note:
            print(f"    {note}")
    print(f"{'─'*65}")
    print(f"  Composite Score : {cs:.2f} / 5.00")
    print(f"  Risk Band       : {emoji}  {band}")
    print(f"  Recommendation  : {action}")
    if ev.notes:
        print(f"{'─'*65}")
        print(f"  Notes: {ev.notes}")
    print(f"{'═'*65}\n")


# ── Interactive mode ──────────────────────────────────────────────────────────

def interactive():
    """Walk the user through scoring a decision interactively."""
    print("\n" + "═"*65)
    print("  Decision Risk Scoring Tool — Interactive Mode")
    print("  AI in Business · Arivue Analytics × UTD Spring 2026")
    print("═"*65)

    decision    = input("\n  Decision being evaluated: ").strip()
    industry    = input("  Industry / Domain: ").strip()
    model_name  = input("  AI Model / System: ").strip()
    evaluated_by = input("  Your name: ").strip()

    scores = {}
    score_notes = {}

    for code, rubric in RUBRICS.items():
        print(f"\n{'─'*65}")
        print(f"  {rubric['name']} ({code})")
        print(f"  {rubric['description']}\n")
        for lvl, cond in rubric["levels"].items():
            print(f"    {lvl} — {cond}")
        while True:
            try:
                val = int(input(f"\n  Enter score for {code} (1–5): ").strip())
                if 1 <= val <= 5:
                    scores[code] = val
                    break
                print("  Please enter a number between 1 and 5.")
            except ValueError:
                print("  Invalid input. Please enter a number.")
        note = input(f"  Optional note for {code} (press Enter to skip): ").strip()
        score_notes[code] = note

    overall_notes = input("\n  Overall notes (press Enter to skip): ").strip()

    ev = evaluate(
        decision=decision,
        industry=industry,
        model_name=model_name,
        evaluated_by=evaluated_by,
        dq=scores["DQ"], ov=scores["OV"],
        ca=scores["CA"], ms=scores["MS"],
        dq_notes=score_notes["DQ"], ov_notes=score_notes["OV"],
        ca_notes=score_notes["CA"], ms_notes=score_notes["MS"],
        notes=overall_notes,
    )
    print_report(ev)
    return ev


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive()
    else:
        # ── Example 1: Supply Chain ──────────────────────────────────────────
        ev1 = evaluate(
            decision="Trigger automated inventory reorder for Product SKU-4821",
            industry="Supply Chain",
            model_name="Demand Forecast Model v3.2",
            evaluated_by="UTD Research Team",
            dq=2, ov=3, ca=2, ms=1,
            dq_notes="POS data refreshed 6 hours ago, 92% fields populated",
            ov_notes="CV ~12% over past 4 weeks — moderate fluctuation",
            ca_notes="Minor seasonal shift flagged — approaching holiday period",
            ms_notes="Model retrained 18 days ago, no drift detected",
            notes="Standard weekly reorder evaluation.",
        )
        print_report(ev1)

        # ── Example 2: Healthcare ────────────────────────────────────────────
        ev2 = evaluate(
            decision="Escalate patient #10482 to ICU based on deterioration model output",
            industry="Healthcare",
            model_name="Patient Deterioration Risk Score v1.8",
            evaluated_by="UTD Research Team",
            dq=4, ov=2, ca=3, ms=3,
            dq_notes="Missing 3 of 10 required lab values — results pending",
            ov_notes="Risk score stable over last 6 hours",
            ca_notes="Patient recently transferred from different care unit — context shift",
            ms_notes="Model retrained 75 days ago, minor metric drift observed",
            notes="Critical decision — incomplete data increases risk significantly.",
        )
        print_report(ev2)

        # ── Example 3: Finance ───────────────────────────────────────────────
        ev3 = evaluate(
            decision="Approve credit limit increase for customer segment B",
            industry="Finance",
            model_name="Credit Risk Scoring Model v5.0",
            evaluated_by="UTD Research Team",
            dq=1, ov=1, ca=4, ms=2,
            dq_notes="Full data available, refreshed this morning",
            ov_notes="Scores highly stable — CV <3%",
            ca_notes="Interest rate environment has shifted significantly since model training",
            ms_notes="Model retrained 45 days ago, slight metric delta observed",
            notes="Macroeconomic context shift is the primary concern here.",
        )
        print_report(ev3)

        print("─"*65)
        print("  To run interactively:  python decision_risk_scorer.py --interactive")
        print("─"*65 + "\n")
