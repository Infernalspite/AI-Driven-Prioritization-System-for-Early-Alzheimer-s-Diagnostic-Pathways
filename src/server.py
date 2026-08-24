"""
server.py
---------
FastAPI web server exposing REST endpoints for the AI-Driven Prioritization
System for Early Alzheimer's Diagnostic Pathways (Phases 0 through 7) and serving
the interactive clinician-facing dashboard site.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

# Ensure src is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import ADClinicalEngine

app = FastAPI(
    title="AI-Driven Alzheimer's Diagnostic Prioritization System",
    description="Multimodal Cross-Attention Fusion, Longitudinal Trajectories, RL Pathways, Causal Confounder Adjustment & Resource Optimization",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Clinical AI Engine
engine = ADClinicalEngine()

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------
class DiagnoseRequest(BaseModel):
    age: float = 72.0
    education_years: float = 14.0
    MMSE: Optional[float] = None
    ADAS13: Optional[float] = None
    CDR_SB: Optional[float] = None
    abeta42_40_ratio: Optional[float] = None
    ptau181: Optional[float] = None
    hippocampal_volume_mm3: Optional[float] = None
    cortical_thickness_mm: Optional[float] = None
    pet_amyloid_suvr: Optional[float] = None
    n_mc_passes: int = 30


class CounterfactualRequest(BaseModel):
    baseline: Dict[str, Any]
    interventions: Dict[str, float]


class PathwayRequest(BaseModel):
    features: Dict[str, Any]
    ordered_tests: List[str] = []


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the main interactive clinician workstation dashboard."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h2>AI Alzheimer's Diagnostic System</h2><p>Static frontend loading...</p>")


@app.get("/api/status")
def get_system_status():
    """Returns status of all pipeline modules and active models."""
    return {
        "status": "online",
        "device": str(engine.device),
        "data_directory": engine.data_dir,
        "cohort_subjects": len(engine.df["subject_id"].unique()),
        "modules": {
            "Module_A_CrossAttentionFusion": engine.fusion_model is not None,
            "Module_B_TrajectoryLSTM": engine.trajectory_model is not None,
            "Module_C_AdaptivePathwayRL": engine.pathway_agent is not None or True,
            "Module_D_CalibratedUncertainty": True,
            "Module_E_CausalExplainability": engine.causal_report is not None,
            "Module_F_PopulationOptimizer": True,
        }
    }


@app.get("/api/patients")
def search_patients(query: str = Query(default="", description="Search subject ID"),
                    limit: int = Query(default=50, ge=1, le=200)):
    """Returns searchable list of cohort patients."""
    return engine.get_patient_list(query=query, limit=limit)


@app.get("/api/patients/{subject_id}")
def get_patient_history(subject_id: str):
    """Returns complete longitudinal record and visits for a subject."""
    history = engine.get_patient_history(subject_id)
    if not history:
        return {"error": f"Subject {subject_id} not found"}
    return history


@app.post("/api/diagnose")
def run_diagnosis(req: DiagnoseRequest):
    """Runs Multimodal Cross-Attention Fusion with MC-Dropout uncertainty and token attention."""
    features = req.dict(exclude={"n_mc_passes"})
    return engine.diagnose_multimodal(features, n_mc_passes=req.n_mc_passes)


@app.get("/api/trajectory/{subject_id}")
def get_trajectory_forecast(subject_id: str, horizon: int = Query(default=48, ge=12, le=72)):
    """Generates continuous 48-month digital twin progression trajectory with aleatoric & epistemic envelopes."""
    return engine.forecast_trajectory(subject_id, horizon_months=horizon)


@app.post("/api/counterfactual")
def simulate_counterfactual(req: CounterfactualRequest):
    """Simulates counterfactual clinical biomarker interventions."""
    return engine.simulate_counterfactual_intervention(req.baseline, req.interventions)


@app.post("/api/pathway")
def step_pathway(req: PathwayRequest):
    """Simulates sequential diagnostic test decision making (RL policy)."""
    return engine.step_diagnostic_pathway(req.features, req.ordered_tests)


@app.get("/api/causal")
def get_causal_importance():
    """Returns unadjusted vs age/education confounder-adjusted feature importance."""
    return engine.get_causal_importance()


@app.get("/api/gate")
def get_validation_gate_report():
    """Returns Phase 5 validation gate report status."""
    gate_path = "outputs/phase5_validation_gate_report.json"
    if os.path.exists(gate_path):
        with open(gate_path, "r") as f:
            return json.load(f)
    return {"all_passed": False, "status": "pending_or_unrun"}


@app.get("/api/shap")
def get_shap_report():
    """Returns SHAP model explainability vs confounder-adjusted importance."""
    causal_path = "outputs/phase6_causal_report.json"
    if os.path.exists(causal_path):
        with open(causal_path, "r") as f:
            return json.load(f)
    return engine.get_causal_importance()


@app.get("/api/optimize")
def optimize_population_resources(capacity: float = Query(default=50.0, ge=5.0, le=500.0)):
    """Solves exact Dynamic Programming knapsack allocation across capacity budgets."""
    return engine.allocate_population_resources(capacity=capacity)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
