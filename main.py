# --- START OF FULL CORRECTED beespector_api/main.py ---
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

# --- Pydantic Models ---

class ExplicitAliasFeaturesModel(BaseModel):
    age: int
    hours_per_week: int = Field(serialization_alias="hours-per-week", validation_alias="hours-per-week")
    klass: str = Field(serialization_alias="class", validation_alias="class")
    education: str
    occupation: str
    model_config = ConfigDict(populate_by_name=True)

class InitialDataPoint(BaseModel): # For GET /api/datapoints response
    id: int
    x1: float
    x2: float
    true_label: int
    features: ExplicitAliasFeaturesModel
    pred_label: int # Base model's initial prediction
    pred_prob: float  # Base model's initial prediction probability
    mitigated_pred_label: int # Mitigated model's initial prediction
    mitigated_pred_prob: float  # Mitigated model's initial prediction probability
    model_config = ConfigDict(populate_by_name=True)

class EvaluatedPointPrediction(BaseModel):
    pred_label: int
    pred_prob: float

class EvaluatedPointData(BaseModel): # For PUT /api/datapoints/{id}/evaluate response
    id: int
    x1: float
    x2: float
    features: ExplicitAliasFeaturesModel
    true_label: int
    base_model_prediction: EvaluatedPointPrediction
    mitigated_model_prediction: EvaluatedPointPrediction
    model_config = ConfigDict(populate_by_name=True)

# --- Mock Database (Base features only, predictions will be calculated) ---
MOCK_DATAPOINTS_DB_INTERNAL_BASE_FEATURES: List[Dict[str, Any]] = [
    {"id": 1, "x1": 10.0, "x2": 20.0, "true_label": 0, "features": {"age": 25, "hours_per_week": 40, "klass": "A", "education": "BSc", "occupation": "Dev"}},
    {"id": 2, "x1": 15.0, "x2": 25.0, "true_label": 1, "features": {"age": 35, "hours_per_week": 45, "klass": "B", "education": "MSc", "occupation": "Eng"}},
    {"id": 3, "x1": 12.5, "x2": 18.0, "true_label": 1, "features": {"age": 42, "hours_per_week": 35, "klass": "A", "education": "PhD", "occupation": "Sci"}},
    {"id": 4, "x1": 18.0, "x2": 22.0, "true_label": 0, "features": {"age": 29, "hours_per_week": 50, "klass": "C", "education": "BSc", "occupation": "Mgr"}},
]

# --- Prediction Logic Helper (This was missing from your version) ---
def _calculate_mock_predictions(x1: float, x2: float, features_instance: ExplicitAliasFeaturesModel) -> tuple[EvaluatedPointPrediction, EvaluatedPointPrediction]:
    print(f"--- Backend: _calculate_mock_predictions called ---")
    print(f"Input x1: {x1}, type: {type(x1)}")
    print(f"Input x2: {x2}, type: {type(x2)}")
    print(f"Input features (attributes): age={features_instance.age}, hpw={features_instance.hours_per_week}, class={features_instance.klass}, edu={features_instance.education}, occ={features_instance.occupation}")

    current_age = features_instance.age
    current_hpw = features_instance.hours_per_week # Pythonic attribute name, correctly accessed
    current_klass = features_instance.klass       # Pythonic attribute name, correctly accessed

    base_pred_prob = (x1 + x2) / 100 + (current_age / 200)
    base_pred_prob = min(max(base_pred_prob, 0.01), 0.99)
    base_pred_label = 1 if base_pred_prob > 0.5 else 0

    mitigated_pred_prob = base_pred_prob * 0.8
    if current_klass == "A":
        mitigated_pred_prob = base_pred_prob * 1.1
    mitigated_pred_prob = min(max(mitigated_pred_prob, 0.01), 0.99)
    mitigated_pred_label = 1 if mitigated_pred_prob > 0.5 else 0

    base_prediction = EvaluatedPointPrediction(pred_label=base_pred_label, pred_prob=round(base_pred_prob, 3))
    mitigated_prediction = EvaluatedPointPrediction(pred_label=mitigated_pred_label, pred_prob=round(mitigated_pred_prob, 3))

    print(f"Calculated Base: Label={base_prediction.pred_label}, Prob={base_prediction.pred_prob}")
    print(f"Calculated Mitigated: Label={mitigated_prediction.pred_label}, Prob={mitigated_prediction.pred_prob}")
    print(f"--- Backend: _calculate_mock_predictions finished ---")
    return base_prediction, mitigated_prediction

# --- FastAPI App and CORS ---
app = FastAPI(title="Beespector API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Accept", "Authorization", "Content-Type", "X-Requested-With"],
)

# --- API Endpoints ---
@app.get("/api/datapoints", response_model=Dict[str, List[InitialDataPoint]])
async def get_all_datapoints():
    response_data = []
    for p_base_features_dict in MOCK_DATAPOINTS_DB_INTERNAL_BASE_FEATURES:
        # Create ExplicitAliasFeaturesModel instance from the nested dict in MOCK_DATAPOINTS_DB_INTERNAL_BASE_FEATURES
        # Pydantic's populate_by_name and validation_alias handle mapping from dict keys
        # (e.g. "hours_per_week") to model attributes (e.g. hours_per_week) IF the dict keys were the aliases.
        # Since MOCK_DATAPOINTS_DB_INTERNAL_BASE_FEATURES features dict uses pythonic names, direct unpacking works.
        features_model_instance = ExplicitAliasFeaturesModel(**p_base_features_dict["features"])
        
        base_pred, mitigated_pred = _calculate_mock_predictions( # Call the helper
            p_base_features_dict["x1"], p_base_features_dict["x2"], features_model_instance
        )
        
        response_data.append(
            InitialDataPoint(
                id=p_base_features_dict["id"],
                x1=p_base_features_dict["x1"],
                x2=p_base_features_dict["x2"],
                true_label=p_base_features_dict["true_label"],
                features=features_model_instance, # This is an ExplicitAliasFeaturesModel instance
                pred_label=base_pred.pred_label,
                pred_prob=base_pred.pred_prob,
                mitigated_pred_label=mitigated_pred.pred_label,
                mitigated_pred_prob=mitigated_pred.pred_prob
            )
        )
    return {"data": response_data}

class UpdatePointPayloadFeatures(BaseModel): # For parsing features from PUT request body
    age: int
    hours_per_week: int = Field(validation_alias="hours-per-week") # Expects "hours-per-week" from JSON
    klass: str = Field(validation_alias="class")                   # Expects "class" from JSON
    education: str
    occupation: str
    model_config = ConfigDict(populate_by_name=True)

class UpdatePointPayload(BaseModel): # For the whole PUT request body
    x1: float
    x2: float
    features: UpdatePointPayloadFeatures
    model_config = ConfigDict(populate_by_name=True)

@app.put("/api/datapoints/{point_id}/evaluate", response_model=EvaluatedPointData)
async def evaluate_modified_point(point_id: int, payload: UpdatePointPayload):
    original_point_base_info = next((p for p in MOCK_DATAPOINTS_DB_INTERNAL_BASE_FEATURES if p["id"] == point_id), None)
    if not original_point_base_info:
        raise HTTPException(status_code=404, detail="Point not found")

    # payload.features is an instance of UpdatePointPayloadFeatures.
    # Its attributes (e.g. payload.features.hours_per_week) are Pythonic names,
    # correctly parsed from JSON thanks to validation_alias.
    # We need to pass an ExplicitAliasFeaturesModel instance to _calculate_mock_predictions.
    current_features_for_calc = ExplicitAliasFeaturesModel(
        age=payload.features.age,
        hours_per_week=payload.features.hours_per_week, # Use Pythonic attribute
        klass=payload.features.klass,                 # Use Pythonic attribute
        education=payload.features.education,
        occupation=payload.features.occupation
    )
    
    base_pred, mitigated_pred = _calculate_mock_predictions( # Call the helper
        payload.x1, payload.x2, current_features_for_calc
    )
    
    return EvaluatedPointData(
        id=original_point_base_info["id"],
        x1=payload.x1,
        x2=payload.x2,
        features=current_features_for_calc, # This is an ExplicitAliasFeaturesModel instance
        true_label=original_point_base_info["true_label"],
        base_model_prediction=base_pred,
        mitigated_model_prediction=mitigated_pred
    )

# --- Placeholder Endpoints (remain the same) ---
@app.get("/api/partial_dependence")
async def get_partial_dependence(): return {"partial_dependence_data": []}
@app.get("/api/performance_fairness")
async def get_performance_fairness(): return {"roc_curve": [], "pr_curve": [], "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0}, "fairness_metrics": {"StatisticalParityDiff": 0, "DisparateImpact": 0, "EqualOpportunityDiff": 0}, "performance_metrics": {"Accuracy": 0, "F1Score": 0, "AUC": 0}}
@app.get("/api/features")
async def get_features(): return {"features": []}

# --- END OF FULL CORRECTED beespector_api/main.py ---