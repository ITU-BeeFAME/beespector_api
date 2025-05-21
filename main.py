from typing import List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import joblib
import os
import json
import numpy as np

DATA_DIR = "data"
MODEL_DIR = "models"
DATA_FILE = os.path.join(DATA_DIR, "adult.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "base_adult_logistic_model.pkl")
COLUMNS_FILE = os.path.join(MODEL_DIR, "adult_model_columns.json")

dataset_df: pd.DataFrame = None
model_pipeline: Any = None
model_columns_info: Dict[str, List[str]] = None
initial_datapoints_cache: List[Dict[str, Any]] = []

class StandardFeaturesModel(BaseModel): 
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int = Field(serialization_alias="hours-per-week") 
    native_country: str   
    klass: str = Field(serialization_alias="class") 
    

class InitialDataPoint(BaseModel):
    id: int; x1: float; x2: float; true_label: int
    features: StandardFeaturesModel 
    pred_label: int; pred_prob: float
    mitigated_pred_label: int; mitigated_pred_prob: float

class EvaluatedPointPrediction(BaseModel):
    pred_label: int; pred_prob: float

class EvaluatedPointData(BaseModel):
    id: int; x1: float; x2: float
    features: StandardFeaturesModel 
    true_label: int
    base_model_prediction: EvaluatedPointPrediction
    mitigated_model_prediction: EvaluatedPointPrediction

def get_predictions(input_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:

    if model_pipeline is None: raise RuntimeError("Model not loaded.")
    try:
        ordered_input_df = input_df[model_columns_info['all_features_in_order']]
        pred_probs_all_classes = model_pipeline.predict_proba(ordered_input_df)
        pred_probs_positive_class = pred_probs_all_classes[:, 1]
        pred_labels = model_pipeline.predict(ordered_input_df)
        return pred_labels, pred_probs_positive_class
    except Exception as e:
        print(f"Error during prediction: {e}")
        return np.zeros(len(input_df), dtype=int), np.zeros(len(input_df), dtype=float)

def simulate_mitigated_predictions(base_labels: np.ndarray, base_probs: np.ndarray, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    
    print(f"DEBUG: Simulating mitigated predictions. Input base_probs: {base_probs.tolist()}")
    print(f"DEBUG: features_df for simulation (first row if multiple): \n{features_df.head(1)}")
    mitigated_probs = base_probs.copy()
    sex_col = 'sex'; race_col = 'race'
    if sex_col in features_df.columns:
        female_condition = features_df[sex_col].astype(str).str.strip().str.lower() == 'female'
        if np.any(female_condition):
            print(f"DEBUG: Found {np.sum(female_condition)} 'Female' entries for sex-based mitigation.")
            mitigated_probs[female_condition] = np.clip(mitigated_probs[female_condition] * 0.9, 0.01, 0.99)
    else: print(f"DEBUG: '{sex_col}' column not found in features_df for simulation. Columns: {features_df.columns.tolist()}")
    if race_col in features_df.columns:
        black_condition = features_df[race_col].astype(str).str.strip().str.lower() == 'black'
        if np.any(black_condition):
            print(f"DEBUG: Found {np.sum(black_condition)} 'Black' entries for race-based mitigation.")
            mitigated_probs[black_condition] = np.clip(mitigated_probs[black_condition] * 1.1, 0.01, 0.99)
    else: print(f"DEBUG: '{race_col}' column not found in features_df for simulation. Columns: {features_df.columns.tolist()}")
    mitigated_labels = (mitigated_probs > 0.5).astype(int)
    print(f"DEBUG: Output mitigated_probs: {mitigated_probs.tolist()}, mitigated_labels: {mitigated_labels.tolist()}")
    return mitigated_labels, np.round(mitigated_probs, 3)


app = FastAPI(title="Beespector API")

@app.on_event("startup")
async def startup_event():
    global dataset_df, model_pipeline, model_columns_info, initial_datapoints_cache
    print("Loading dataset and model at startup...")
    try:
        temp_df = pd.read_csv(DATA_FILE)
        temp_df.columns = temp_df.columns.str.strip().str.replace('-', '_', regex=False).str.replace('.', '_', regex=False)
        actual_target_col = None
        possible_targets = ['income_per_year', 'income', 'class', 'target']
        for ct_name in possible_targets:
            if ct_name in temp_df.columns: actual_target_col = ct_name; break
        if not actual_target_col: raise ValueError(f"Target column not found. Checked: {possible_targets}")
        print(f"Using target column: {actual_target_col}")
        temp_df['target'] = temp_df[actual_target_col].apply(lambda x: 1 if str(x).strip() == '>50K' else 0)
        dataset_df = temp_df.drop(columns=[actual_target_col])
        dataset_df = dataset_df.reset_index().rename(columns={'index': 'id'})
        with open(COLUMNS_FILE, 'r') as f: model_columns_info = json.load(f)
        if 'all_features_in_order' not in model_columns_info: raise ValueError(f"Key 'all_features_in_order' missing in {COLUMNS_FILE}")
        model_pipeline = joblib.load(MODEL_FILE)
        print("Model loaded successfully.")

        SAMPLE_SIZE_FOR_CACHE = 200
        df_for_cache = dataset_df.sample(n=min(SAMPLE_SIZE_FOR_CACHE, len(dataset_df)), random_state=42).copy().reset_index(drop=True)
        print(f"Processing {len(df_for_cache)} sampled points for initial cache.")
        
        features_for_pred_df = df_for_cache[model_columns_info['all_features_in_order']].copy()
        for col in features_for_pred_df.columns:
            if features_for_pred_df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(features_for_pred_df[col]): fill_value = features_for_pred_df[col].median()
                else: fill_value = features_for_pred_df[col].mode()[0] if not features_for_pred_df[col].mode().empty else "Unknown"
                features_for_pred_df[col] = features_for_pred_df[col].fillna(fill_value)

        base_labels, base_probs = get_predictions(features_for_pred_df)
        miti_labels, miti_probs = simulate_mitigated_predictions(base_labels, base_probs, features_for_pred_df.copy())

        x1_col, x2_col = 'age', 'hours_per_week'
        
        if x1_col not in df_for_cache.columns: x1_col = model_columns_info['all_features_in_order'][0] if model_columns_info['all_features_in_order'] else 'age' 
        if x2_col not in df_for_cache.columns: x2_col = model_columns_info['all_features_in_order'][1] if len(model_columns_info['all_features_in_order']) > 1 else x1_col 
        
        for i in range(len(df_for_cache)):
            row = df_for_cache.iloc[i]
            
            features_data_for_pydantic_model = {
                "age": int(row["age"]),
                "workclass": str(row["workclass"]),
                "fnlwgt": int(row["fnlwgt"]),
                "education": str(row["education"]),
                "education_num": int(row["education_num"]),
                "marital_status": str(row["marital_status"]),
                "occupation": str(row["occupation"]),
                "relationship": str(row["relationship"]),
                "race": str(row["race"]),
                "sex": str(row["sex"]),
                "capital_gain": int(row["capital_gain"]),
                "capital_loss": int(row["capital_loss"]),
                "hours_per_week": int(row["hours_per_week"]), 
                "native_country": str(row["native_country"]), 
                "klass": str(row["workclass"])                  
            }
            
            for k, v_val in features_data_for_pydantic_model.items():
                 if pd.isna(v_val): 
                    if k in ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country", "klass"]:
                         features_data_for_pydantic_model[k] = "Unknown"
            
            initial_datapoints_cache.append(dict(
                id=int(row['id']), x1=float(row[x1_col]), x2=float(row[x2_col]),
                true_label=int(row['target']),
                features=features_data_for_pydantic_model,
                pred_label=int(base_labels[i]), pred_prob=float(base_probs[i]),
                mitigated_pred_label=int(miti_labels[i]), mitigated_pred_prob=float(miti_probs[i])
            ))
        print(f"Initial cache populated with {len(initial_datapoints_cache)} points.")
    except Exception as e: print(f"FATAL STARTUP ERROR: {e}"); import traceback; traceback.print_exc()

app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/api/datapoints", response_model=Dict[str, List[InitialDataPoint]])
async def get_all_datapoints():
    if not initial_datapoints_cache: return {"data": []}
    try:
        return {"data": [InitialDataPoint(**dp) for dp in initial_datapoints_cache]}
    except Exception as e: print(f"Error creating InitialDataPoint instances for response: {e}"); raise HTTPException(500, "Error processing data for API response")


class FeaturesPayload(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    
    hours_per_week: int = Field(validation_alias="hours-per-week")
    native_country: str 
    klass: str = Field(validation_alias="class")
    model_config = ConfigDict(populate_by_name=True)

class UpdatePointPayload(BaseModel):
    x1: float; x2: float
    features: FeaturesPayload 
    model_config = ConfigDict(populate_by_name=True)

@app.put("/api/datapoints/{point_id}/evaluate", response_model=EvaluatedPointData)
async def evaluate_modified_point(point_id: int, payload: UpdatePointPayload):
    if not model_pipeline: raise HTTPException(503, "Model not ready.")
    original_row = dataset_df[dataset_df['id'] == point_id]
    if original_row.empty: raise HTTPException(404, "Point ID not found.")
    true_label = int(original_row.iloc[0]['target'])

    features_for_df_dict = payload.features.model_dump(by_alias=False) 
    input_features_df = pd.DataFrame([features_for_df_dict])
    
    input_features_df = input_features_df[model_columns_info['all_features_in_order']] 
    for col in input_features_df.columns: 
        if input_features_df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(input_features_df[col]): fill_value = dataset_df[col].median()
            else: fill_value = dataset_df[col].mode()[0] if not input_features_df[col].mode().empty else "Unknown"
            input_features_df[col] = input_features_df[col].fillna(fill_value)

    base_labels, base_probs = get_predictions(input_features_df)
    miti_labels, miti_probs = simulate_mitigated_predictions(base_labels, base_probs, input_features_df.copy())

    response_features_model = StandardFeaturesModel(**features_for_df_dict)

    return EvaluatedPointData(
        id=point_id, x1=payload.x1, x2=payload.x2, features=response_features_model,
        true_label=true_label,
        base_model_prediction=EvaluatedPointPrediction(pred_label=int(base_labels[0]), pred_prob=float(base_probs[0])),
        mitigated_model_prediction=EvaluatedPointPrediction(pred_label=int(miti_labels[0]), pred_prob=float(miti_probs[0]))
    )

@app.get("/api/partial_dependence")
async def get_partial_dependence(): return {"partial_dependence_data": []}
@app.get("/api/performance_fairness")
async def get_performance_fairness(): return { "roc_curve": [], "pr_curve": [], "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0}, "fairness_metrics": {"StatisticalParityDiff": 0, "DisparateImpact": 0, "EqualOpportunityDiff": 0}, "performance_metrics": {"Accuracy": 0, "F1Score": 0, "AUC": 0}}
@app.get("/api/features")
async def get_features(): return {"features": []}