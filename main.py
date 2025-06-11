from typing import List, Dict, Any, Tuple, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.inspection import partial_dependence
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for current context
class BeespectorContext:
    def __init__(self):
        self.dataset_df: Optional[pd.DataFrame] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.base_model: Optional[Any] = None
        self.mitigated_model: Optional[Any] = None
        self.feature_columns: List[str] = []
        self.categorical_features: List[str] = []
        self.numerical_features: List[str] = []
        self.target_column: str = ""
        self.dataset_name: str = ""
        self.base_classifier_type: str = ""
        self.mitigation_method: str = ""
        self.sensitive_feature: str = ""
        self.x1_feature: str = "age"
        self.x2_feature: str = "hours_per_week"
        self.preprocessor: Optional[Any] = None
        self.is_initialized: bool = False

context = BeespectorContext()


class InitializeContextRequest(BaseModel):
    dataset_name: str
    base_classifier: str
    classifier_params: Dict[str, Any] = {}
    mitigation_method: str
    mitigation_params: Dict[str, Any] = {}
    sensitive_feature: str
    x1_feature: Optional[str] = None
    x2_feature: Optional[str] = None

class StandardFeaturesModel(BaseModel):
    model_config = ConfigDict(extra='allow')

class InitialDataPoint(BaseModel):
    id: int
    x1: float
    x2: float
    true_label: int
    features: Dict[str, Any]
    pred_label: int
    pred_prob: float
    mitigated_pred_label: int
    mitigated_pred_prob: float

class EvaluatedPointPrediction(BaseModel):
    pred_label: int
    pred_prob: float

class EvaluatedPointData(BaseModel):
    id: int
    x1: float
    x2: float
    features: Dict[str, Any]
    true_label: int
    base_model_prediction: EvaluatedPointPrediction
    mitigated_model_prediction: EvaluatedPointPrediction

class FeatureStats(BaseModel):
    featureName: str
    count: int
    missing: int
    mean: float
    min: float
    max: float
    median: float
    std: float
    histogram: List[Dict[str, Any]]


def load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load dataset based on name."""
    
    # Map dataset slugs to file names
    dataset_map = {
        "adult": "adult.csv",
        "german": "german.csv"
    }
    
    if dataset_name not in dataset_map:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    data_file = os.path.join("data", dataset_map[dataset_name])
    if not os.path.exists(data_file):
        raise ValueError(f"Dataset file not found: {data_file}. Please ensure {dataset_map[dataset_name]} is in the data/ directory")
    
    df = pd.read_csv(data_file)
    
    # Handle missing values
    df = df.replace('?', np.nan)
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.replace('-', '_', regex=False).str.replace('.', '_', regex=False)
    
    logger.info(f"Dataset columns after standardization: {df.columns.tolist()}")
    
    if dataset_name == "adult":
        # Handle adult dataset target column
        target_candidates = ['income_per_year', 'income', 'class', 'target']
        target_col = None
        for col in target_candidates:
            if col in df.columns:
                target_col = col
                break
        
        if not target_col:
            raise ValueError("Target column not found in adult dataset")
        
        df['target'] = df[target_col].apply(lambda x: 1 if str(x).strip() == '>50K' else 0)
        if target_col != 'target':
            df = df.drop(columns=[target_col])
            
    elif dataset_name == "german":
        # Handle german dataset - note to self: adjust this, right now it assumes a 'class' column with 1 (good) and 2 (bad)
        if 'class' in df.columns:
            df['target'] = df['class'].apply(lambda x: 0 if x == 2 else 1)  # 0 for bad, 1 for good
            df = df.drop(columns=['class'])
        else:
            # If no class column, create a synthetic target for testing
            df['target'] = np.random.randint(0, 2, len(df))
    
    # Drop rows with missing target
    df = df.dropna(subset=['target'])
    
    # Ensure we have an ID column
    if 'id' not in df.columns:
        df.reset_index(drop=True, inplace=True)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'id'}, inplace=True)
    
    return df

def get_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    """Identify categorical and numerical features."""
    features = [col for col in df.columns if col != target_col and col != 'id']
    
    categorical = []
    numerical = []
    
    # Known numerical features that should always be treated as numerical
    known_numerical = ['age', 'hours_per_week', 'capital_gain', 'capital_loss', 'education_num', 
                      'fnlwgt', 'credit_amount', 'duration', 'installment_commitment', 
                      'residence_since', 'num_dependents']
    
    for col in features:
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            # Check if it's a known numerical feature first
            if col in known_numerical:
                numerical.append(col)
            # Otherwise use unique value count heuristic
            elif df[col].nunique() < 10:
                categorical.append(col)
            else:
                numerical.append(col)
        else:
            categorical.append(col)
    
    logger.info(f"Categorical features: {categorical}")
    logger.info(f"Numerical features: {numerical}")
    
    return categorical, numerical

def create_preprocessor(categorical_features: List[str], numerical_features: List[str]):
    """Create preprocessing pipeline."""
    
    # Numerical pipeline
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
    
    return preprocessor

def train_base_model(X_train: pd.DataFrame, y_train: pd.Series, 
                    classifier_type: str, params: Dict[str, Any],
                    preprocessor: Any) -> Pipeline:
    """Train base classifier."""
    
    # Map BeeFAME classifier names to sklearn/xgboost classes
    classifier_map = {
        'xgbclassifier': XGBClassifier,
        'random_forest_classifier': RandomForestClassifier,
        'support_vector_classification_(svc)': SVC,
        'logistic_regression': LogisticRegression
    }
    
    # Normalize classifier name
    classifier_key = classifier_type.lower().replace(' ', '_')
    
    if classifier_key not in classifier_map:
        raise ValueError(f"Unsupported classifier: {classifier_type}")
    
    # Get classifier class
    ClassifierClass = classifier_map[classifier_key]
    
    # Set up default parameters based on classifier type
    default_params = {
        'random_state': 34
    }
    
    if classifier_key == 'xgbclassifier':
        default_params.update({
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'n_jobs': -1
        })
    elif classifier_key == 'support_vector_classification_(svc)':
        default_params.update({
            'probability': True
        })
    elif classifier_key == 'logistic_regression':
        default_params.update({
            'max_iter': 1000
        })
    elif classifier_key == 'random_forest_classifier':
        default_params.update({
            'n_jobs': -1
        })
    
    # Merge with user params (user params override defaults)
    final_params = {**default_params, **params}
    
    # Remove any None or empty string values
    final_params = {k: v for k, v in final_params.items() if v is not None and v != ''}
    
    logger.info(f"Creating classifier {ClassifierClass.__name__} with params: {final_params}")
    
    # Create classifier instance
    classifier = ClassifierClass(**final_params)
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    return pipeline

def apply_mitigation(base_model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                    mitigation_method: str, sensitive_feature: str, 
                    mitigation_params: Dict[str, Any]) -> Pipeline:
    """Apply fairness mitigation."""
    
    # Map BeeFAME mitigation names
    mitigation_map = {
        'none': 'none',
        'relabeller': 'relabeller',
        'prevalence_sampling': 'prevalence_sampling',
        'data_repairer': 'data_repairer'
    }
    
    mitigation_key = mitigation_method.lower().replace(' ', '_')
    
    if mitigation_key == 'none' or mitigation_key == '':
        return base_model
    
    # For now, implement a simple reweighing approach for all mitigations
    
    weights = np.ones(len(y_train))
    
    if sensitive_feature in X_train.columns:
        sensitive_values = X_train[sensitive_feature]
        unique_values = sensitive_values.unique()
        
        # Calculate group statistics
        group_stats = {}
        for val in unique_values:
            mask = sensitive_values == val
            group_stats[val] = {
                'count': mask.sum(),
                'positive_rate': y_train[mask].mean() if mask.sum() > 0 else 0
            }
        
        overall_positive_rate = y_train.mean()
        
        # Apply different weighting strategies based on mitigation method
        for idx, (sensitive_val, label) in enumerate(zip(sensitive_values, y_train)):
            if mitigation_key == 'relabeller':
                # Boost underrepresented positive examples
                if group_stats[sensitive_val]['positive_rate'] < overall_positive_rate and label == 1:
                    weights[idx] = 1.5
            elif mitigation_key == 'prevalence_sampling':
                # Balance group sizes
                max_count = max(stat['count'] for stat in group_stats.values())
                weights[idx] = max_count / group_stats[sensitive_val]['count']
            elif mitigation_key == 'data_repairer':
                # Adjust based on fairness gap
                if group_stats[sensitive_val]['positive_rate'] > 0:
                    weights[idx] = overall_positive_rate / group_stats[sensitive_val]['positive_rate']
    
    # Create new classifier with same type and parameters
    base_classifier = base_model.named_steps['classifier']
    classifier_class = type(base_classifier)
    classifier_params = base_classifier.get_params()
    
    mitigated_classifier = classifier_class(**classifier_params)
    
    # Create new pipeline
    mitigated_pipeline = Pipeline([
        ('preprocessor', base_model.named_steps['preprocessor']),
        ('classifier', mitigated_classifier)
    ])
    
    # Fit with weights if classifier supports it
    try:
        mitigated_pipeline.fit(X_train, y_train, classifier__sample_weight=weights)
    except TypeError:
        # If classifier doesn't support sample weights, fit normally
        logger.warning(f"Classifier {classifier_class.__name__} doesn't support sample weights, fitting without mitigation")
        mitigated_pipeline.fit(X_train, y_train)
    
    return mitigated_pipeline

def get_predictions(model: Pipeline, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions and probabilities from model."""
    try:
        pred_labels = model.predict(X)
        pred_probs = model.predict_proba(X)[:, 1]
        return pred_labels, pred_probs
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return np.zeros(len(X)), np.ones(len(X)) * 0.5


app = FastAPI(title="Beespector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/api/initialize_context")
async def initialize_context(request: InitializeContextRequest):
    """Initialize Beespector with dataset and models based on parameters."""
    global context
    
    try:
        logger.info(f"Initializing context with dataset: {request.dataset_name}")
        logger.info(f"Base classifier: {request.base_classifier}")
        logger.info(f"Mitigation method: {request.mitigation_method}")
        logger.info(f"Sensitive feature: {request.sensitive_feature}")
        
        # Load dataset
        df = load_dataset(request.dataset_name)
        context.dataset_df = df.copy()
        context.dataset_name = request.dataset_name
        
        # Set target column
        context.target_column = 'target'
        context.feature_columns = [col for col in df.columns if col not in ['target', 'id']]
        
        # Determine x1 and x2 features based on dataset
        if request.dataset_name == "adult":
            context.x1_feature = "age" if "age" in context.feature_columns else context.feature_columns[0]
            context.x2_feature = "hours_per_week" if "hours_per_week" in context.feature_columns else context.feature_columns[1]
        elif request.dataset_name == "german":
            # For German dataset, use appropriate numerical features
            numerical_candidates = ['age', 'duration', 'credit_amount', 'installment_commitment']
            available_numerical = [col for col in numerical_candidates if col in context.feature_columns]
            if len(available_numerical) >= 2:
                context.x1_feature = available_numerical[0]
                context.x2_feature = available_numerical[1]
            else:
                # Fallback to first two numerical columns
                numerical_cols = [col for col in context.feature_columns if df[col].dtype in ['int64', 'float64']]
                context.x1_feature = numerical_cols[0] if len(numerical_cols) > 0 else context.feature_columns[0]
                context.x2_feature = numerical_cols[1] if len(numerical_cols) > 1 else context.feature_columns[1]
        
        # Override with request values if provided
        if request.x1_feature:
            context.x1_feature = request.x1_feature
        if request.x2_feature:
            context.x2_feature = request.x2_feature
            
        logger.info(f"Using x1_feature: {context.x1_feature}, x2_feature: {context.x2_feature}")
        
        X = df[context.feature_columns]
        y = df[context.target_column]
        
        # Split data with fixed random state for reproducibility
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=34, stratify=y
        )
        
        context.X_train = X_train
        context.X_test = X_test
        context.y_train = y_train
        context.y_test = y_test
        
        # Get feature types
        context.categorical_features, context.numerical_features = get_feature_types(X, context.target_column)
        
        # Create preprocessor
        context.preprocessor = create_preprocessor(
            context.categorical_features, 
            context.numerical_features
        )
        
        # Train base model
        context.base_model = train_base_model(
            X_train, y_train,
            request.base_classifier,
            request.classifier_params,
            context.preprocessor
        )
        context.base_classifier_type = request.base_classifier
        
        # Apply mitigation
        context.mitigated_model = apply_mitigation(
            context.base_model,
            X_train, y_train,
            request.mitigation_method,
            request.sensitive_feature,
            request.mitigation_params
        )
        context.mitigation_method = request.mitigation_method
        context.sensitive_feature = request.sensitive_feature
        
        context.is_initialized = True
        
        return {
            "status": "success",
            "message": "Context initialized successfully",
            "dataset": context.dataset_name,
            "n_samples": len(df),
            "n_features": len(context.feature_columns),
            "base_classifier": context.base_classifier_type,
            "mitigation_method": context.mitigation_method,
            "x1_feature": context.x1_feature,
            "x2_feature": context.x2_feature
        }
        
    except Exception as e:
        logger.error(f"Error initializing context: {str(e)}")
        context.is_initialized = False
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/datapoints", response_model=Dict[str, List[InitialDataPoint]])
async def get_all_datapoints():
    """Get sample of datapoints with predictions from both models."""
    if not context.is_initialized:
        raise HTTPException(status_code=400, detail="Context not initialized. Call /api/initialize_context first.")
    
    try:
        # Use fixed random seed for consistent sampling
        np.random.seed(34)
        sample_size = min(50, len(context.X_test))
        sample_indices = np.random.choice(context.X_test.index, size=sample_size, replace=False)
        
        X_sample = context.X_test.loc[sample_indices]
        y_sample = context.y_test.loc[sample_indices]
        
        # Get predictions
        base_labels, base_probs = get_predictions(context.base_model, X_sample)
        mit_labels, mit_probs = get_predictions(context.mitigated_model, X_sample)
        
        # Create datapoints
        datapoints = []
        for i, (idx, row) in enumerate(X_sample.iterrows()):
            features_dict = {}
            
            # Convert features to JSON-serializable types
            for col, val in row.items():
                if pd.isna(val) or (isinstance(val, float) and np.isinf(val)):
                    if col in context.numerical_features:
                        features_dict[col] = 0.0
                    else:
                        features_dict[col] = "Unknown"
                else:
                    if isinstance(val, (np.integer, np.int64, np.int32)):
                        features_dict[col] = int(val)
                    elif isinstance(val, (np.floating, np.float64, np.float32)):
                        features_dict[col] = float(val)
                    else:
                        features_dict[col] = str(val)
            
            # Handle x1 and x2 values
            try:
                x1_val = float(row[context.x1_feature]) if context.x1_feature in row and pd.notna(row[context.x1_feature]) else 0.0
                if np.isnan(x1_val) or np.isinf(x1_val):
                    x1_val = 0.0
            except (ValueError, TypeError):
                logger.warning(f"Could not convert x1 feature '{context.x1_feature}' value '{row.get(context.x1_feature)}' to float")
                x1_val = 0.0
                
            try:
                x2_val = float(row[context.x2_feature]) if context.x2_feature in row and pd.notna(row[context.x2_feature]) else 0.0
                if np.isnan(x2_val) or np.isinf(x2_val):
                    x2_val = 0.0
            except (ValueError, TypeError):
                logger.warning(f"Could not convert x2 feature '{context.x2_feature}' value '{row.get(context.x2_feature)}' to float")
                x2_val = 0.0
            
            # Ensure probabilities are valid
            pred_prob = float(base_probs[i])
            mit_prob = float(mit_probs[i])
            
            if np.isnan(pred_prob) or np.isinf(pred_prob):
                pred_prob = 0.5
            if np.isnan(mit_prob) or np.isinf(mit_prob):
                mit_prob = 0.5
            
            datapoint = InitialDataPoint(
                id=int(idx),
                x1=x1_val,
                x2=x2_val,
                true_label=int(y_sample.loc[idx]),
                features=features_dict,
                pred_label=int(base_labels[i]),
                pred_prob=pred_prob,
                mitigated_pred_label=int(mit_labels[i]),
                mitigated_pred_prob=mit_prob
            )
            datapoints.append(datapoint)
        
        return {"data": datapoints}
        
    except Exception as e:
        logger.error(f"Error getting datapoints: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/datapoints/{point_id}/evaluate", response_model=EvaluatedPointData)
async def evaluate_modified_point(point_id: int, payload: Dict[str, Any]):
    """Evaluate modified point with both models."""
    if not context.is_initialized:
        raise HTTPException(status_code=400, detail="Context not initialized.")
    
    try:
        # Extract features
        x1 = payload.get('x1', 0)
        x2 = payload.get('x2', 0)
        features = payload.get('features', {})
        
        # Create feature dataframe
        feature_data = {}
        for col in context.feature_columns:
            if col == context.x1_feature:
                feature_data[col] = x1
            elif col == context.x2_feature:
                feature_data[col] = x2
            else:
                val = features.get(col, 0)
                # Handle missing values
                if pd.isna(val) or (isinstance(val, float) and np.isinf(val)):
                    if col in context.numerical_features:
                        feature_data[col] = 0
                    else:
                        feature_data[col] = "Unknown"
                else:
                    feature_data[col] = val
        
        X_point = pd.DataFrame([feature_data])
        
        # Get predictions
        base_labels, base_probs = get_predictions(context.base_model, X_point)
        mit_labels, mit_probs = get_predictions(context.mitigated_model, X_point)
        
        # Get true label if available
        true_label = 0
        if point_id in context.dataset_df.index:
            true_label = int(context.dataset_df.loc[point_id, 'target'])
        
        # Clean features for response
        clean_features = {}
        for k, v in features.items():
            if pd.isna(v) or (isinstance(v, float) and np.isinf(v)):
                if k in context.numerical_features:
                    clean_features[k] = 0.0
                else:
                    clean_features[k] = "Unknown"
            else:
                if isinstance(v, (np.integer, np.int64, np.int32)):
                    clean_features[k] = int(v)
                elif isinstance(v, (np.floating, np.float64, np.float32)):
                    clean_features[k] = float(v)
                else:
                    clean_features[k] = str(v)
        
        # Ensure probabilities are valid
        base_prob = float(base_probs[0])
        mit_prob = float(mit_probs[0])
        
        if np.isnan(base_prob) or np.isinf(base_prob):
            base_prob = 0.5
        if np.isnan(mit_prob) or np.isinf(mit_prob):
            mit_prob = 0.5
        
        return EvaluatedPointData(
            id=point_id,
            x1=float(x1) if not (np.isnan(x1) or np.isinf(x1)) else 0.0,
            x2=float(x2) if not (np.isnan(x2) or np.isinf(x2)) else 0.0,
            features=clean_features,
            true_label=true_label,
            base_model_prediction=EvaluatedPointPrediction(
                pred_label=int(base_labels[0]),
                pred_prob=base_prob
            ),
            mitigated_model_prediction=EvaluatedPointPrediction(
                pred_label=int(mit_labels[0]),
                pred_prob=mit_prob
            )
        )
        
    except Exception as e:
        logger.error(f"Error evaluating point: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/features")
async def get_features():
    """Get feature statistics and distributions."""
    if not context.is_initialized:
        raise HTTPException(status_code=400, detail="Context not initialized.")
    
    try:
        features_list = []
        
        for col in context.feature_columns:
            if col in context.numerical_features:
                # Get statistics for numerical features
                series = context.dataset_df[col]
                series_clean = series.dropna()
                
                if len(series_clean) == 0:
                    continue
                
                # Create histogram
                hist, bins = np.histogram(series_clean, bins=10)
                histogram = []
                for i in range(len(hist)):
                    bin_label = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
                    histogram.append({"bin": bin_label, "value": int(hist[i])})
                
                # Calculate statistics
                mean_val = float(series_clean.mean()) if not series_clean.empty else 0.0
                min_val = float(series_clean.min()) if not series_clean.empty else 0.0
                max_val = float(series_clean.max()) if not series_clean.empty else 0.0
                median_val = float(series_clean.median()) if not series_clean.empty else 0.0
                std_val = float(series_clean.std()) if not series_clean.empty else 0.0
                
                # Handle invalid values
                if np.isnan(mean_val) or np.isinf(mean_val): mean_val = 0.0
                if np.isnan(min_val) or np.isinf(min_val): min_val = 0.0
                if np.isnan(max_val) or np.isinf(max_val): max_val = 0.0
                if np.isnan(median_val) or np.isinf(median_val): median_val = 0.0
                if np.isnan(std_val) or np.isinf(std_val): std_val = 0.0
                
                feature_stats = FeatureStats(
                    featureName=col,
                    count=int(series_clean.count()),
                    missing=int(series.isna().sum()),
                    mean=mean_val,
                    min=min_val,
                    max=max_val,
                    median=median_val,
                    std=std_val,
                    histogram=histogram
                )
                features_list.append(feature_stats)
        
        return {"features": features_list}
        
    except Exception as e:
        logger.error(f"Error getting features: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/performance_fairness")
async def get_performance_fairness():
    """Get performance and fairness metrics."""
    if not context.is_initialized:
        raise HTTPException(status_code=400, detail="Context not initialized.")
    
    try:
        # Get predictions on test set
        base_labels, base_probs = get_predictions(context.base_model, context.X_test)
        
        # Calculate metrics
        fpr, tpr, _ = roc_curve(context.y_test, base_probs)
        precision, recall, _ = precision_recall_curve(context.y_test, base_probs)
        cm = confusion_matrix(context.y_test, base_labels)
        
        # Calculate fairness metrics
        statistical_parity_diff = 0.0
        disparate_impact = 1.0
        equal_opportunity_diff = 0.0
        
        if context.sensitive_feature in context.X_test.columns:
            # Get unique groups
            sensitive_groups = context.X_test[context.sensitive_feature].dropna().unique()
            if len(sensitive_groups) >= 2:
                group_rates = []
                for group in sensitive_groups[:2]:  # Compare first two groups
                    mask = context.X_test[context.sensitive_feature] == group
                    if mask.sum() > 0:
                        group_pred_rate = base_labels[mask].mean()
                        group_rates.append(group_pred_rate)
                
                if len(group_rates) == 2:
                    statistical_parity_diff = group_rates[0] - group_rates[1]
                    disparate_impact = group_rates[0] / (group_rates[1] + 1e-10)
                    
                    # Handle invalid values
                    if np.isnan(statistical_parity_diff) or np.isinf(statistical_parity_diff):
                        statistical_parity_diff = 0.0
                    if np.isnan(disparate_impact) or np.isinf(disparate_impact):
                        disparate_impact = 1.0
        
        # Prepare response data
        roc_data = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr[::5], tpr[::5])]  # Downsample
        pr_data = [{"recall": float(r), "precision": float(p)} for r, p in zip(recall[::5], precision[::5])]
        
        return {
            "roc_curve": roc_data,
            "pr_curve": pr_data,
            "confusion_matrix": {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1])
            },
            "fairness_metrics": {
                "StatisticalParityDiff": float(statistical_parity_diff),
                "DisparateImpact": float(disparate_impact),
                "EqualOpportunityDiff": float(equal_opportunity_diff)
            },
            "performance_metrics": {
                "Accuracy": float((cm[0, 0] + cm[1, 1]) / cm.sum()),
                "F1Score": float(2 * cm[1, 1] / (2 * cm[1, 1] + cm[0, 1] + cm[1, 0])),
                "AUC": float(np.trapz(tpr, fpr))
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/partial_dependence")
async def get_partial_dependence():
    """Get partial dependence data."""
    if not context.is_initialized:
        raise HTTPException(status_code=400, detail="Context not initialized.")
    
    try:
        # Get feature indices
        features_to_plot = [
            context.feature_columns.index(context.x1_feature),
            context.feature_columns.index(context.x2_feature)
        ]
        
        # Calculate partial dependence
        pd_results = partial_dependence(
            context.base_model,
            context.X_test.head(100),  # Use subset for performance
            features=[features_to_plot[0]],
            grid_resolution=20
        )
        
        pd_data = []
        
        grid_values = pd_results.get('grid_values', pd_results.get('values', [None]))[0]
        if grid_values is None:
            raise ValueError("Could not find grid values in partial dependence results")
            
        for i, val in enumerate(grid_values):
            pd_data.append({
                "x": float(val),
                "pd_x1": float(pd_results['average'][0][i]),
                "pd_x2": 0.0  # Placeholder
            })
        
        return {"partial_dependence_data": pd_data}
        
    except Exception as e:
        logger.error(f"Error getting partial dependence: {str(e)}")
        # Return empty data on error
        return {"partial_dependence_data": []}