import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib # For saving and loading scikit-learn models
import os
import json

# Define paths
DATA_DIR = "data"
MODEL_DIR = "models"
DATA_FILE = os.path.join(DATA_DIR, "adult.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "base_adult_logistic_model.pkl")
COLUMNS_FILE = os.path.join(MODEL_DIR, "adult_model_columns.json") # To store column order and types

def train_and_save_model():
    print(f"Attempting to load data from: {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Data file not found at {DATA_FILE}")
        print("Please download 'adult.csv' and place it in the 'data' directory.")
        return

    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    # Column names are based on the adult.names file or common knowledge of the dataset
    # The CSV from AIF360 link already has headers. If not, provide them.
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print("Data loaded successfully. Shape:", df.shape)
    print("First 5 rows:\n", df.head())
    print("\nColumn names:", df.columns.tolist())
    print("\nData types:\n", df.dtypes)

    # Define target variable and clean it ('>50K' becomes 1, '<=50K' becomes 0)
    target_column = 'income' 
    if target_column not in df.columns:
        print(f"ERROR: Target column '{target_column}' not found in CSV. Available columns: {df.columns.tolist()}")
        # Attempt to infer common alternatives
        possible_targets = ['income', 'class', 'target', 'Probability']
        inferred_target = None
        for pt in possible_targets:
            if pt in df.columns:
                inferred_target = pt
                print(f"Trying to use inferred target column: '{inferred_target}'")
                break
        if inferred_target:
            target_column = inferred_target
        else:
            return


    df['target'] = df[target_column].apply(lambda x: 1 if str(x).strip() == '>50K' else 0)
    df = df.drop(columns=[target_column])

    # Identify categorical and numerical features
    # For simplicity, we'll manually define some based on common adult dataset knowledge
    # This CSV has spaces in column names, remove them.
    df.columns = df.columns.str.strip().str.replace('-', '_').str.replace('.', '_')

    # Redefine features after cleaning column names
    categorical_features = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
    numerical_features = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

    # Ensure all defined features are present
    all_expected_features = categorical_features + numerical_features
    actual_columns = df.columns.tolist()
    
    missing_categorical = [f for f in categorical_features if f not in actual_columns]
    if missing_categorical:
        print(f"WARNING: Missing categorical features in CSV: {missing_categorical}")
        categorical_features = [f for f in categorical_features if f in actual_columns]

    missing_numerical = [f for f in numerical_features if f not in actual_columns]
    if missing_numerical:
        print(f"WARNING: Missing numerical features in CSV: {missing_numerical}")
        numerical_features = [f for f in numerical_features if f in actual_columns]

    if not categorical_features and not numerical_features:
        print("ERROR: No valid features identified after checking CSV columns. Exiting.")
        return

    features = categorical_features + numerical_features
    X = df[features]
    y = df['target']

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first') # drop='first' to avoid multicollinearity

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any (should not be if features list is exhaustive)
    )

    # Create the full pipeline with preprocessing and the model
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', LogisticRegression(solver='liblinear', random_state=42, max_iter=200))])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Train the model
    print("\nTraining the model...")
    model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model (optional, but good practice)
    y_pred_test = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")

    # Save the trained pipeline (includes preprocessor and model)
    joblib.dump(model_pipeline, MODEL_FILE)
    print(f"\nModel pipeline saved to: {MODEL_FILE}")

    # Save the column information (order and types, for consistent preprocessing later)
    # This is crucial for when we load the model and need to preprocess new data
    column_info = {
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'all_features_in_order': features # This is X.columns.tolist() before preprocessing
    }
    with open(COLUMNS_FILE, 'w') as f:
        json.dump(column_info, f, indent=4)
    print(f"Column info saved to: {COLUMNS_FILE}")

if __name__ == "__main__":
    train_and_save_model()