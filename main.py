from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import io
import json
import os
import logging
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Prediction API", 
    version="1.0.0",
    description="API for predicting loan default risk using machine learning"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DEFAULT_MODEL_PATHS = [
    "model.joblib",
    "./model.joblib",
    r"D:\swastik\.0. Geakminds\3. Machine Learning\use case projects\model.joblib"
]

model = None

def find_model_path():
    for path in DEFAULT_MODEL_PATHS:
        if os.path.exists(path):
            return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_model_path = os.path.join(script_dir, "model.joblib")
    if os.path.exists(script_model_path):
        return script_model_path
    
    return None

MODEL_PATH = find_model_path()

def load_model():
    
    global model, MODEL_PATH
    try:
        if MODEL_PATH and os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
            return True
        else:
            
            MODEL_PATH = find_model_path()
            if MODEL_PATH:
                model = joblib.load(MODEL_PATH)
                logger.info(f"Model loaded successfully from {MODEL_PATH}")
                return True
            
            logger.error(f"Model file not found in any of the expected locations: {DEFAULT_MODEL_PATHS}")
            return False
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False
    
load_model()

class CreditApplication(BaseModel):

    CODE_GENDER: str = Field(..., description="Gender: M or F")
    FLAG_OWN_CAR: str = Field(..., description="Own car: Y or N")
    FLAG_OWN_REALTY: str = Field(..., description="Own realty: Y or N")
    CNT_CHILDREN: int = Field(..., ge=0, le=20, description="Number of children")
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Total income amount")
    AMT_CREDIT: float = Field(..., gt=0, description="Credit amount")
    AMT_ANNUITY: float = Field(..., gt=0, description="Loan annuity")
    AMT_GOODS_PRICE: float = Field(..., gt=0, description="Goods price")
    
    # Employment & Education
    NAME_TYPE_SUITE: Optional[str] = Field("Unaccompanied", description="Who accompanied client")
    NAME_INCOME_TYPE: str = Field(..., description="Income type")
    NAME_EDUCATION_TYPE: str = Field(..., description="Education type")
    NAME_FAMILY_STATUS: str = Field(..., description="Family status")
    NAME_HOUSING_TYPE: str = Field(..., description="Housing type")
    OCCUPATION_TYPE: Optional[str] = Field(None, description="Occupation type")
    
    # Age & Experience
    DAYS_BIRTH: int = Field(..., lt=0, description="Days before current day when client was born")
    DAYS_EMPLOYED: int = Field(..., description="Days before current day when client started employment")
    DAYS_REGISTRATION: float = Field(..., description="Days before application when client changed registration")
    DAYS_ID_PUBLISH: int = Field(..., description="Days before application when client changed identity document")
    
    # Contact Information
    FLAG_MOBIL: int = Field(..., ge=0, le=1, description="Mobile phone flag")
    FLAG_EMP_PHONE: int = Field(..., ge=0, le=1, description="Work phone flag")
    FLAG_WORK_PHONE: int = Field(..., ge=0, le=1, description="Work phone available flag")
    FLAG_CONT_MOBILE: int = Field(..., ge=0, le=1, description="Contact mobile flag")
    FLAG_PHONE: int = Field(..., ge=0, le=1, description="Phone flag")
    FLAG_EMAIL: int = Field(..., ge=0, le=1, description="Email flag")
    
    # Region & External Sources
    REGION_POPULATION_RELATIVE: float = Field(..., ge=0, le=1, description="Region population relative")
    WEEKDAY_APPR_PROCESS_START: str = Field(..., description="Weekday of application")
    HOUR_APPR_PROCESS_START: int = Field(..., ge=0, le=23, description="Hour of application")
    
    # External scores (optional)
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0, le=1, description="External source 1 score")
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0, le=1, description="External source 2 score")
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0, le=1, description="External source 3 score")
    
    @field_validator('CODE_GENDER')
    @classmethod
    def validate_gender(cls, v):
        if v not in ['M', 'F']:
            raise ValueError('Gender must be M or F')
        return v
    
    @field_validator('FLAG_OWN_CAR', 'FLAG_OWN_REALTY')
    @classmethod
    def validate_yes_no(cls, v):
        if v not in ['Y', 'N']:
            raise ValueError('Value must be Y or N')
        return v
    
    @field_validator('WEEKDAY_APPR_PROCESS_START')
    @classmethod
    def validate_weekday(cls, v):
        valid_days = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
        if v not in valid_days:
            raise ValueError(f'Weekday must be one of {valid_days}')
        return v

class PredictionResponse(BaseModel):
    loan_approved: bool
    default_probability: float
    risk_level: str
    confidence_score: float
    
class BatchPredictionResponse(BaseModel):
    predictions: List[Dict]
    total_processed: int
    approved_count: int
    denied_count: int
    average_default_probability: float

def preprocess_input(data: dict) -> pd.DataFrame:
   
    try:
        df = pd.DataFrame([data])
        
  
        categorical_mappings = {
            'CODE_GENDER': {'M': 1, 'F': 0},
            'FLAG_OWN_CAR': {'Y': 1, 'N': 0},
            'FLAG_OWN_REALTY': {'Y': 1, 'N': 0},
        }
        
        for col, mapping in categorical_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        # Target encode other categorical variables
        
        target_encode_cols = ['NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
                             'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
                             'WEEKDAY_APPR_PROCESS_START']
        
        default_encodings = {
            'NAME_INCOME_TYPE': {
                'Working': 0.080, 
                'Commercial associate': 0.090, 
                'Pensioner': 0.050, 
                'State servant': 0.060, 
                'Student': 0.070, 
                'Businessman': 0.100,
                'Unemployed': 0.120,
                'Maternity leave': 0.085
            },
            'NAME_EDUCATION_TYPE': {
                'Secondary / secondary special': 0.080, 
                'Higher education': 0.070, 
                'Incomplete higher': 0.080, 
                'Lower secondary': 0.100, 
                'Academic degree': 0.060
            },
            'NAME_FAMILY_STATUS': {
                'Married': 0.080, 
                'Single / not married': 0.090, 
                'Civil marriage': 0.090, 
                'Separated': 0.100, 
                'Divorced': 0.095,
                'Widow': 0.060,
                'Unknown': 0.085
            },
            'NAME_HOUSING_TYPE': {
                'House / apartment': 0.080, 
                'With parents': 0.090, 
                'Municipal apartment': 0.080, 
                'Rented apartment': 0.090, 
                'Office apartment': 0.100, 
                'Co-op apartment': 0.080
            },
            'NAME_TYPE_SUITE': {
                'Unaccompanied': 0.080,
                'Family': 0.075,
                'Spouse, partner': 0.077,
                'Children': 0.082,
                'Other_B': 0.085,
                'Other_A': 0.083,
                'Group of people': 0.090
            },
            'OCCUPATION_TYPE': {
                'Laborers': 0.090, 
                'Core staff': 0.080, 
                'Accountants': 0.070, 
                'Managers': 0.080,
                'Drivers': 0.090, 
                'Sales staff': 0.090, 
                'Cleaning staff': 0.100, 
                'Cooking staff': 0.080,
                'Private service staff': 0.090, 
                'Medicine staff': 0.070, 
                'Security staff': 0.080,
                'High skill tech staff': 0.070, 
                'Waiters/barmen staff': 0.090, 
                'Low-skill Laborers': 0.100,
                'Realty agents': 0.090, 
                'Secretaries': 0.080, 
                'IT staff': 0.070, 
                'HR staff': 0.080
            },
            'WEEKDAY_APPR_PROCESS_START': {
                'MONDAY': 0.080, 
                'TUESDAY': 0.080, 
                'WEDNESDAY': 0.080, 
                'THURSDAY': 0.080, 
                'FRIDAY': 0.080, 
                'SATURDAY': 0.082, 
                'SUNDAY': 0.082
            }
        }
        
        global_mean = 0.080  
        
        for col in target_encode_cols:
            if col in df.columns:
                if col in default_encodings:
                    df[col] = df[col].map(lambda x: default_encodings[col].get(x, global_mean) if pd.notna(x) else global_mean)
                else:
                    df[col] = global_mean
        

        bureau_features = {
            'bureau_AMT_CREDIT_SUM_mean': 500000.0,
            'bureau_AMT_CREDIT_SUM_sum': 1500000.0,
            'bureau_AMT_CREDIT_SUM_DEBT_mean': 150000.0,
            'bureau_AMT_CREDIT_SUM_DEBT_sum': 450000.0,
            'bureau_CREDIT_DAY_OVERDUE_max': 0.0
        }
        
        for feature, default_value in bureau_features.items():
            df[feature] = default_value
        
        expected_columns = [
            'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
            'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
            'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
            'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
            'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
            'FLAG_PHONE', 'FLAG_EMAIL', 'REGION_POPULATION_RELATIVE',
            'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START',
            'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
        ]
        
        for col in expected_columns:
            if col not in df.columns:
                if col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
                    df[col] = np.nan
                else:
                    df[col] = 0
        
        # Fill missing values
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                if df[col].isna().any():
                    median_val = 0.5 if col.startswith('EXT_SOURCE') else df[col].median()
                    df[col].fillna(median_val if pd.notna(median_val) else 0, inplace=True)
            else:
                df[col].fillna(global_mean, inplace=True)
        
        if model is not None:
            try:
                if hasattr(model, 'feature_name_'):
                    feature_names = model.feature_name_
                elif hasattr(model, 'booster_') and hasattr(model.booster_, 'feature_name'):
                    feature_names = model.booster_.feature_name()
                else:
                    logger.warning("Cannot get feature names from model, using all columns")
                    feature_names = None
                
                if feature_names:
                    missing_features = set(feature_names) - set(df.columns)
                    if missing_features:
                        logger.warning(f"Missing features: {missing_features}")
                        for feature in missing_features:
                            df[feature] = 0  
                    df = df[feature_names]
                    
                logger.info(f"Features prepared: {len(df.columns)} columns")
                
            except Exception as e:
                logger.error(f"Error getting feature names: {str(e)}")
                # If we can't get feature names, at least log the column count
                logger.info(f"Using {len(df.columns)} features")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the main index.html file
@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

@app.post("/predict", response_model=PredictionResponse)
async def predict_credit_risk(application: CreditApplication):
  
    if model is None:
        if not load_model():
            raise HTTPException(
                status_code=500, 
                detail="Model not loaded. Please ensure model.joblib exists in the correct location."
            )
    
    try:
        data = application.dict()
        
        # Log input data for debugging
        logger.info(f"Received application data: {data.get('CODE_GENDER', 'N/A')} - Income: {data.get('AMT_INCOME_TOTAL', 0)}")
        
        # Preprocess the data
        df = preprocess_input(data)
        
        probability = model.predict_proba(df)[0][1]  

        approval_threshold = 0.5
        loan_approved = probability < approval_threshold
        
        if probability < 0.3:
            risk_level = "Low Risk"
        elif probability < 0.6:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
        
        confidence_score = max(probability, 1 - probability)
        
        logger.info(f"Prediction made - Approved: {loan_approved}, Risk: {risk_level}")
        
        return PredictionResponse(
            loan_approved=loan_approved,
            default_probability=float(probability),
            risk_level=risk_level,
            confidence_score=float(confidence_score)
        )
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-csv", response_model=BatchPredictionResponse)
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Process a CSV file and return predictions for all rows
    """
    if model is None:
        if not load_model():
            raise HTTPException(
                status_code=500, 
                detail="Model not loaded. Please ensure model.joblib exists in the correct location."
            )
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        logger.info(f"Processing CSV with {len(df)} rows")
        
        results = []
        approved_count = 0
        denied_count = 0
        total_probability = 0
        
        for index, row in df.iterrows():
            try:
                row_dict = row.to_dict()
                
                if 'age' in row_dict and 'DAYS_BIRTH' not in row_dict:
                    row_dict['DAYS_BIRTH'] = -int(row_dict['age']) * 365
                elif 'DAYS_BIRTH' in row_dict and row_dict['DAYS_BIRTH'] > 0:
                    row_dict['DAYS_BIRTH'] = -row_dict['DAYS_BIRTH']
                
                if 'years_employed' in row_dict and 'DAYS_EMPLOYED' not in row_dict:
                    row_dict['DAYS_EMPLOYED'] = -int(row_dict['years_employed']) * 365
                elif 'DAYS_EMPLOYED' in row_dict and row_dict['DAYS_EMPLOYED'] > 0:
                    row_dict['DAYS_EMPLOYED'] = -row_dict['DAYS_EMPLOYED']
                
                processed_df = preprocess_input(row_dict)
                
                probability = model.predict_proba(processed_df)[0][1]
                loan_approved = probability < 0.5
                
                if probability < 0.3:
                    risk_level = "Low Risk"
                elif probability < 0.6:
                    risk_level = "Medium Risk"
                else:
                    risk_level = "High Risk"
                
                if loan_approved:
                    approved_count += 1
                else:
                    denied_count += 1
                
                total_probability += probability
                
                results.append({
                    "row_index": index,
                    "loan_approved": loan_approved,
                    "default_probability": float(probability),
                    "risk_level": risk_level,
                    "confidence_score": float(max(probability, 1 - probability))
                })
                
            except Exception as e:
                logger.error(f"Error processing row {index}: {str(e)}")
                results.append({
                    "row_index": index,
                    "error": str(e),
                    "loan_approved": False,
                    "default_probability": 1.0,
                    "risk_level": "Error",
                    "confidence_score": 0.0
                })
                denied_count += 1
                total_probability += 1.0
        
        avg_probability = total_probability / len(results) if results else 0
        
        logger.info(f"CSV processing complete - Approved: {approved_count}, Denied: {denied_count}")
        
        return BatchPredictionResponse(
            predictions=results,
            total_processed=len(results),
            approved_count=approved_count,
            denied_count=denied_count,
            average_default_probability=avg_probability
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except Exception as e:
        logger.error(f"CSV processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CSV processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Check API health and model status
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH if model is not None else None,
        "api_version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model-info")
async def get_model_info():

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        info = {
            "model_type": str(type(model)),
            "model_loaded": True,
            "model_path": MODEL_PATH
        }
        
        if hasattr(model, 'feature_name_'):
            info["features"] = model.feature_name_
            info["n_features"] = len(model.feature_name_)
        elif hasattr(model, 'booster_') and hasattr(model.booster_, 'feature_name'):
            info["features"] = model.booster_.feature_name()
            info["n_features"] = len(model.booster_.feature_name())
        elif hasattr(model, 'n_features_in_'):
            info["n_features"] = model.n_features_in_
            info["features"] = "Feature names not available"
        else:
            info["features"] = "Unable to determine features"
            info["n_features"] = "Unknown"
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.post("/reload-model")
async def reload_model():
    """
    Reload the model from disk
    """
    success = load_model()
    if success:
        return {"message": "Model reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

if __name__ == "__main__":
    import uvicorn
    import sys
    
    port = 9000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}")
            sys.exit(1)
    
    print(f"Starting server on port {port}")
    print(f"API documentation will be available at http://localhost:{port}/docs")
    
    try:
        uvicorn.run("main:app", host="127.0.0.1", port=9000, reload=True)
    except OSError as e:
        if "error while attempting to bind on address" in str(e):
            print(f"\nError: Port {port} is already in use!")
            print("Try one of the following:")
            print(f"1. Use a different port: python main.py 8001")
            print(f"2. Find and kill the process using port {port}:")
            print(f"   - Windows: netstat -ano | findstr :{port}")
            print("   - Then: taskkill /PID <PID> /F")
            print(f"3. Wait a moment and try again")
        else:
            raise