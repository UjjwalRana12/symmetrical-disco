from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import numpy as np
import pickle
import json

with open("bangalore_homes_prices_model.pickle", "rb") as f:
    model = pickle.load(f)


with open("columns.json", "r") as f:
    columns_data = json.load(f)
    data_columns = columns_data['data_columns']

app = FastAPI(
    title="Bangalore House Price Predictor",
    description="API to predict house prices in Bangalore based on location, square footage, bathrooms, and bedrooms",
    version="1.0.0"
)

# Request schema
class PriceRequest(BaseModel):
    location: str
    sqft: float
    bath: int
    bhk: int
    
    @validator('sqft')
    def sqft_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Square footage must be positive')
        return v
    
    @validator('bath', 'bhk')
    def rooms_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Number of rooms must be positive')
        return v

@app.get("/", tags=["Home"])
def root():
    """Welcome endpoint for the API."""
    return {"message": "Welcome to Bangalore House Price Predictor API!"}

@app.get("/locations", tags=["Metadata"])
def get_locations():
    """Get all available locations for prediction."""
    locations = [loc for loc in data_columns if loc not in ['total_sqft', 'bath', 'bhk']]
    return {"locations": locations}

@app.post("/predict", tags=["Prediction"])
def predict_price(data: PriceRequest):
   
    try:
        x = np.zeros(len(data_columns))
        x[0] = data.sqft
        x[1] = data.bath
        x[2] = data.bhk

        if data.location in data_columns:
            loc_index = data_columns.index(data.location)
            x[loc_index] = 1

        prediction = model.predict([x])[0]
        return {"estimated_price": round(prediction, 2)}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
