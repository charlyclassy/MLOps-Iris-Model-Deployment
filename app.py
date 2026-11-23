from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title='Iris Flower Classifier')
model_path = Path(__file__).with_name('iris_model.joblib')
model = joblib.load(model_path) if model_path.exists() else None
species = ['setosa', 'versicolor', 'virginica']

class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

@app.post('/predict')
def predict(data: IrisInput):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not found. Run train_model.py first.')
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    index = int(model.predict(features)[0])
    return {'prediction': species[index]}
