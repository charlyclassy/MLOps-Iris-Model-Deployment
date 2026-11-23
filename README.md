# MLOps Iris Model Deployment

A beginner-friendly MLOps practical project that trains an Iris flower classifier, saves and reloads the model, serves predictions through a FastAPI endpoint, and runs the application in Docker.

**Repository:** https://github.com/charlyclassy/MLOps-Iris-Model-Deployment

## Assignment objectives

1. Load the built-in Iris dataset with scikit-learn.
2. Train a Random Forest classifier and report test accuracy.
3. Save and reload the model using joblib, then predict one sample.
4. Create a FastAPI `/predict` endpoint with input validation.
5. Prepare a Dockerfile and explain monitoring for drift, latency, and errors.

## Results

| Item | Result |
| --- | --- |
| Dataset | scikit-learn Iris |
| Algorithm | RandomForestClassifier (100 trees) |
| Train/test split | 80% / 20%, stratified |
| Test accuracy | **90.00%** |
| Sample measurements | `[4.4, 3.0, 1.3, 0.2]` |
| Sample prediction | **setosa** |
| API test | HTTP **200 OK** |
| API prediction for example request | **setosa** |
| Docker | Container ran successfully with port 8000 mapped |

The accuracy and sample prediction above are from the local VS Code run. The API response and Docker deployment were also tested locally.

## Project files

| File | Purpose |
| --- | --- |
| `train_model.py` | Loads Iris data, trains and evaluates Random Forest, saves/reloads the model, and predicts a sample |
| `app.py` | FastAPI application exposing `POST /predict` with positive-number input validation |
| `iris_model.joblib` | Saved trained classifier |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Instructions for building the application container |
| `README.md` | Project instructions and results |

## Run locally in VS Code

Open the project folder in VS Code and open **Terminal → New Terminal**.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux, use `source .venv/bin/activate` instead.

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Train, evaluate, save, reload, and test the model:

```powershell
python train_model.py
```

The script prints the test accuracy, sample measurements, and sample prediction, and saves `iris_model.joblib`.

Start the API:

```powershell
python -m uvicorn app:app --reload
```

Open **http://127.0.0.1:8000/docs**, expand `POST /predict`, select **Try it out**, and submit:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

The locally tested response was:

```json
{
  "prediction": "setosa"
}
```

The API validates that all four measurements are positive numbers; invalid values result in a validation error.

## Run with Docker

Start Docker Desktop. From the project directory, build the image:

```powershell
docker build -t iris-api .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 iris-api
```

Visit **http://localhost:8000/docs** and test `POST /predict` with the same JSON input. The Docker-hosted endpoint was tested locally and returned **HTTP 200** with `setosa`.

To check running containers from another terminal:

```powershell
docker ps
```

## Monitoring ideas

These are **proposed monitoring checks**, not an implemented monitoring system.

1. **Data drift:** Compare the distributions of incoming sepal and petal measurements with the training data to identify meaningful changes.
2. **API latency:** Record prediction request response times and investigate unusual slowdowns.
3. **Errors:** Track HTTP 4xx/5xx responses and prediction failures to identify invalid requests or service problems.

## Submission note

The original assignment specifies a Google Colab link. This practical was completed in **VS Code**, and the source code is available in this GitHub repository. Confirm with the instructor that a GitHub repository link is an acceptable alternative.

## Repository housekeeping

Do not commit `.venv/` or `__pycache__/` directories. They contain local environment or generated cache files rather than project source code.

