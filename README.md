# MLOps practical: Iris classifier (VS Code)

This project follows the five practical tasks from the supplied MLOps assignment.

## Run in VS Code

1. Open this folder in VS Code and open its integrated terminal.
2. Create a virtual environment: `python -m venv .venv`
3. Activate it:
   - Windows PowerShell: `.venv\Scripts\Activate.ps1`
   - macOS/Linux: `source .venv/bin/activate`
4. Install dependencies: `python -m pip install -r requirements.txt`
5. Train, evaluate, save, reload, and predict: `python train_model.py`
6. Start the API: `python -m uvicorn app:app --reload`
7. Open http://127.0.0.1:8000/docs and test POST `/predict` with:

```json
{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
```

The API validates that all four measurements are positive numbers. FastAPI returns a validation error for invalid inputs.

## Docker (requires Docker Desktop / Docker Engine)

```bash
docker build -t iris-api .
docker run --rm -p 8000:8000 iris-api
```

Then visit http://127.0.0.1:8000/docs.

## Monitoring ideas (three required)

1. **Data drift:** Compare recent input measurement distributions with training data and investigate large changes.
2. **Latency:** Track request response times and investigate slowdowns.
3. **Errors:** Track HTTP 4xx/5xx counts and prediction failures to catch broken requests or service problems.

## Submission

Record the **actual accuracy and sample prediction** printed by `train_model.py`; include the FastAPI code, Dockerfile, and the three monitoring ideas. The original slide specifies a Colab link, but this VS Code version is a local project. If your instructor requires a link, ask whether a GitHub repository link is acceptable.
