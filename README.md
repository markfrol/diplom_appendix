# Optimization API

This API provides access to the optimization functionality from TEST.ipynb notebook.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API server:
```bash
uvicorn main:app --reload
```

The API will be available at http://localhost:8000

## API Endpoints

### GET /
Welcome message

### GET /datasets
Returns list of available datasets

### GET /hardware-info
Returns information about the hardware running the API

### POST /optimize
Run optimization for a specific dataset and model

Request body:
```json
{
    "dataset": {
        "name": "Mobile operators",
        "path": "Mobile"
    },
    "model_name": "M1",
    "gamma": 1.0,
    "seed_value": 42,
    "timeout": 3600
}
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc 