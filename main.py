from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import sys
import os
import pandas as pd
import networkx as nx
from pathlib import Path

# Import functions from our optimization module
from optimization import ds_load, solve, hw_info

app = FastAPI(
    title="Optimization API",
    description="API for running optimization models from TEST.ipynb",
    version="1.0.0",
)


class DatasetInfo(BaseModel):
    name: str
    path: str


class OptimizationRequest(BaseModel):
    dataset: DatasetInfo
    model_name: str
    gamma: float
    seed_value: Optional[int] = 42
    timeout: Optional[int] = 3600


class OptimizationResult(BaseModel):
    dataset: str
    model: str
    gamma: float
    first_solution: str
    objective_value: Any
    solutions_count: Any
    solving_time: Any
    solver: str
    hardware: str
    x: List
    y: List
    z: List


@app.get("/")
async def root():
    return {"message": "Welcome to Optimization API"}


@app.get("/datasets")
async def get_datasets():
    datasets = [
        {"name": "Mobile operators", "path": "Mobile"},
        {"name": "Perfume (first 1000)", "path": "Perfume"},
        {"name": "Bibsonomy (last 1000)", "path": "Bibsonomy"},
        {"name": "IMDB", "path": "IMDB"},
    ]
    return datasets


@app.get("/hardware-info")
async def get_hardware_info():
    return {"info": hw_info()}


@app.post("/optimize", response_model=OptimizationResult)
async def run_optimization(request: OptimizationRequest):
    try:
        # Load dataset
        G_XY, G_YZ, G_ZX, X, Y, Z = ds_load(request.dataset)

        # Run optimization
        solve(request.dataset, request.model_name, request.gamma, request.seed_value)

        # Read results from CSV
        df_results = pd.read_csv("results.csv")
        latest_result = df_results.iloc[-1].to_dict()

        return OptimizationResult(
            dataset=latest_result["Dataset"],
            model=latest_result["Model"],
            gamma=latest_result["Gamma"],
            first_solution=latest_result["First solution (X,Y,Z)"],
            objective_value=latest_result["|X|+|Y|+|Z| (Objective value)"],
            solutions_count=latest_result["Number of found solutions"],
            solving_time=latest_result["Time, s"],
            solver=latest_result["Solver"],
            hardware=latest_result["HardWare"],
            x=(
                eval(latest_result["x"])
                if isinstance(latest_result["x"], str)
                else latest_result["x"]
            ),
            y=(
                eval(latest_result["y"])
                if isinstance(latest_result["y"], str)
                else latest_result["y"]
            ),
            z=(
                eval(latest_result["z"])
                if isinstance(latest_result["z"], str)
                else latest_result["z"]
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
