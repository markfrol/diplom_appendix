import numpy as np
import pandas as pd
import cplex
import itertools
import networkx as nx
import pyscipopt as scip
import warnings
import os
import psutil
import cpuinfo
import platform
from math import log, floor
from tqdm import tqdm

# Global variables
df_results = pd.DataFrame()
timeout = 3600  # Default timeout in seconds


def hw_info():
    # Определение операционной системы
    system = platform.system()

    if system == "Windows":
        os_type = "Windows"
    elif system == "Darwin":
        os_type = "MacOS"
    elif system == "Linux":
        os_type = "Linux"
    else:
        os_type = "Другая ОС"

    # Информация о процессоре
    cpu_info = cpuinfo.get_cpu_info()
    cpu_brand = cpu_info["brand_raw"]
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()

    # Информация о памяти
    virtual_memory = psutil.virtual_memory()

    info = f"{os_type} | {cpu_count_logical} x {cpu_brand} {cpu_freq.current} MHz | {virtual_memory.total / (1024 ** 3):.2f} GB"

    return info


def ds_reader(dataset):
    X = set()
    Y = set()
    Z = set()
    with open(f"Dataset/{dataset['path']}/12.txt") as f:
        for a in f:
            X.add(a.split()[0])
            Y.add(a.split()[1])
    with open(f"Dataset/{dataset['path']}/12.txt", "rb") as f:
        G_XY = nx.read_edgelist(f)

    with open(f"Dataset/{dataset['path']}/23.txt") as f:
        for a in f:
            Y.add(a.split()[0])
            Z.add(a.split()[1])

    with open(f"Dataset/{dataset['path']}/23.txt", "rb") as f:
        G_YZ = nx.read_edgelist(f)

    with open(f"Dataset/{dataset['path']}/31.txt", "rb") as f:
        G_ZX = nx.read_edgelist(f)

    with open(f"Dataset/{dataset['path']}/31.txt") as f:
        for a in f:
            Z.add(a.split()[0])
            X.add(a.split()[1])
    X = list(X)
    Y = list(Y)
    Z = list(Z)
    return G_XY, G_YZ, G_ZX, X, Y, Z


def ds_load(dataset):
    match dataset["name"]:
        case "Mobile operators":
            result = ds_reader(dataset)
            return result
        case "Perfume (first 1000)":
            result = ds_reader(dataset)
            return result
        case "IMDB":
            result = ds_reader(dataset)
            return result
        case "Bibsonomy (last 1000)":
            result = ds_reader(dataset)
            return result
        case _:
            return "Unknown"


def save_results_to_df(
    dataset,
    model_name,
    gamma,
    best_obj,
    count_x,
    count_y,
    count_z,
    time,
    solutions,
    x,
    y,
    z,
):
    global df_results
    data = pd.DataFrame(
        {
            "Dataset": [dataset["name"]],
            "Model": [model_name],
            "Gamma": [gamma],
            "First solution (X,Y,Z)": f"({count_x}, {count_y}, {count_z})",
            "|X|+|Y|+|Z| (Objective value)": [best_obj],
            "Number of found solutions": [solutions],
            "Time, s": [time],
            "Solver": ["SciOpt"],
            "HardWare": [hw_info()],
            "x": [x],
            "y": [y],
            "z": [z],
        }
    )
    df_results = pd.concat([df_results, data], ignore_index=False)
    df_results.to_csv("results.csv", index=False)


def solve(dataset, model_name, gamma, seed_value=42):
    global df_results, timeout

    # Создание модели SCIP
    model = scip.Model()
    model.hideOutput()
    file_path = f'Model/{model_name}_g{gamma:.1f}_{dataset["path"]}.lp'
    model.readProblem(file_path)

    # Устанавливаем фиксированный сид для всех случайных процессов
    model.setParam("randomization/randomseedshift", seed_value)
    model.setParam("randomization/permutationseed", seed_value)
    model.setParam("randomization/permuteconss", True)
    model.setParam("randomization/permutevars", False)
    model.setParam("randomization/lpseed", seed_value)

    model.setParam("parallel/maxnthreads", 20)  # Установить количество потоков
    model.setParam("limits/time", timeout)  # Установка временного лимита на оптимизацию

    # Оптимизация модели
    model.optimize()

    # Проверка статуса решения
    if model.getStatus() == "optimal":
        best_sol = model.getBestSol()
        best_obj = model.getObjVal()
        print(f"Best objective value: {best_obj}")

        sold = eval(str(best_sol))
        x = [
            (key, sold[key]) for key in sold.keys() if key[0] == "x" and sold[key] > 0.9
        ]
        y = [
            (key, sold[key])
            for key in sold.keys()
            if key[0] == "y" and key[1] != "(" and sold[key] > 0.9
        ]
        z = [
            (key, sold[key])
            for key in sold.keys()
            if key[0] == "z" and key[1] != "(" and sold[key] > 0.9
        ]
        xfull = [(key, sold[key]) for key in sold.keys() if key[0] == "x"]
        yfull = [
            (key, sold[key]) for key in sold.keys() if key[0] == "y" and key[1] != "("
        ]
        zfull = [
            (key, sold[key]) for key in sold.keys() if key[0] == "z" and key[1] != "("
        ]

        xl, yl, zl = len(x), len(y), len(z)

        solving_time = model.getSolvingTime()  # Затраченное время
        num_solutions = model.getNSols()  # Получение числа найденных решений

        print(f"({xl}, {yl}, {zl})")
        print(f"Time: {solving_time}")
        print(f"Solutions: {num_solutions}")

        # Сохранение данных
        save_results_to_df(
            dataset,
            model_name,
            gamma,
            best_obj,
            xl,
            yl,
            zl,
            solving_time,
            num_solutions,
            x,
            y,
            z,
        )

    else:
        print(f"No solution in time: {timeout}'s")
        save_results_to_df(
            dataset,
            model_name,
            gamma,
            "-",
            "-",
            "-",
            "-",
            f"{timeout}",
            f"No solution in time",
            "-",
            "-",
            "-",
        )
