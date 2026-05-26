from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import json
import re
import warnings
from contextlib import contextmanager
from time import perf_counter

import numpy as np
import pandas as pd

try:
    import duckdb
except ModuleNotFoundError:
    duckdb = None

import xgboost as xgb

try:
    from catboost import CatBoostRegressor
except ModuleNotFoundError:
    CatBoostRegressor = None

from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    from lightgbm import LGBMRegressor
except ModuleNotFoundError:
    LGBMRegressor = None

__all__ = [
    "dataclass",
    "field",
    "Path",
    "Iterable",
    "json",
    "re",
    "warnings",
    "contextmanager",
    "perf_counter",
    "np",
    "pd",
    "duckdb",
    "xgb",
    "CatBoostRegressor",
    "KMeans",
    "AdaBoostRegressor",
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
    "mutual_info_regression",
    "RBFSampler",
    "LinearRegression",
    "Ridge",
    "mean_absolute_error",
    "mean_squared_error",
    "StandardScaler",
    "SVR",
    "DecisionTreeRegressor",
    "LGBMRegressor",
]
