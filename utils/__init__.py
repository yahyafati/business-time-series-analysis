from . import *

import numpy as np
import time
from functools import wraps

from dataclasses import dataclass


def mean_absolute_percentage_error(y_true: list[float], y_pred: list[float]) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def plot_prediction(
    y_train: list[float],
    y_tests: dict[str, list[float]] = dict(),
    properties: dict[str, dict[str, object]] = dict(),
    title=None,
) -> None:

    plt.figure(figsize=(18, 6))
    plt.plot(y_train, label="Train", linestyle="-")
    empty_start = [None for _ in y_train]
    for name, y_test in y_tests.items():
        props = properties.get(name, {})
        plt.plot(empty_start + [x for x in y_test], label=name, **props)

    if title:
        plt.title(title)
    plt.xlabel("Year", fontsize=18)
    plt.ylabel("Value", fontsize=18)
    plt.legend(loc="best")


def timed_function(function_name=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            name = function_name if function_name else func.__name__
            print(f"Execution time of {name}: {execution_time:.6f} seconds")
            return result

        return wrapper

    return decorator


# def timed_function(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         start_time = time_module.time()
#         result = func(*args, **kwargs)
#         end_time = time_module.time()
#         execution_time = end_time - start_time
#         print(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")
#         return result

#     return wrapper
