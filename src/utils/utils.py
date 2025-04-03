import time
from functools import wraps

import numpy as np
from datasets import Dataset as HFDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Calculate the duration
        duration = end_time - start_time

        # Convert to a readable format
        days, remainder = divmod(duration, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(
            f"execution time: {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds"
        )
        return result

    return wrapper


def scale(dset: HFDataset, scale_method: str):

    match scale_method:
        case "std":
            scaler = StandardScaler()
        case "minmax":
            scaler = MinMaxScaler()
        case _:
            raise ValueError(f"Scaler method {scale_method} not supported")

    X_train = np.array(dset["train"]["y"]).reshape(-1, 1)
    X_val = np.array(dset["val"]["y"]).reshape(-1, 1)
    X_test = np.array(dset["test"]["y"]).reshape(-1, 1)

    scaler.fit(X_train)

    # 4. Transform all datasets
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # If you need to replace the original datasets with the scaled data, you can do so
    dset["train"] = dset["train"].add_column(
        "scaled_y", X_train_scaled.squeeze().tolist()
    )  # Adjust as needed
    dset["val"] = dset["val"].add_column("scaled_y", X_val_scaled.squeeze().tolist())
    dset["test"] = dset["test"].add_column("scaled_y", X_test_scaled.squeeze().tolist())

    return dset
