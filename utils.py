import os
import json
from datetime import datetime

import pandas as pd
from catboost.core import CatBoost


def create_submit_files(result: pd.DataFrame, meta_info:  dict=None, name: str=None):
	creation_time = "_".join(str(datetime.now()).split())
	folder_name = f"{creation_time}_{name}"
	os.mkdir(f"submits/{folder_name}")
	result.to_csv(f"submits/{folder_name}/prediction.csv", index=False)
	print(f"submits/{folder_name}/prediction.csv created")
	if meta_info:
		with open(f"submits/{folder_name}/meta_info.json", "w") as f:
			json.dump(meta_info, f)
		print(f"submits/{folder_name}/meta_info.json created")


def make_submit(model, preprocess_func, meta_info: dict=None, name: str=None):
    alfabattle2_prediction_session_timestamp = pd.read_csv("alfabattle2_prediction_session_timestamp.csv")
    test = preprocess_func(alfabattle2_prediction_session_timestamp)
    if isinstance(model, CatBoost):
        prediction = model.predict(test, prediction_type='Class').reshape(-1)
    else:
        prediction =  model.predict(test)
    alfabattle2_prediction_session_timestamp["prediction"] = pd.Series(prediction)
    alfabattle2_prediction_session_timestamp.drop("timestamp", axis=1, inplace=True) 
    create_submit_files(alfabattle2_prediction_session_timestamp, meta_info, name)  