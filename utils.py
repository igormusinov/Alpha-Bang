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


def make_submit(model, preprocess_func, pred_df, meta_info: dict=None, name: str=None, **kwargs):
    test = preprocess_func(pred_df, **kwargs)
    if isinstance(model, CatBoost):
        prediction = model.predict(test, prediction_type='Class').reshape(-1)
    else:
        prediction =  model.predict(test)
    for_pred = pred_df.copy()
    for_pred["prediction"] = pd.Series(prediction)
    for_pred = for_pred.drop("timestamp", axis=1) 
    create_submit_files(for_pred, meta_info, name)  