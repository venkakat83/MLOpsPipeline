import mlflow
from typing import Any
from mlflow_utils import get_ml_flow_experiment

if __name__ == "__main__" :
    experiment = get_ml_flow_experiment(exp_name="Testing_MLFlow")
    print(experiment)

    experiment = get_ml_flow_experiment(exp_id="385460147425525610")
    print(experiment)
