import mlflow
from typing import Any

import mlflow.entities
import mlflow.entities.experiment

def create_ml_flow_experiement(exp_name:str, artifact_location: str, tags:dict[str : Any] ) -> str:

    try : 
        exp_id = mlflow.create_experiment(
            name=exp_name,
            artifact_location=artifact_location,
            tags= tags)

    except:
        print(f"Expriment {exp_name} already exists")
        exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id

    
    return exp_id


def get_ml_flow_experiment(exp_id:str = None, exp_name:str = None) -> mlflow.entities.experiment :

    if exp_id is not None:
        experiment = mlflow.get_experiment(exp_id)
    elif exp_name is not None:
        experiment = mlflow.get_experiment_by_name(exp_name)
    else:    
        raise ValueError("Either exp_id or exp_name should be provided")
    
    return experiment



    