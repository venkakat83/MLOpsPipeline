import mlflow
from mlflow_utils import create_ml_flow_experiement


if __name__ == "__main__" :
    
    name="Testing_MLFlow"
    artifact_location="testing_mlflow_artifacts"
    tags= {"env": "dev", "version" : "1.0.0"}

    exp_id = create_ml_flow_experiement(exp_name=name, artifact_location=artifact_location, tags=tags)
    print(f"Experiment ID {exp_id}")


