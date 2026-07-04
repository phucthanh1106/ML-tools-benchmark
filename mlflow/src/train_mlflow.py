from ultralytics import YOLO
import os 
from ultralytics import settings
import torch
import mlflow
import argparse
import sys
from datetime import datetime
import yaml
from clearml import Dataset

# =========================
# 1. Parsing arguments from command
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="MLflow pipeline for Yolo")

    # The primary configuration file
    parser.add_argument(
        '--config', 
        type=str,
        default='config.yaml', 
        help='Path to config file'
    )

    # Optional flags, default to None so it checks config.yaml first
    parser.add_argument(
        '--data-id', 
        type=str, 
        default=None,
        help="The ID of the dataset in clearML"
    )
    parser.add_argument(
        '--data-path', 
        type=str, 
        default=None,
        help="Path to the dataset.yaml file"
    )
    parser.add_argument(
        '--weights', 
        type=str, 
        default=None,
        help="Path to initial model weights (.pt file)"
    )
    parser.add_argument(
        '--batch', 
        type=int, 
        default=None, 
        help="Batch size"
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=None, 
        help="Number of epochs"
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default=None, 
        help="Override hardware target. Options: 'mps', 'cpu', '0', or '0,1' for multi-GPU"
    )

    return parser.parse_args()


# =========================
# 2. Handle multiple cases of user's input of device 
# =========================
def resolve_device_type(device_string):
    """Converts user input string to string or list of integers for Ultralytics."""
    if not device_string:
        return None
        
    # If the user passed comma-separated numbers like "0,1" -> return [0, 1]
    if ',' in device_string:
        try:
            return [int(x.strip()) for x in device_string.split(',')]
        except ValueError:
            pass
            
    # If the user passed a single integer string like "0" -> return int 0
    if device_string.isdigit():
        return int(device_string)
        
    # Otherwise return standard strings like 'mps' or 'cpu'
    return device_string

if __name__ == '__main__':
    # Shut down automatic background tracker 
    settings.update({"mlflow": False})

    args = parse_args()

    # Load the base configurations from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Fall back to localhost for tracking uri
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)  
    mlflow.set_experiment(config.get("experiment", []))

    # Customize run name
    current_date = datetime.now().strftime("%Y%m%d")  # e.g., "20260624"
    architecture = config.get("architecture", "Unnamed Architecture")
    task = config.get("task", "Unnamed Task")
    run_name = f"{architecture}_{task}_{current_date}"

    # Extract hyperparams, weights path and dataset path
    hyperparams = config.get('hyperparams', {})
    weights = config.get('weights', None)
    clearml_dataset_id = args.data_id or config.get("data_id")

    # Handling user's input
    if clearml_dataset_id:
        print(f"ClearML Cache Engine: Fetching Dataset ID [{clearml_dataset_id}]...")
        # 1. Download dataset into ClearML storage cache
        clearml_dataset_path = Dataset.get(dataset_id=clearml_dataset_id).get_local_copy()
        print(f"The path to the dataset is: {clearml_dataset_path}")
        
        # Path to the dataset.yaml file sitting inside the cache
        yaml_path = os.path.join(clearml_dataset_path, "dataset.yaml")
        
        # 2. DYNAMICALLY REWRITE THE PATHS INSIDE THE YAML
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
            
            # Force the 'path' parameter to point to the actual extraction folder dynamically
            yaml_content['path'] = os.path.abspath(clearml_dataset_path)

            # Convert hardcoded absolute keys into relative folder paths
            # This strips away '/workspace/data/vehicle-dataset/' and leaves just 'train/images', etc.
            for key in ['train', 'val', 'test']:
                if key in yaml_content and yaml_content[key]:
                    # If it's already a clean relative path (doesn't start with /), leave it alone
                    if not str(yaml_content[key]).startswith('/'):
                        continue
                        
                    # Extract just the last two directory elements (e.g., 'train/images' or 'valid/images')
                    path_parts = yaml_content[key].split('/')
                    if len(path_parts) >= 2:
                        relative_subpath = os.path.join(path_parts[-2], path_parts[-1]) # e.g., 'train/images'
                        yaml_content[key] = relative_subpath
            
            # Write the updated configuration back down
            with open(yaml_path, 'w') as f:
                yaml.safe_dump(yaml_content, f)
        
        # 3. Hand the corrected YAML path over to your hyperparams dictionary
        hyperparams["data"] = yaml_path
        
    else:
        if args.data_path:
            hyperparams["data"] = args.data_path
        else:
            print("Something is wrong with the data path. Please either check your latest command or your config file!")


    if args.batch:
        hyperparams["batch"] = args.batch
    if args.epochs:
        hyperparams["epochs"] = args.epochs
    if args.weights:
        weights = args.weights

    if args.device:
        hyperparams['device'] = resolve_device_type(args.device)
    elif config.get('device'):
        hyperparams['device'] = resolve_device_type(config['device'])
    else:
        hyperparams['device'] = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")    

    # =========================================================================
    # 3. MLFlow Initialization
    # =========================================================================
    with mlflow.start_run(run_name=run_name) as run:
        print(f"Active MLOps Run ID: {run.info.run_id}")
        print(f"Running task: {task} | Architecture: {architecture}")
        
        # Log your hyperparameter dictionary straight to MLflow parameters table
        mlflow.log_params(hyperparams)

        try:
            model = YOLO(weights)

            print("Starting Model Training Loop...")
            # Unpack the dictionary parameters cleanly using **
            results = model.train(**hyperparams)
            
            # Explicit Artifact Logging
            # Force log the dataset configuration file to ensure the run is self-contained
            data_path = hyperparams["data"]
            if os.path.exists(data_path):
                mlflow.log_artifact(data_path, artifact_path="dataset_configs")

            print("Training completed successfully!")
            mlflow.end_run(status="FINISHED")

        except KeyboardInterrupt:
            mlflow.end_run(status="KILLED")
            sys.exit(0)
            
        except Exception as e:
            print(f"Training error encountered: {str(e)}")
            # Log the crash reason straight to your MLflow UI dashboard tags for easy debugging
            mlflow.set_tag("crash_reason", str(e)[:250])
            mlflow.end_run(status="FAILED")
            raise e