import os 
import sys
import argparse
from datetime import datetime
import yaml
import torch
from ultralytics import YOLO
from ultralytics import settings
from clearml import Task, Dataset
import time

# =========================
# 1. Parsing arguments from command
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="ClearML pipeline for YOLO")

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
        help="The ID of the dataset in ClearML"
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
        
    if ',' in device_string:
        try:
            return [int(x.strip()) for x in device_string.split(',')]
        except ValueError:
            pass
            
    if device_string.isdigit():
        return int(device_string)
        
    return device_string


if __name__ == '__main__':
    # CRITICAL: Re-enable the clearml setting callback while keeping mlflow false
    settings.update({"clearml": True, "mlflow": False})

    args = parse_args()

    # Load the base configurations from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract project names and configuration parameters
    project_name = config.get("experiment", "YOLO_Detections")
    architecture = config.get("architecture", "Unnamed_Architecture")
    task_type = config.get("task", "Unnamed_Task")
    
    # Customize run name
    current_date = datetime.now().strftime("%Y%m%d")  # e.g., "20260702"
    run_name = f"{architecture}_{task_type}_{current_date}"

    # Extract hyperparams and weights path
    hyperparams = config.get('hyperparams', {})
    weights = config.get('weights', None)
    clearml_dataset_id = args.data_id or config.get("data_id")

    # ========================================================
    # ⏱️ START BENCHMARK TIMER
    # ========================================================
    start_time = time.perf_counter()

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
    # 3. ClearML Initialization
    # =========================================================================
    # Initialize the tracking task room. This binds directly to the running session context.
    task = Task.init(
        project_name=project_name,
        task_name=run_name,
        output_uri=True # Keeps local artifact/weight tracking points default
    )
    
    print(f"Active ClearML Task ID: {task.id}")
    print(f"Running task: {task_type} | Architecture: {architecture}")
    
    # ClearML auto-tracks parameters dictionaries assigned directly via connect
    task.connect(hyperparams)

    # Log explicit configuration artifact directly into the tracking pipeline workspace
    data_path = hyperparams["data"]
    if os.path.exists(data_path):
        task.upload_artifact(name="dataset_config_yaml", artifact_object=data_path)

    # Run the model loop execution blocks natively
    model = YOLO(weights)
    print("Starting Model Training Loop...")
    
    # Ultralytics detects the active ClearML task context and pipes all metric charts automatically
    results = model.train(**hyperparams)


    # ========================================================
    # ⏱️ STOP BENCHMARK TIMER
    # ========================================================
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time

    print("\n" + "="*50)
    print("📊 CLEARML PUSH BENCHMARK RESULTS")
    print("="*50)
    print(f"Total Train Time: {elapsed_seconds:.2f} seconds")
    print("="*50 + "\n")
    
    print("Training completed successfully!")
    task.close()