from clearml import Dataset
import os
import sys
import shutil
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(
		description="Pulling back a ClearML dataset version to the workspace."
	)

    parser.add_argument(
        "--target-path",
        type=str,
        default="/",
        help="The target path that you want to store your dataset version back in"
    )

    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="The id of the dataset on ClearML that you want to pull back to your workspace"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.target_path:
        target_path = args.target_path
    else:
        raise ValueError(
            "There is something wrong with the target path. Please check your command and provide an appropriate target path!"
	    )

    if os.path.exists(args.target_path):
        print(f"[WARNING] A folder already exists at: {args.target_path}")
        print("Aborting to prevent accidental data overwrites. Clear it manually first if needed.")
        sys.exit(1)

    try:
        # ========================================================
        # ⏱️ START BENCHMARK TIMER
        # ========================================================
        start_time = time.perf_counter()

        print(f"Retrieving ClearML data whose ID is {args.dataset_id}")
        dataset = Dataset.get(dataset_id=args.dataset_id)

        # Extracting raw files from the zipped dataset to a cache path
        dataset_path = dataset.get_local_copy()
        print(f"Data safely verified in system cache layout at:\n -> {dataset_path}")

        # Move the extracted dataset to your active workspace
        print(f"\nMoving raw physical files to active workspace target location at {target_path}...")
        shutil.move(dataset_path, target_path)

        # ========================================================
        # ⏱️ STOP BENCHMARK TIMER
        # ========================================================
        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time

        print("\n" + "="*50)
        print("📊 CLEARML PUSH BENCHMARK RESULTS")
        print("="*50)
        print(f"Total Pull Time: {elapsed_seconds:.2f} seconds")
        print("="*50 + "\n")

        print(f"Success! Dataset is fully restored and active at:\n -> {target_path}")
    except Exception as e:
        print(f"\n[ERROR] Failed to restore dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


        


    