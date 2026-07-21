import argparse
from pathlib import Path
from clearml import Dataset
import os
import time

def parse_args():
	"""Read dataset versioning inputs from the command line."""
	parser = argparse.ArgumentParser(
		description="Create a new ClearML dataset version from an existing base dataset."
	)
	parser.add_argument(
		"--version",
		type=str,
		default="1.0.1",
		help="Version tag for the new dataset, for example 1.0.1.",
	)
	parser.add_argument(
		"--data-path",
		type=str,
		default=None,
		help="Path to the folder that contains the new or modified dataset files.",
	)
	parser.add_argument(
		"--dataset-name",
		type=str,
		default="Dataset 1",
		help="Name of the dataset as it should appear in ClearML.",
	)
	parser.add_argument(
		"--dataset-project",
		type=str,
		default=None,
		help="ClearML project where the new dataset version will be created.",
	)
	parser.add_argument(
		"--tags",
		type=str,
		default=None,
		help="Optional comma-separated tags for the new dataset version.",
	)
	parser.add_argument(
		"--parent-id",
		type=str,
		default=None,
		help="ClearML dataset ID of the base version to build from.",
	)
	parser.add_argument(
		"--parent-project",
		type=str,
		default=None,
		help="Project name of the base dataset, used when resolving by name/version.",
	)
	parser.add_argument(
		"--parent-name",
		type=str,
		default=None,
		help="Dataset name of the base version, used when resolving by name/version.",
	)
	parser.add_argument(
		"--parent-version",
		type=str,
		default=None,
		help="Version of the base dataset, used when resolving by name/version.",
	)
	parser.add_argument(
		"--output-url",
		type=str,
		default=None,
		help=(
			"Upload destination for the dataset. Use a file:// URL for a local "
			"warehouse, or another ClearML-supported output URI."
		),
	)
	return parser.parse_args()


def normalize_output_url(output_url: str | None) -> str:
	"""Convert a local path into a file:// URL when needed."""
	if output_url:
		if output_url.startswith("file://"):
			return output_url
		return f"file://{Path(output_url).expanduser().resolve()}"

	# Default to a local warehouse folder in the current working directory.
	return f"file://{Path.cwd().joinpath('clearml_store').resolve()}"


def resolve_parent_dataset(args: argparse.Namespace) -> Dataset:
	"""Return the base dataset that the new version should inherit from."""
	if args.parent_id:
		return Dataset.get(dataset_id=args.parent_id)

	if args.parent_name and args.parent_project and args.parent_version:
		return Dataset.get(
			dataset_project=args.parent_project,
			dataset_name=args.parent_name,
			dataset_version=args.parent_version,
			only_completed=True,
		)

	raise ValueError(
		"Provide either --parent-id or the full set of "
		"--parent-project, --parent-name, and --parent-version."
	)


def main() -> None:
    """Create a new ClearML dataset version from a parent dataset."""
    args = parse_args()

    # Process the output url
    if args.output_url:
        output_url = normalize_output_url(args.output_url)
    else: 
        raise ValueError("Please provide an output path for this dataset")

	# ========================================================
    # ⏱️ START BENCHMARK TIMER
    # ========================================================
    start_time = time.perf_counter()

    # Resolve the base dataset so ClearML can link this version to it.
    parent_dataset = resolve_parent_dataset(args)
    parent_dataset_path = parent_dataset.get_local_copy()
    print(f"The path to the parent's dataset is: {parent_dataset_path}")

    dataset_tags = None
    if args.tags:
        dataset_tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
	
    project_name = args.dataset_project if args.dataset_project else parent_dataset.project

    # Pass parent_dataset.id (the string) to create a child version
    dataset = Dataset.create(
        dataset_project=args.dataset_project,
        dataset_name=args.dataset_name,
        dataset_tags=dataset_tags,
        parent_datasets=[parent_dataset.id], 
        dataset_version=args.version,
    )

    # Use sync_folder so modifications/deletions carry over correctly
    if args.data_path:
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"The specified data_path does not exist: '{args.data_path}'")
		
        dataset.sync_folder(local_path=str(args.data_path))
    else:
        raise ValueError(
            "\n[ERROR] Missing required dataset source path!\n"
            "To sync a new dataset version, you must explicitly pass '--data_path'.\n"
            "Do not point this to a ClearML cache directory (get_local_copy()). "
            "Point it to your active workspace folder (e.g., '~/workspace/.../dataset')."
        )


    # Upload the new version to the chosen storage location.
    print(f"Parent dataset ID linked successfully: {parent_dataset.id}")
    print(f"Syncing and uploading data from: {args.data_path}")
    print(f"Using output URL: {output_url}")
    dataset.upload(output_url=output_url)

    # Finalize the version so it becomes immutable and ready for use.
    dataset.finalize()

	# ========================================================
    # ⏱️ STOP BENCHMARK TIMER
    # ========================================================
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time

    print("\n" + "="*50)
    print("📊 CLEARML PUSH BENCHMARK RESULTS")
    print("="*50)
    print(f"Total Push Time: {elapsed_seconds:.2f} seconds")
    print("="*50 + "\n")

    print("Dataset version created successfully!")

if __name__ == "__main__":
	main()
