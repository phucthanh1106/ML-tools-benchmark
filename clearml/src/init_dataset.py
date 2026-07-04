import argparse
from pathlib import Path
from clearml import Dataset


def parse_args():
    """Read the dataset metadata and paths from the command line."""
    parser = argparse.ArgumentParser(
        description="Version a dataset with ClearML using user-provided values."
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Dataset version tag, for example 1.0.0.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/",
        help="Path to the dataset folder that contains the files to version.",
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
        default="Project 1",
        help="ClearML project name where the dataset should be created.",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Think of it as sticky notes for your dataset timeline",
    )
    parser.add_argument(
        "--output-url",
        type=str,
        default="/",
        help=(
            "Upload destination for the dataset. Use a file:// URL for a local "
            "warehouse, or an artifact URL supported by your ClearML setup."
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


def main() -> None:
    """Create, upload, and finalize a versioned ClearML dataset."""
    args = parse_args()
    data_path = args.data_path
    output_url = normalize_output_url(args.output_url)

    # Create the dataset entry in ClearML using the user-provided metadata.
    dataset = Dataset.create(
        dataset_project=args.dataset_project,
        dataset_name=args.dataset_name,
        dataset_version=args.version,
    )

    # Add the dataset folder so ClearML can track all files in the version.
    dataset.add_files(path=str(data_path))

    # Upload to the configured warehouse location.
    print(f"Uploading dataset from: {data_path}")
    print(f"Using output URL: {output_url}")
    dataset.upload(output_url=output_url)

    # Finalize to lock the dataset version and prevent accidental changes.
    dataset.finalize()
    print("Dataset locked and versioned successfully!")

if __name__ == "__main__":
    main()

