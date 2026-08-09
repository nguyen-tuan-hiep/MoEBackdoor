import argparse
import json
import os

from datasets import DatasetDict, load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge separately-built AG News confident reference/probe datasets."
    )
    parser.add_argument("--reference_path", type=str, required=True)
    parser.add_argument("--probe_path", type=str, required=True)
    parser.add_argument("--reference_split", type=str, default="validation")
    parser.add_argument("--probe_split", type=str, default="test")
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def read_metadata(dataset_path: str):
    metadata_path = os.path.join(dataset_path, "selection_metadata.json")
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()

    reference_dataset = load_from_disk(args.reference_path)
    probe_dataset = load_from_disk(args.probe_path)

    merged = DatasetDict(
        {
            args.reference_split: reference_dataset[args.reference_split],
            args.probe_split: probe_dataset[args.probe_split],
        }
    )
    merged.save_to_disk(args.output_path)

    merged_metadata = {
        "reference_source_path": args.reference_path,
        "probe_source_path": args.probe_path,
        "reference_metadata": read_metadata(args.reference_path),
        "probe_metadata": read_metadata(args.probe_path),
    }
    metadata_path = os.path.join(args.output_path, "selection_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(merged_metadata, handle, indent=2)

    print(f"Merged dataset saved to {args.output_path}")


if __name__ == "__main__":
    main()
