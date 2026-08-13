"""Export the SIGS production corpus to Arrow or the Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from sigs.hf_data import load_hdf5_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None and args.repo_id is None:
        parser.error("provide --output-dir, --repo-id, or both")
    return args


def main() -> None:
    args = parse_args()
    dataset = load_hdf5_dataset(
        args.data,
        validation_split=args.validation_split,
        test_split=args.test_split,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    if args.output_dir:
        dataset.save_to_disk(args.output_dir)
        print(f"Saved Arrow dataset to {args.output_dir}")
    if args.repo_id:
        dataset.push_to_hub(args.repo_id, private=args.private)
        card = Path(__file__).parents[1] / "hub" / "dataset-card.md"
        HfApi().upload_file(
            path_or_fileobj=card,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"Pushed dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
