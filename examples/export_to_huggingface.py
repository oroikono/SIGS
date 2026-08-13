"""Convert a SIGS Lightning checkpoint to Transformers/Hub format."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from sigs.huggingface import SIGSConfig, SIGSModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-id",
        help="Optional Hugging Face repository to create/update after local export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open() as handle:
        native_config = yaml.safe_load(handle)

    config = SIGSConfig.from_native_config(native_config)
    model = SIGSModel.from_lightning_checkpoint(args.checkpoint, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)

    card_template = Path(__file__).parents[1] / "hub" / "model-card.md"
    card_destination = args.output_dir / "README.md"
    if not card_destination.exists():
        shutil.copyfile(card_template, card_destination)

    print(f"Exported Transformers model to {args.output_dir}")
    if args.repo_id:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=args.output_dir,
            repo_id=args.repo_id,
            repo_type="model",
        )
        print(f"Pushed model to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
