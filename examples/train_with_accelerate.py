"""Train the SIGS Grammar-VAE with Hugging Face Datasets and Accelerate."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import default_data_collator

from sigs.hf_data import load_hdf5_dataset
from sigs.huggingface import SIGSConfig, SIGSModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed)

    with args.config.open() as handle:
        native_config = yaml.safe_load(handle)
    config = SIGSConfig.from_native_config(native_config)
    model = (
        SIGSModel.from_pretrained(args.model_name_or_path)
        if args.model_name_or_path
        else SIGSModel(config)
    )

    split_config = native_config.get("data", {})
    dataset = load_hdf5_dataset(
        args.data,
        validation_split=float(split_config.get("validation_split", 0.2)),
        test_split=float(split_config.get("test_split", 0.1)),
        seed=args.seed,
        cache_dir=args.cache_dir,
    ).with_format("torch")
    train_loader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=default_data_collator,
    )
    validation_loader = DataLoader(
        dataset["validation"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=default_data_collator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model, optimizer, train_loader, validation_loader = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        validation_loader,
    )

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            with accelerator.accumulate(model):
                output = model(inputs=batch["inputs"], labels=batch["labels"])
                accelerator.backward(output.loss)
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in validation_loader:
                output = model(inputs=batch["inputs"], labels=batch["labels"])
                repeated_loss = output.loss.repeat(batch["inputs"].shape[0])
                validation_losses.append(accelerator.gather_for_metrics(repeated_loss))
        mean_loss = torch.cat(validation_losses).mean().item()
        accelerator.print(f"epoch={epoch + 1} validation_loss={mean_loss:.6f}")

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
        safe_serialization=True,
    )
    if accelerator.is_main_process:
        card_template = Path(__file__).parents[1] / "hub" / "model-card.md"
        shutil.copyfile(card_template, args.output_dir / "README.md")


if __name__ == "__main__":
    main()
