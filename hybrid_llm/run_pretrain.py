import argparse
import os
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import ClearMLLogger, CSVLogger

from hybrid_llm.configs import get_config, get_training_config
from hybrid_llm.training import HybridLightningModule, StreamingDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train Hybrid LLM with streaming datasets")
    parser.add_argument("--model-size", default="base", choices=["nano", "small", "base", "large"])
    parser.add_argument("--train-preset", default="week_100b", choices=["debug", "quick", "week_100b", "single_gpu", "multi_gpu"])
    parser.add_argument("--checkpoint-dir", default=None, help="Override checkpoint directory")
    parser.add_argument("--disable-clearml", action="store_true", help="Skip ClearML logging")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    return parser.parse_args()


def main():
    args = parse_args()
    model_cfg = get_config(args.model_size)
    train_cfg = get_training_config(args.train_preset)
    
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps
    if args.checkpoint_dir:
        train_cfg.checkpoint_dir = args.checkpoint_dir
    
    # Tokenization/Data
    datamodule = StreamingDataModule(train_cfg=train_cfg, num_workers=0)
    
    # Model + LightningModule
    module = HybridLightningModule(model_cfg=model_cfg, train_cfg=train_cfg)
    
    # Logging
    loggers = []
    if train_cfg.use_clearml and not args.disable_clearml:
        loggers.append(ClearMLLogger(project=train_cfg.clearml_project, task_name=train_cfg.clearml_task))
    loggers.append(CSVLogger(save_dir=train_cfg.checkpoint_dir, name="metrics"))
    
    # Checkpointing every N steps
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=train_cfg.checkpoint_dir,
        every_n_train_steps=train_cfg.save_every_steps,
        save_last=True,
        monitor="train/loss",
        mode="min",
        save_top_k=train_cfg.save_total_limit,
        filename="step-{step}",
    )
    
    trainer = L.Trainer(
        accelerator="gpu",
        devices=train_cfg.num_gpus,
        precision=train_cfg.precision,
        max_steps=train_cfg.max_steps,
        accumulate_grad_batches=train_cfg.gradient_accumulation_steps,
        gradient_clip_val=train_cfg.max_grad_norm,
        enable_checkpointing=True,
        callbacks=[checkpoint_cb],
        logger=loggers,
        log_every_n_steps=train_cfg.log_every_steps,
        deterministic=train_cfg.deterministic,
        enable_progress_bar=True,
    )
    
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    # Script is set up but not executed automatically.
    main()
