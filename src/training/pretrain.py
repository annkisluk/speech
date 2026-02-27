"""
Pre-training Script for Session 0

Trains the SepFormer backbone on multiple noise types.

Paper Reference: Section IV.A
- 40 epochs for pre-training
- Adam optimizer with lr=15e-5
- Batch size: 2
- Loss: SI-SNR
"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse

from ..models.sepformer import create_sepformer
from ..data.dataset import get_session_dataloaders
from ..training.trainer import Trainer, setup_optimizer, setup_scheduler
from ..utils.config import ProjectConfig, get_default_config


def train_pretrain(
    config: ProjectConfig,
    data_root: str = "data/final_data",
    resume_from: str = None
):
    """
    Train Session 0 (pre-training)
    
    Paper: "we obtain the pre-trained model Θ0 using D0 via the 
    scale-invariant source-to-noise ratio (SI-SNR) loss function"
    
    Args:
        config: Project configuration
        data_root: Root directory with session data
        resume_from: Path to checkpoint to resume from
    """
    print("\n" + "="*80)
    print("SESSION 0: PRE-TRAINING".center(80))
    print("="*80 + "\n")
    
    # Device
    device = config.training.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    print(f"Device: {device}")
    print(f"Data root: {data_root}")
    
    # Verify CUDA is working
    if device == 'cuda':
        n_gpus = torch.cuda.device_count()
        print(f"CUDA Devices Available: {n_gpus}")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB)")
    print()
    
    # Create model - USE LNA MODEL FOR PRE-TRAINING (not plain SepFormer)
    # This ensures adapter_layers get trained, not just sepformer.masking_network
    print("Creating LNA model for pre-training...")
    from ..models.lna_model import LNAModel
    
    model = LNAModel(
        n_basis=config.sepformer.N,
        kernel_size=config.sepformer.L,
        num_layers=config.sepformer.num_layers,
        nhead=config.sepformer.nhead,
        dim_feedforward=config.sepformer.d_ffn,
        dropout=config.sepformer.dropout,
        adapter_bottleneck_dim=config.adapter.bottleneck_dim,
        max_sessions=6
    )
    
    # Set to session 0 mode (no adapters, just train base transformer)
    model.set_training_mode(session_id=0)
    
    print(f"  Total parameters: {model.get_num_parameters():,}")
    print(f"  Trainable parameters: {model.get_num_parameters(trainable_only=True):,}")
    
    # Enable multi-GPU training with DataParallel
    if device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"\nUsing DataParallel with {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
        print(f"  Training will be parallelized across GPUs: {list(range(torch.cuda.device_count()))}")
    
    # Create dataloaders
    print("\nLoading Session 0 data...")
    train_loader, val_loader, test_loader = get_session_dataloaders(
        data_root=data_root,
        session_id=0,
        batch_size_train=config.data.train_batch_size,
        batch_size_val=config.data.val_batch_size,
        batch_size_test=config.data.test_batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        sample_rate=config.data.sample_rate
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Setup optimizer
    print("\nSetting up optimizer...")
    optimizer = setup_optimizer(
        model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        optimizer_type=config.training.optimizer
    )
    
    print(f"  Optimizer: {config.training.optimizer}")
    print(f"  Learning rate: {config.training.learning_rate}")
    
    # Setup scheduler
    scheduler = None
    if config.training.use_scheduler:
        print(f"\nSetting up scheduler...")
        scheduler = setup_scheduler(
            optimizer,
            scheduler_type=config.training.scheduler_type,
            patience=config.training.patience,
            factor=config.training.factor
        )
        print(f"  Scheduler: {config.training.scheduler_type}")
    
    # Create trainer
    checkpoint_dir = Path(config.checkpoint_dir) / "session0_pretrain"
    log_dir = Path(config.log_dir) / "session0_pretrain"
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        use_amp=config.training.use_amp,
        log_dir=str(log_dir),
        checkpoint_dir=str(checkpoint_dir),
        max_grad_norm=config.training.max_grad_norm
    )
    
    # Verify model is on correct device
    model_for_check = trainer.model.module if isinstance(trainer.model, torch.nn.DataParallel) else trainer.model
    print(f"\n✓ Model device: {next(model_for_check.parameters()).device}")
    print(f"✓ Training will use: {trainer.device}")
    if isinstance(trainer.model, torch.nn.DataParallel):
        print(f"✓ Multi-GPU: Enabled ({torch.cuda.device_count()} GPUs)")
    
    # Resume from checkpoint if specified
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        trainer.load_checkpoint(Path(resume_from))
    
    # Train
    print("\n" + "="*80)
    print("STARTING TRAINING".center(80))
    print("="*80 + "\n")
    
    print(f"Training for {config.training.pretrain_epochs} epochs")
    print(f"Batch size: {config.data.train_batch_size}")
    print(f"Gradient clipping: {config.training.max_grad_norm}")
    print(f"Early stopping patience: {config.training.early_stopping_patience}")
    print()
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.pretrain_epochs,
        early_stopping_patience=config.training.early_stopping_patience,
        save_every_n_epochs=config.training.save_every_n_epochs,
        validate_every_n_epochs=config.training.validate_every_n_epochs
    )
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET".center(80))
    print("="*80 + "\n")
    
    test_metrics = trainer.validate(test_loader, compute_metrics=True)
    
    print("Test Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save final model in LNA format for incremental learning
    print("\nSaving pre-trained LNA model for incremental learning...")
    
    # Model is already LNA format - just save it directly
    # Handle DataParallel: unwrap model if needed
    lna_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    lna_model.is_pretrained = True
    
    # Save
    lna_checkpoint_path = checkpoint_dir / "lna_pretrained.pt"
    lna_model.save_checkpoint(
        str(lna_checkpoint_path),
        session_id=0,
        test_metrics=test_metrics
    )
    
    print(f"✓ Pre-trained LNA model saved: {lna_checkpoint_path}")
    print("\nReady for incremental training!")
    
    return history, test_metrics


def main():
    """Main function for CLI"""
    parser = argparse.ArgumentParser(description="Pre-train SepFormer (Session 0)")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/final_data",
        help="Root directory with session data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = ProjectConfig.from_yaml(args.config)
    else:
        config = get_default_config()
    
    # Override device if specified
    if args.device:
        config.training.device = args.device
    
    # Train
    history, metrics = train_pretrain(
        config=config,
        data_root=args.data_root,
        resume_from=args.resume
    )
    
    print("\n✓ Pre-training complete!")


if __name__ == "__main__":
    main()