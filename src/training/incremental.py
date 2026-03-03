
import torch
from pathlib import Path
import argparse
import numpy as np

from ..models.lna_model import LNAModel
from ..data.dataset import get_session_dataloaders, MultiSessionDataset, create_dataloader
from ..training.trainer import Trainer, setup_optimizer, setup_scheduler
from ..selectors.noise_selector import create_selector
from ..utils.config import ProjectConfig, get_default_config
from ..utils.audio import extract_features_for_clustering


def train_incremental_session(
    config: ProjectConfig,
    session_id: int,
    pretrained_model_path: str,
    data_root: str = "data/final_data",
    selector_path: str = None,
    resume_from: str = None
):
    if session_id == 0:
        raise ValueError("Use pretrain.py for session 0")
    
    print(f"SESSION {session_id}: INCREMENTAL TRAINING")
    
    # Device
    device = config.training.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    print(f"Device: {device}")
    print(f"Session ID: {session_id}")
    print(f"Data root: {data_root}")
    
    # Verify CUDA is working
    if device == 'cuda':
        n_gpus = torch.cuda.device_count()
        print(f"CUDA Devices Available: {n_gpus}")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB)")
    print()
    
    # Load pre-trained model
    print("Loading pre-trained LNA model")
    model = LNAModel(
        n_basis=config.sepformer.N,
        kernel_size=config.sepformer.L,
        num_layers=config.sepformer.num_layers,
        num_blocks=config.sepformer.num_blocks,
        nhead=config.sepformer.nhead,
        dim_feedforward=config.sepformer.d_ffn,
        dropout=config.sepformer.dropout,
        adapter_bottleneck_dim=config.adapter.bottleneck_dim,
        max_sessions=6
    )
    
    # For incremental sessions > 1, recreate previous sessions' adapters
    # before loading checkpoint so state_dict keys match
    if session_id > 1:
        print(f"  Recreating adapters for sessions 1 to {session_id - 1}...")
        for prev_session in range(1, session_id):
            model.add_new_session(
                session_id=prev_session,
                bottleneck_dim=config.adapter.bottleneck_dim
            )
    
    checkpoint = model.load_checkpoint(pretrained_model_path)
    print(f"Loaded from: {pretrained_model_path}")
    
    # Add new session adapters and decoder
    print(f"\nAdding adapters and decoder for session {session_id}")
    model.add_new_session(
        session_id=session_id,
        bottleneck_dim=config.adapter.bottleneck_dim
    )
    
    # Set training mode (freeze backbone, train new adapters/decoder)
    print("\nConfiguring training mode")
    model.set_training_mode(
        session_id=session_id,
        freeze_backbone=config.incremental.freeze_backbone,
        freeze_previous_adapters=config.incremental.freeze_previous_adapters,
        freeze_previous_decoders=config.incremental.freeze_previous_decoders
    )
    
    adapter_info = model.get_adapter_info()
    print(f"  Adapter parameters: {adapter_info['adapter_parameters']:,}")
    print(f"  Adapter percentage: {adapter_info['adapter_percentage']:.2f}%")
    
    # Enable multi-GPU training with DataParallel (after model configuration)
    if device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"\nUsing DataParallel with {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
        print(f"  Training will be parallelized across GPUs: {list(range(torch.cuda.device_count()))}")
    
    # Create dataloaders
    print(f"\nLoading Session {session_id} data.")
    train_loader, _, _ = get_session_dataloaders(
        data_root=data_root,
        session_id=session_id,
        batch_size_train=config.data.train_batch_size,
        batch_size_val=config.data.val_batch_size,
        batch_size_test=config.data.test_batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        sample_rate=config.data.sample_rate
    )

    # Cumulative val/test loaders: all sessions 0..session_id (paper: Z^{1,...,t})
    cumulative_session_ids = list(range(0, session_id + 1))
    print(f"\nBuilding cumulative val/test sets for sessions {cumulative_session_ids}...")

    cum_val_dataset = MultiSessionDataset(
        data_root=data_root,
        session_ids=cumulative_session_ids,
        split="val",
        sample_rate=config.data.sample_rate,
        max_length=4 * config.data.sample_rate  # Cap at 4s — same as single-session val
    )
    val_loader = create_dataloader(
        cum_val_dataset,
        batch_size=config.data.val_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    cum_test_dataset = MultiSessionDataset(
        data_root=data_root,
        session_ids=cumulative_session_ids,
        split="test",
        sample_rate=config.data.sample_rate,
        max_length=4 * config.data.sample_rate  # Cap at 4s — same as single-session test
    )
    test_loader = create_dataloader(
        cum_test_dataset,
        batch_size=config.data.test_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Cumulative val batches: {len(val_loader)} ({len(cum_val_dataset)} samples)")
    print(f"  Cumulative test batches: {len(test_loader)} ({len(cum_test_dataset)} samples)")
    
    # Setup optimizer 
    optimizer = setup_optimizer(
        model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        optimizer_type=config.training.optimizer
    )
    
    # Get trainable parameter count 
    model_for_check = model.module if isinstance(model, torch.nn.DataParallel) else model
    print(f"  Trainable parameters: {model_for_check.get_num_parameters(trainable_only=True):,}")
    
    #Verify backbone is actually frozen
    print("\n  Verifying freezing configuration:")
    backbone_frozen = not any(p.requires_grad for p in model_for_check.sepformer.parameters())
    print(f"  Backbone frozen: {backbone_frozen}")
    
    # Count trainable params by type
    adapter_trainable = sum(
        p.numel() for block in model_for_check.dpt_blocks
        for p in block.parameters() if p.requires_grad
    )
    decoder_trainable = sum(p.numel() for p in model_for_check.decoders[f'session_{session_id}'].parameters() if p.requires_grad)
    print(f" Trainable adapter params: {adapter_trainable:,}")
    print(f" Trainable decoder params: {decoder_trainable:,}")
    
    # Setup scheduler
    scheduler = None
    if config.training.use_scheduler:
        scheduler = setup_scheduler(
            optimizer,
            scheduler_type=config.training.scheduler_type,
            patience=config.training.patience,
            factor=config.training.factor
        )
    
    # Create trainer
    checkpoint_dir = Path(config.checkpoint_dir) / f"session{session_id}_incremental"
    log_dir = Path(config.log_dir) / f"session{session_id}_incremental"
    
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
    print(f"\nModel device: {next(model_for_check.parameters()).device}")
    print(f"Training will use: {trainer.device}")
    if isinstance(trainer.model, torch.nn.DataParallel):
        print(f"Multi-GPU: Enabled ({torch.cuda.device_count()} GPUs)")
    
    # Resume from checkpoint if specified
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        trainer.load_checkpoint(Path(resume_from))

    # Train
    print("STARTING INCREMENTAL TRAINING")
    
    print(f"Training for {config.training.incremental_epochs} epochs")
    print(f"Frozen: Backbone (encoder + masking network)")
    print(f"Frozen: Previous adapters and decoders")
    print(f"Training: New adapters and decoder for session {session_id}")
    print()
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.incremental_epochs,
        early_stopping_patience=config.training.early_stopping_patience,
        save_every_n_epochs=config.training.save_every_n_epochs,
        validate_every_n_epochs=config.training.validate_every_n_epochs
    )
    
    # Save model
    model_checkpoint_path = checkpoint_dir / f"lna_session{session_id}.pt"
    # Handle DataParallel
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    model_to_save.save_checkpoint(
        str(model_checkpoint_path),
        session_id=session_id
    )
    
    # Train Noise Selector
    print("TRAINING NOISE SELECTOR")
    
    # Load or create selector
    if selector_path and Path(selector_path).exists():
        print(f"Loading existing selector from {selector_path}")
        selector = create_selector(
            selector_type=config.selector.selector_type,
            feature_dim=config.sepformer.N
        )
        selector.load(selector_path)
    else:
        print(f"Creating new {config.selector.selector_type} selector")
        # Build kwargs based on selector type
        selector_kwargs = {'feature_dim': config.sepformer.N}
        if config.selector.selector_type == 'kmeans':
            selector_kwargs['n_clusters'] = config.selector.n_clusters
        elif config.selector.selector_type == 'meanshift':
            if config.selector.bandwidth is not None:
                selector_kwargs['bandwidth'] = config.selector.bandwidth
        elif config.selector.selector_type == 'gmm':
            selector_kwargs['n_components'] = config.selector.n_components
            selector_kwargs['covariance_type'] = config.selector.covariance_type
        
        selector = create_selector(
            selector_type=config.selector.selector_type,
            **selector_kwargs
        )
    
    # Extract features from training data
    print(f"\nExtracting features for session {session_id}...")
    model.eval()
    
    # Get the actual model 
    model_for_features = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    all_features = []
    with torch.no_grad():
        for noisy, clean, lengths, info in train_loader:
            noisy = noisy.to(device)
            
            # Extract encoder features
            features = model_for_features.get_encoder_features(noisy)  # [B, N, L]
            
            # Length-aware mean pooling: only average over real frames, not padding
            # Encoder: Conv1d(stride=8, kernel=16, no padding) -> L_enc = (T-16)//8 + 1
            enc_lengths = (lengths - 16) // 8 + 1  # [B]
            B_feat, N_feat, L_feat = features.shape
            for b in range(B_feat):
                real_len = min(enc_lengths[b].item(), L_feat)
                feat = features[b, :, :real_len].mean(dim=1)  # [N]
                all_features.append(feat.cpu().numpy().reshape(1, -1))
    
    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)  # [N_samples, N]
    print(f"  Extracted {all_features.shape[0]} feature vectors")
    print(f"  Feature dimension: {all_features.shape[1]}")
    
    # Fit selector for this session
    print(f"\nFitting selector for session {session_id}...")
    selector.fit_session(all_features, session_id=session_id)
    
    # Save selector
    selector_save_path = checkpoint_dir / f"selector_upto_session{session_id}.pkl"
    selector.save(str(selector_save_path))
    print(f"✓ Selector saved: {selector_save_path}")
    
    # Test selector accuracy on validation set
    print("\nTesting selector on validation set...")
    correct = 0
    total = 0
    
    # Get the actual model
    model_for_features = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    model.eval()
    with torch.no_grad():
        for noisy, clean, lengths, info in val_loader:
            noisy = noisy.to(device)
            features = model_for_features.get_encoder_features(noisy)
            B_feat, N_feat, L_feat = features.shape
            enc_lengths = (lengths - 16) // 8 + 1
            
            for i in range(len(noisy)):
                real_len = min(enc_lengths[i].item(), L_feat)
                feat = features[i, :, :real_len].mean(dim=1).cpu().numpy()
                predicted = selector.predict(feat)
                # For this session, all samples should be predicted as this session
                if predicted == session_id:
                    correct += 1
                total += 1
    
    accuracy = 100 * correct / total
    print(f"  Selector accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    print(f"\n Session {session_id} training complete")
    
    return history, model_checkpoint_path, selector_save_path


def train_all_incremental_sessions(
    config: ProjectConfig,
    pretrained_model_path: str,
    data_root: str = "data/final_data",
    session_ids: list = [1, 2, 3, 4, 5],
    resume_if_exists: bool = True
):
    current_model_path = pretrained_model_path
    current_selector_path = None
    
    results = {}
    
    # Selector is fitted only on incremental sessions (1..t).    
    for session_id in session_ids:
        print(f"# TRAINING SESSION {session_id}".center(78, ' ') + " #")
        
        resume_from = None
        if resume_if_exists:
            checkpoint_dir = Path(config.checkpoint_dir) / f"session{session_id}_incremental"
            checkpoint_paths = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if checkpoint_paths:
                def _epoch_num(path: Path) -> int:
                    stem = path.stem
                    return int(stem.split("checkpoint_epoch_")[-1])
                resume_from = str(max(checkpoint_paths, key=_epoch_num))

        history, model_path, selector_path = train_incremental_session(
            config=config,
            session_id=session_id,
            pretrained_model_path=current_model_path,
            data_root=data_root,
            selector_path=current_selector_path,
            resume_from=resume_from
        )
        
        # Update paths for next session
        current_model_path = model_path
        current_selector_path = selector_path
        
        results[f'session_{session_id}'] = {
            'model_path': str(model_path),
            'selector_path': str(selector_path),
            'history': history
        }
    
    print("ALL INCREMENTAL SESSIONS COMPLETE")
    
    print("Trained sessions:")
    for session_id in session_ids:
        info = results[f'session_{session_id}']
        print(f"  Session {session_id}:")
        print(f"    Model: {info['model_path']}")
        print(f"    Selector: {info['selector_path']}")
    
    return results


def main():
    #Main function for CLI
    parser = argparse.ArgumentParser(description="Incremental Training (Sessions 1-5)")
    parser.add_argument(
        "--session_id",
        type=int,
        required=True,
        choices=[1, 2, 3, 4],
        help="Session ID to train"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        required=True,
        help="Path to pre-trained LNA model"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/final_data",
        help="Root directory with session data"
    )
    parser.add_argument(
        "--selector",
        type=str,
        default=None,
        help="Path to existing selector (for continuing from previous session)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to use"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all incremental sessions (1-5) sequentially"
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = ProjectConfig.from_yaml(args.config)
    else:
        config = get_default_config()
    
    # Override device
    if args.device:
        config.training.device = args.device
    
    # Train
    if args.all:
        results = train_all_incremental_sessions(
            config=config,
            pretrained_model_path=args.pretrained_model,
            data_root=args.data_root
        )
    else:
        history, model_path, selector_path = train_incremental_session(
            config=config,
            session_id=args.session_id,
            pretrained_model_path=args.pretrained_model,
            data_root=args.data_root,
            selector_path=args.selector
        )
    
    print("\n Incremental training complete")


if __name__ == "__main__":
    main()