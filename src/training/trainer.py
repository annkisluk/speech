
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Optional, Callable
from pathlib import Path
from tqdm import tqdm
import json
import time

from ..utils.audio import si_snr_loss
from ..evaluation.metrics import MetricsCalculator


class Trainer:

    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: str = 'cuda',
        scheduler: Optional[any] = None,
        use_amp: bool = False,
        log_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        max_grad_norm: float = 5.0
    ):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.max_grad_norm = max_grad_norm
        
        # Directories
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Metrics calculator
        self.metrics_calc = MetricsCalculator(
            sample_rate=8000,
            metrics=['si_snr', 'sdr']
        )
        
        # AMP scaler
        if use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        max_grad_norm: float = 5.0,
        log_every_n_steps: int = 100
    ) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (noisy, clean, lengths, info) in enumerate(pbar):
            # Move to device
            noisy = noisy.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass (with or without AMP)
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    enhanced = self.model(noisy)
                    loss = si_snr_loss(enhanced, clean)
                
                # Backward with scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                enhanced = self.model(noisy)
                loss = si_snr_loss(enhanced, clean)
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Logging
            if self.global_step % log_every_n_steps == 0:
                avg_loss = total_loss / num_batches
                print(f"  Step {self.global_step}: Loss = {avg_loss:.4f}")
        
        # Epoch metrics
        avg_loss = total_loss / num_batches
        
        return {
            'train_loss': avg_loss,
            'epoch': self.current_epoch
        }
    
    def _forward_with_chunking(
        self,
        noisy: torch.Tensor,
        chunk_size: int,
        chunk_overlap: int,
        model: Optional[nn.Module] = None
    ) -> torch.Tensor:
        fwd_model = model if model is not None else self.model
        if noisy.shape[0] != 1:
            return fwd_model(noisy)

        total_len = noisy.shape[-1]
        if total_len <= chunk_size:
            return fwd_model(noisy)

        step = chunk_size - chunk_overlap
        if step <= 0:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        device = noisy.device
        # Use Hann window for smooth blending (prevents clicks)
        window = torch.hann_window(chunk_size, device=device)
        window = window.view(1, 1, -1)

        out_len = ((total_len - 1) // step) * step + chunk_size
        acc = torch.zeros((1, 1, out_len), device=device)
        weight = torch.zeros((1, 1, out_len), device=device)

        for start in range(0, out_len, step):
            if start >= total_len:
                break
            end = start + chunk_size
            chunk = noisy[..., start:min(end, total_len)]
            if chunk.shape[-1] < chunk_size:
                pad_len = chunk_size - chunk.shape[-1]
                chunk = F.pad(chunk, (0, pad_len))

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    enhanced_chunk = fwd_model(chunk)
            else:
                enhanced_chunk = fwd_model(chunk)

            acc[..., start:end] += enhanced_chunk * window
            weight[..., start:end] += window

        enhanced = acc / weight.clamp_min(1e-8)
        return enhanced[..., :total_len]

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        compute_metrics: bool = True,
        metrics_max_items: int = 50,
        chunk_size: int = 32000,
        chunk_overlap: int = 8000
    ) -> Dict[str, float]:
        # Unwrap DataParallel for validation 
        val_model = self.model.module if hasattr(self.model, 'module') else self.model
        val_model.eval()

        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        total_loss = 0.0
        num_batches = 0
        
        all_metrics = [] if compute_metrics else None
        metric_items = 0
        
        pbar = tqdm(val_loader, desc="Validation")
        
        for noisy, clean, lengths, info in pbar:
            # Use cuda:0 only for validation (unwrapped model)
            noisy = noisy.to('cuda:0', non_blocking=True)
            clean = clean.to('cuda:0', non_blocking=True)
            
            # Forward pass
            if noisy.shape[-1] > chunk_size:
                enhanced = self._forward_with_chunking(noisy, chunk_size, chunk_overlap, model=val_model)
            else:
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        enhanced = val_model(noisy)
                else:
                    enhanced = val_model(noisy)
            
            # Loss
            loss = si_snr_loss(enhanced, clean)
            total_loss += loss.item()
            num_batches += 1
            
            # Compute metrics
            if compute_metrics and metric_items < metrics_max_items:
                remaining = metrics_max_items - metric_items
                batch_size = noisy.shape[0]
                take = min(remaining, batch_size)

                enhanced_cpu = enhanced.detach().float().cpu()[:take]
                clean_cpu = clean.detach().float().cpu()[:take]
                lengths_cpu = lengths[:take].cpu()

                batch_metrics = self.metrics_calc.calculate_batch(
                    enhanced_cpu, clean_cpu, lengths_cpu
                )

                for i in range(take):
                    sample_metrics = {
                        metric: values[i]
                        for metric, values in batch_metrics.items()
                    }
                    all_metrics.append(sample_metrics)

                metric_items += take
        
        # Aggregate metrics
        avg_loss = total_loss / num_batches
        results = {'val_loss': avg_loss}
        
        if compute_metrics and all_metrics:
            aggregated = self.metrics_calc.aggregate_metrics(all_metrics)
            results.update(aggregated)
        
        return results
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        early_stopping_patience: int = 10,
        save_every_n_epochs: int = 5,
        validate_every_n_epochs: int = 1
    ) -> Dict[str, list]:
        print(f"Starting Training")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        start_epoch = 0
        skip_training = False
        if self.global_step > 0:
            if not getattr(self, '_epoch_validated', True):
                start_epoch = self.current_epoch
                skip_training = True
                print(f"Resuming epoch {self.current_epoch + 1}: skipping training, running pending validation")
            else:
                start_epoch = self.current_epoch + 1
            if start_epoch >= num_epochs:
                print(
                    f"Resumed from epoch {self.current_epoch}. "
                    f"Training already complete (num_epochs={num_epochs})."
                )
                return history

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            if skip_training:
                skip_training = False
            else:
                # Train epoch
                train_metrics = self.train_epoch(train_loader, max_grad_norm=getattr(self, 'max_grad_norm', 5.0))
                history['train_loss'].append(train_metrics['train_loss'])
                
                # Get learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                history['learning_rate'].append(current_lr)
                
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
                print(f"  Learning Rate: {current_lr:.6f}")
                
                # Save checkpoint BEFORE validation
                self._epoch_validated = False
                if (epoch + 1) % save_every_n_epochs == 0:
                    self.save_checkpoint(
                        self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                    )
            
            # Validation
            if (epoch + 1) % validate_every_n_epochs == 0:
                val_metrics = self.validate(val_loader, compute_metrics=True)
                history['val_loss'].append(val_metrics['val_loss'])
                
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                if 'si_snr_mean' in val_metrics:
                    print(f"  Val SI-SNR: {val_metrics['si_snr_mean']:.2f} dB")
                
                # Learning rate scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val_loss'])
                    else:
                        self.scheduler.step()
                
                # Check for improvement
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.epochs_without_improvement = 0
                    
                    # Save best model
                    self.save_checkpoint(
                        self.checkpoint_dir / "best_model.pt",
                        val_metrics
                    )
                    print(" New best model saved!")
                else:
                    self.epochs_without_improvement += 1
                    print(f" No improvement for {self.epochs_without_improvement} epochs")
                
                # Early stopping
                if self.epochs_without_improvement >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
                
                # Re-save checkpoint with validated=True so resume skips to next epoch
                self._epoch_validated = True
                if (epoch + 1) % save_every_n_epochs == 0:
                    self.save_checkpoint(
                        self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                    )
            
            # Epoch time
            epoch_time = time.time() - epoch_start_time
            print(f"  Epoch time: {epoch_time:.2f}s")
            print()
        
        print(f"Training Complete")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(
            self.checkpoint_dir / "final_model.pt"
        )
        
        # Save training history
        history_path = self.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def save_checkpoint(
        self,
        path: Path,
        metrics: Optional[Dict] = None
    ):
        # Unwrap DataParallel
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'validated': getattr(self, '_epoch_validated', True)
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Load into unwrapped model to handle DataParallel prefix differences
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.epochs_without_improvement = checkpoint['epochs_without_improvement']
        self._epoch_validated = checkpoint.get('validated', False)
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded: {path}")
        print(f"  Epoch: {self.current_epoch + 1}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        if not self._epoch_validated:
            print(f"  Validation pending for epoch {self.current_epoch + 1}")


# Utility Functions

def setup_optimizer(
    model: nn.Module,
    learning_rate: float = 15e-5,
    weight_decay: float = 0.0,
    optimizer_type: str = 'adam'
) -> Optimizer:
    # Only optimize trainable parameters 
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    return optimizer


def setup_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = 'plateau',
    patience: int = 3,
    factor: float = 0.5
) -> Optional[any]:
    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5
        )
    elif scheduler_type is None or scheduler_type == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    return scheduler


